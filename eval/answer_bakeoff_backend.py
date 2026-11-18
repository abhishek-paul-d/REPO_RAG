"""Colab-only backends. Imports are cheap; explicit constructors load local open weights.

Do not instantiate these during offline tests or capsule preparation.
"""

from __future__ import annotations

import gc
import hashlib
import io
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings

from eval.answer_bakeoff_io import load
from repo_rag.agent.bakeoff import BudgetFailure, Generation
from repo_rag.agent.bakeoff_tools import BoundTools
from repo_rag.config import Settings
from repo_rag.history.metadata_db import MetadataDB
from repo_rag.retrieval.retrievers import HybridRetriever
from repo_rag.store.bm25 import Bm25Index, bm25_path
from repo_rag.store.symbol_table import SymbolTable
from repo_rag.store.vectorstore import VectorStore


class PinnedQueryEmbeddings(Embeddings):
    def __init__(self, manifest: dict) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        self.manifest = manifest
        path = snapshot_download(
            manifest["embedding_model"],
            revision=manifest["embedding_revision"],
            token=False,
            ignore_patterns=["*.bin", "onnx/*", "openvino/*"],
        )
        self.model = SentenceTransformer(
            path,
            device="cuda",
            model_kwargs={"dtype": torch.float32},
            processor_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = manifest["embedding_max_tokens"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Frozen corpus vectors must not be re-embedded")

    def embed_query(self, text: str) -> list[float]:
        from repo_rag.store.embeddings import truncate_and_normalise

        vectors = self.model.encode(
            [text],
            prompt=self.manifest["embedding_query_instruction"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return truncate_and_normalise(vectors, self.manifest["embedding_dims"])[0].tolist()


class Runtime:
    def __init__(self, capsule: Path, work: Path) -> None:
        self.manifest = load(capsule / "index_manifest.json")
        self.recipe = load(capsule / "git_recipe.json")
        self.checkouts = work / "repos"
        self.settings = Settings(
            data_dir=work / "index",
            embedding_dtype="float32",
            compute_device="cuda",
            retrieval_k=50,
            reranker_mode="off",
            evidence_expansion="none",
            dense_weight=1.0,
            bm25_weight=0.3,
            rrf_k=60,
            embedding_dims=self.manifest["embedding_dims"],
        )
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(capsule / "metadata.db", self.settings.metadata_db_path)
        self.embeddings = PinnedQueryEmbeddings(self.manifest)
        # Synthetic warmup, with no dataset question or label, before answer-model load/timing.
        self.embeddings.embed_query("Locate the public interface.")
        self.store = VectorStore(self.settings, self.embeddings)
        for short, repo in (("requests", "psf/requests"), ("flask", "pallets/flask")):
            data = load(capsule / "code" / f"{short}.json")
            vectors = np.load(
                io.BytesIO((capsule / "code" / f"{short}.npy").read_bytes()), allow_pickle=False
            )
            self.store.replace_records(
                "code", repo, data["ids"], data["texts"], data["metas"], vectors
            )
            Bm25Index.build(self.store.repo_documents("code", repo)).save(
                bm25_path(self.settings.bm25_dir, repo, "code")
            )
        self.retriever = HybridRetriever(self.store, self.settings)
        self.db = MetadataDB(self.settings.metadata_db_path)
        self.symbols = SymbolTable(self.settings.metadata_db_path)

    def tools(self, repo: str) -> BoundTools:
        return BoundTools(
            repo,
            self.checkouts / repo.replace("/", "__"),
            self.manifest["pins"][repo],
            self.settings,
            self.retriever,
            self.db,
            self.symbols,
        )

    def close(self) -> None:
        self.db.connection.close()
        self.symbols.connection.close()


class HFAnswerModel:
    def __init__(self, candidate: dict, limits: dict) -> None:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if transformers.__version__ != "4.57.6":
            raise RuntimeError("Use the frozen Transformers 4.57.6 runtime")
        self.torch, self.limits = torch, limits
        started = time.perf_counter()
        config_path = hf_hub_download(
            candidate["model_id"], "config.json", revision=candidate["revision"], token=False
        )
        self.config_sha256 = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        if self.config_sha256 != candidate["source_metadata_hashes"]["config.json"]:
            raise RuntimeError("Model config differs from approved revision")
        self.tokenizer = AutoTokenizer.from_pretrained(
            candidate["model_id"],
            revision=candidate["revision"],
            trust_remote_code=False,
            token=False,
        )
        template = self.tokenizer.chat_template
        if (
            not isinstance(template, str)
            or hashlib.sha256(template.encode()).hexdigest()
            != candidate["chat_template"]["sha256_utf8"]
        ):
            raise RuntimeError("Native template differs from approved revision")
        factory: Any = AutoModelForCausalLM  # Third-party lazy auto-factory has an inaccurate stub.
        self.model = (
            factory.from_pretrained(
                candidate["model_id"],
                revision=candidate["revision"],
                dtype=torch.bfloat16,
                trust_remote_code=False,
                token=False,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.template_kwargs = candidate["chat_template"]["arguments"]
        self.hardware = torch.cuda.get_device_name(0)
        self.device_bytes = torch.cuda.get_device_properties(0).total_memory

    def start_question(self) -> None:
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def finish_question(self) -> dict[str, Any]:
        self.torch.cuda.synchronize()
        return {
            "hardware": self.hardware,
            "total_device_bytes": self.device_bytes,
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
            "device_level_peak_sampled": False,
        }

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, **self.template_kwargs
        )
        count = inputs["input_ids"].shape[-1]
        if count > self.limits["max_input_tokens"]:
            raise BudgetFailure("Input token cap")
        inputs = inputs.to("cuda")
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=self.limits["max_new_tokens_per_turn"],
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = output[0, count:].tolist()
        total = len(tokens)
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        # Remove exactly a terminal EOS, never reasoning blocks or malformed text.
        if tokens and tokens[-1] in eos_ids:
            tokens = tokens[:-1]
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        return Generation(
            text, total, elapsed, count, total >= self.limits["max_new_tokens_per_turn"]
        )

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
