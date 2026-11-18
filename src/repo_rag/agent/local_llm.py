"""Local chat model on one CUDA GPU, configured as evaluated in the Phase 7 bake-offs.

bf16, no quantization, SDPA attention, greedy decoding, batch size 1, the model's native chat
template (refused if its hash differs from `Settings.llm_chat_template_sha256`), the input and
output token caps from `Settings`. Importing this module loads nothing; `LocalChatModel()` needs
`torch`, `transformers` and a GPU with room for the weights (about 31 GB for the 14B model).
"""

from __future__ import annotations

import gc
import hashlib
import time
from typing import Any

from repo_rag.agent.bakeoff import BudgetFailure, Generation
from repo_rag.config import Settings


def trim_terminal_eos(tokens: list[int], eos_ids: list[int | None]) -> list[int]:
    """Remove exactly one terminal end-of-sequence token, never reasoning or malformed text."""
    return tokens[:-1] if tokens and tokens[-1] in eos_ids else tokens


def check_input_tokens(count: int, limit: int) -> None:
    if count > limit:
        raise BudgetFailure("Input token cap")


class LocalChatModel:
    def __init__(self, settings: Settings | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings = settings or Settings()
        name, revision = self.settings.llm_model, self.settings.llm_model_revision
        if not name or not revision:
            raise ValueError("Set llm_model and llm_model_revision (pinned) before loading")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required: the evaluated setup is bf16 on one GPU")
        self.torch = torch
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revision, trust_remote_code=False, token=False
        )
        expected = self.settings.llm_chat_template_sha256
        template = self.tokenizer.chat_template
        if expected and (
            not isinstance(template, str)
            or hashlib.sha256(template.encode()).hexdigest() != expected
        ):
            raise RuntimeError("Chat template differs from the evaluated revision")
        factory: Any = AutoModelForCausalLM  # lazy auto-factory has an inaccurate stub
        self.model = (
            factory.from_pretrained(
                name,
                revision=revision,
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
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            **self.settings.llm_template_arguments,
        )
        count = inputs["input_ids"].shape[-1]
        check_input_tokens(count, self.settings.agent_max_input_tokens)
        inputs = inputs.to("cuda")
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=self.settings.agent_max_new_tokens,
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
        tokens = trim_terminal_eos(tokens, eos if isinstance(eos, list) else [eos])
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        return Generation(text, total, elapsed, count, total >= self.settings.agent_max_new_tokens)

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
