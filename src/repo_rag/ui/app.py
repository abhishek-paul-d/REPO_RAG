"""Streamlit chat UI over the API (`repo_rag.api.server`).

    uv run uvicorn repo_rag.api.server:app          # the backend, on localhost
    uv run streamlit run src/repo_rag/ui/app.py     # this UI

The UI holds no model token and loads no model: it lists repos with `GET /repos` and asks with the
streaming `POST /ask`. The backend runs retrieval, tools and the citation check on this machine and
reaches the answer model on the remote GPU (`notebooks/serve_model_on_colab.ipynb`, configured
there with `REPO_RAG_LLM_SERVER_URL` and `REPO_RAG_LLM_SERVER_TOKEN`). The API has no
authentication: keep it on localhost.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import streamlit as st

from repo_rag.config import Settings
from repo_rag.ui.api_client import ApiClient, ApiError

ClientFactory = Callable[[str], Any]

STATUS = {"": "answered", "parse": "bad model reply", "budget": "over a limit"}
STEPS = {
    "prestep": "Reading the question",
    "answer": "Searching and thinking (each model turn is a GPU call)",
    "check": "Checking citations",
    "remember": "Done",
}


def describe(result: dict[str, Any]) -> str:
    """One line under an answer: how it ended and whether its citations were checked."""
    if result["abstain"] and result["failure"] == "":
        return "The model said it could not answer from what it retrieved."
    if result["failure"]:
        return f"No answer ({STATUS.get(result['failure'], result['failure'])})."
    cites = result["citations"]
    if cites.get("checked", True):
        line = f"{cites['valid']} citation(s) match retrieved code"
        if cites["removed"]:
            line += f"; removed {len(cites['removed'])} that did not"
        if cites["attached"]:
            line += f"; added {len(cites['attached'])} the model had named without lines"
        if cites["uncited"]:
            line += "; the answer cites nothing"
        line += ". Citations show where the code is, not that the answer is right."
    else:
        line = (
            "Citations were not checked: no code or docs were retrieved for this answer "
            "(it came from commit, history, caller or count lookups)."
        )
    if result.get("unverified_shas"):
        shas = ", ".join(result["unverified_shas"])
        line += f" Commit id(s) in the answer that no tool returned: {shas}."
    return line


_SOURCE = re.compile(r"^(?P<path>\S+?):(?P<start>\d+)-(?P<end>\d+)(?:\s+(?P<rest>.*))?$")


def source_link(repo: str, sha: str, source: str) -> str | None:
    """Markdown for a `path:start-end symbol` source with a GitHub permalink at the indexed
    commit, or None when the source does not have that shape (then it is shown as plain text)."""
    match = _SOURCE.match(source.strip())
    if match is None or not sha:
        return None
    path, start, end, rest = match["path"], match["start"], match["end"], match["rest"]
    url = f"https://github.com/{repo}/blob/{sha}/{quote(path)}#L{start}-L{end}"
    line = f"[`{path}:{start}-{end}`]({url})"
    return f"{line} {rest}" if rest else line


def show_result(result: dict[str, Any], revision: tuple[str, str] | None = None) -> None:
    """`revision` is (owner/name, commit sha) of the index; with it, sources link to GitHub."""
    st.markdown(result["answer"])
    st.caption(describe(result))
    if result.get("error"):
        st.warning(result["error"])
    if result.get("raw_reply"):
        with st.expander("What the model said (the reply that failed)"):
            st.code(result["raw_reply"], language=None)
    if result.get("sources"):
        with st.expander(f"Retrieved code ({len(result['sources'])})"):
            for source in result["sources"]:
                link = source_link(*revision, source) if revision else None
                if link:
                    st.markdown(link)
                else:
                    st.code(source, language=None)
    if result.get("tool_calls"):
        with st.expander(f"Tool calls ({len(result['tool_calls'])})"):
            for call in result["tool_calls"]:
                st.code(f"{call['name']}({call['arguments']})", language="python")


def run(client_factory: ClientFactory = ApiClient, settings: Settings | None = None) -> None:
    settings = settings or Settings()
    st.set_page_config(page_title="Repository assistant", layout="wide")
    st.title("Repository assistant")
    with st.sidebar:
        st.header("API")
        url = st.text_input("URL", value=settings.api_url).strip()
        if st.button("New chat"):
            st.session_state.update(messages=[], thread=uuid.uuid4().hex)
    if not url:
        st.info("Enter the API URL, e.g. http://127.0.0.1:8000.")
        return
    client = client_factory(url)
    try:
        listing = client.repos()
    except ApiError as exc:
        st.error(f"Could not reach the API: {exc}")
        st.info("Start it with `uv run uvicorn repo_rag.api.server:app`.")
        return
    repos = {r["repo"]: r for r in listing["repos"] if r["searchable"]}
    with st.sidebar:
        repo = st.selectbox("Repository", sorted(repos)) if repos else None
        model = (settings.llm_model or "the answer model").rpartition("/")[2]
        st.caption(
            f"Answers are generated by {model} on the remote GPU. "
            "Commit, PR and issue questions need that history indexed."
        )
    if repo is None:
        st.info(
            "Nothing searchable is indexed. Build an index with `notebooks/index_on_colab.ipynb` "
            "and import the zip (`repo-rag import-index`, or `POST /repos/import`)."
        )
        return

    if st.session_state.get("repo") != (url, repo):
        st.session_state.update(repo=(url, repo), messages=[], thread=uuid.uuid4().hex)
    sha = repos[repo].get("commit_sha") or ""
    revision = (repo, sha) if sha else None
    st.sidebar.success(f"Connected to the API: {len(repos)} repo(s) searchable")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                show_result(message["result"], revision)
    question = st.chat_input(f"Ask about {repo}")
    if not question:
        return
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Working...", expanded=False) as status:

            def progress(node: str) -> None:
                status.update(label=STEPS.get(node, node))

            try:
                result = client.ask(repo, question, st.session_state["thread"], progress)
            except ApiError as exc:
                status.update(label="Failed", state="error")
                st.error(str(exc))
                return
            status.update(label="Done", state="complete")
        show_result(result, revision)
    st.session_state["messages"].append({"role": "assistant", "result": result})


if __name__ == "__main__":
    run()
