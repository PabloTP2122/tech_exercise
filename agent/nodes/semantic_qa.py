"""Semantic QA nodes: retrieve, relevance_gate (conditional edge fn), generate.

Public API (node factories)
---------------------------
make_retrieve_node(vector_store)            -> Callable[[dict[str, Any]], dict[str, Any]]
make_relevance_gate_fn(threshold, no_match) -> Callable[[dict[str, Any]], str]
make_generate_node(llm)                     -> Callable[[dict[str, Any]], dict[str, Any]]
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from agent.prompts import NO_MATCH_RESPONSE, SEMANTIC_QA_SYSTEM_PROMPT
from agent.retrieval import dedupe_chunks, keyword_search_chunks, rrf_fuse
from ingest.clean import clean_title
from ingest.load_vectorstore import wrap_as_data

if TYPE_CHECKING:
    import sqlalchemy as sa
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI
    from langchain_postgres import PGVectorStore


logger = logging.getLogger(__name__)


class _QAResult(BaseModel):
    """Structured-output schema for the semantic-QA generate node."""

    answer: str
    cited_urls: list[str] = []


def make_retrieve_node(
    vector_store: PGVectorStore,
    *,
    engine: sa.Engine | None = None,
    table_name: str = "chunks",
    top_k: int = 4,
    keyword_k: int = 4,
    max_per_article: int = 2,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a retrieve node: vector search, optional keyword fusion, dedupe.

    ``state["scores"]`` stays **vector-only** so the relevance gate and the
    no-match invariant are unaffected by hybrid search — keyword hits enrich
    and reorder the LLM context (``state["docs"]``) but never open the gate.

    Args:
        vector_store: pgvector store for dense retrieval.
        engine: SQLAlchemy engine for the full-text pass.  ``None`` disables
            hybrid search (pure-vector behaviour, used by offline tests).
        table_name: Chunks table name for the keyword SQL.
        top_k: Dense hits fetched per question.
        keyword_k: Full-text hits fused via RRF; ``0`` disables hybrid search.
        max_per_article: Context-diversity cap applied after fusion.
    """

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        results: list[tuple[Document, float]] = (
            vector_store.similarity_search_with_relevance_scores(state["question"], k=top_k)
        )
        vector_docs = [doc for doc, _score in results]
        scores = [score for _doc, score in results]

        keyword_docs: list[Document] = []
        if engine is not None and keyword_k > 0:
            keyword_docs = keyword_search_chunks(engine, table_name, state["question"], keyword_k)

        docs = rrf_fuse(vector_docs, keyword_docs) if keyword_docs else vector_docs
        docs = dedupe_chunks(docs, max_per_article=max_per_article)
        return {"docs": docs, "scores": scores}

    return retrieve


def make_relevance_gate_fn(
    threshold: float, no_match: str = NO_MATCH_RESPONSE
) -> Callable[[dict[str, Any]], str]:
    """Return a conditional-edge function that routes on relevance score.

    Returns ``"generate"`` when at least one retrieved doc meets ``threshold``,
    or ``"no_match"`` to terminate with the no-match response.

    Note: the calling graph must add a ``"no_match"`` edge to END that sets
    ``answer = no_match`` in state; this fn only returns the routing string.
    """

    def relevance_gate(state: dict[str, Any]) -> str:
        scores: list[float] = state.get("scores", [])
        logger.debug("relevance scores=%s threshold=%.2f", scores, threshold)
        if scores and max(scores) >= threshold:
            return "generate"
        return "no_match"

    return relevance_gate


NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


def make_no_match_node(no_match: str = NO_MATCH_RESPONSE) -> NodeFn:
    """Return a terminal node that sets answer to no_match and sources to []."""

    def no_match_node(_state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": no_match, "sources": []}

    return no_match_node


def make_generate_node(llm: ChatOpenAI) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a generate node that calls the LLM with retrieved context."""

    def generate(state: dict[str, Any]) -> dict[str, Any]:
        docs: list[Document] = state.get("docs", [])
        context = "\n\n".join(doc.page_content for doc in docs)
        # ADR-0008: {source_url: title} whitelist; context already wrapped at ingest.
        # ADR-0008: build whitelist; clean_title unescapes HTML entities (e.g. &amp; → &).
        whitelist: dict[str, str] = {
            doc.metadata["source_url"]: clean_title(doc.metadata.get("title", ""))
            for doc in docs
            if doc.metadata.get("source_url")
        }
        # ADR-0008: wrap the question so the model treats it as DATA, not instructions.
        prompt = SEMANTIC_QA_SYSTEM_PROMPT.format(
            question=wrap_as_data(state["question"]),
            context=context,
        )
        try:
            structured: Any = llm.with_structured_output(_QAResult)
            result: _QAResult = structured.invoke([SystemMessage(content=prompt)])
        except Exception:
            logger.warning("structured output failed; falling back to plain invoke")
            response = llm.invoke([SystemMessage(content=prompt)])
            sources = [{"title": t, "url": u} for u, t in whitelist.items()]
            return {"answer": str(response.content), "sources": sources}
        # No-match guard: no sources for the canonical no-match sentence.
        if result.answer.strip() == NO_MATCH_RESPONSE:
            return {"answer": result.answer, "sources": []}
        # Whitelist filter: drop any URL the model invented that isn't in retrieved docs.
        cited = [u for u in result.cited_urls if u in whitelist]
        # Parity fallback: guarantee ≥1 source (reference-link-parity rule, Deviation #15).
        if not cited:
            cited = list(whitelist)
        sources = [{"title": whitelist[u], "url": u} for u in cited]
        return {"answer": result.answer, "sources": sources}

    return generate
