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

from agent.prompts import NO_MATCH_RESPONSE, SEMANTIC_QA_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_openai import ChatOpenAI
    from langchain_postgres import PGVectorStore


logger = logging.getLogger(__name__)


def make_retrieve_node(vector_store: PGVectorStore) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a retrieve node that fetches docs + cosine scores from pgvector."""

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        results: list[tuple[Document, float]] = (
            vector_store.similarity_search_with_relevance_scores(state["question"], k=4)
        )
        if results:
            docs, scores = zip(*results, strict=False)
            return {"docs": list(docs), "scores": list(scores)}
        return {"docs": [], "scores": []}

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
        prompt = SEMANTIC_QA_SYSTEM_PROMPT.format(
            question=state["question"],
            context=context,
        )
        response = llm.invoke([SystemMessage(content=prompt)])
        sources = [
            {
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("source_url", ""),
            }
            for doc in docs
            if doc.metadata.get("source_url")
        ]
        return {"answer": str(response.content), "sources": sources}

    return generate
