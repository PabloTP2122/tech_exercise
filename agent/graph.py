"""Bitovi blog RAG agent — compiled LangGraph StateGraph.

Graph topology (ADR-0006):

    classify
        └─ conditional_edges(route_query)
             ├─ "retrieve"       → retrieve → conditional_edges(relevance_gate_fn)
             │                          ├─ "generate"  → generate → END
             │                          └─ "no_match"  → no_match_node → END
             ├─ "sql_count"      → END
             ├─ "sql_enumerate"  → END
             └─ "hybrid_recency" → END

Only ``semantic_qa`` calls the LLM.  All other paths are deterministic.
Module-level ``graph`` is what ``langgraph.json`` loads for Studio.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import sqlalchemy as sa
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVectorStore
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agent.classifier import classify
from agent.nodes.catalog import make_sql_count_node, make_sql_enumerate_node
from agent.nodes.recency import make_hybrid_recency_node
from agent.nodes.semantic_qa import (
    make_generate_node,
    make_no_match_node,
    make_relevance_gate_fn,
    make_retrieve_node,
)
from agent.routes.route import route_query
from api.config import get_settings
from ingest.catalog_db import list_category_slugs
from ingest.load_vectorstore import METADATA_COLUMN_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    question: str
    query_type: Literal["semantic_qa", "enumeration", "count", "recency"]
    category_slug: str | None
    docs: list[Document]
    scores: list[float]
    answer: str
    sources: list[dict[str, str]]  # {title, url} — maps to api/models.py SourceRef


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    *,
    engine: sa.Engine | None = None,
    vector_store: PGVectorStore | None = None,
    llm: ChatOpenAI | None = None,
    known_slugs: list[str] | None = None,
) -> StateGraph[AgentState]:
    """Build and return an uncompiled StateGraph.

    All dependencies are injectable for offline testing (pass mocks).
    When called with no arguments (production path) deps are created from
    ``get_settings()``.

    Args:
        engine: SQLAlchemy engine for catalog queries.  Created from
            ``settings.database_url`` when ``None``.
        vector_store: pgvector store for semantic retrieval.  Created from
            settings when ``None``.
        llm: ChatOpenAI instance for the generate node.  Created from
            settings when ``None``.
        known_slugs: Category slugs for the classify node.  Loaded from DB
            when ``None`` (requires a live DB).

    Returns:
        Uncompiled :class:`~langgraph.graph.StateGraph`.
    """
    settings = get_settings()

    # --- build deps that were not injected ---
    if engine is None:
        engine = sa.create_engine(settings.database_url)

    if vector_store is None:
        from langchain_openai import OpenAIEmbeddings
        from langchain_postgres import PGEngine, PGVectorStore

        pg_engine = PGEngine.from_connection_string(url=settings.database_url)
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        vector_store = PGVectorStore.create_sync(
            engine=pg_engine,
            embedding_service=embeddings,
            table_name=settings.collection_name,
            metadata_columns=METADATA_COLUMN_NAMES,
        )

    if llm is None:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    slugs = known_slugs if known_slugs is not None else list_category_slugs(engine)

    # --- node factories ---
    retrieve_node = make_retrieve_node(
        vector_store,
        engine=engine,
        table_name=settings.collection_name,
        top_k=settings.retrieval_top_k,
        keyword_k=settings.keyword_top_k,
        max_per_article=settings.max_chunks_per_article,
    )
    generate_node = make_generate_node(llm)
    no_match_node = make_no_match_node()
    sql_count_node = make_sql_count_node(engine)
    sql_enumerate_node = make_sql_enumerate_node(engine)
    hybrid_recency_node = make_hybrid_recency_node(engine)
    relevance_gate_fn = make_relevance_gate_fn(settings.similarity_threshold)

    # --- classify node (closure over slugs) ---
    def classify_node(state: AgentState) -> dict[str, Any]:
        result = classify(state["question"], known_slugs=slugs)
        return {
            "query_type": result.query_type,
            "category_slug": result.category_slug,
            "docs": [],
            "scores": [],
            "answer": "",
            "sources": [],
        }

    # --- assemble graph ---
    builder: StateGraph[AgentState] = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)  # type: ignore[arg-type]
    builder.add_node("generate", generate_node)  # type: ignore[arg-type]
    builder.add_node("no_match", no_match_node)  # type: ignore[arg-type]
    builder.add_node("sql_count", sql_count_node)  # type: ignore[arg-type]
    builder.add_node("sql_enumerate", sql_enumerate_node)  # type: ignore[arg-type]
    builder.add_node("hybrid_recency", hybrid_recency_node)  # type: ignore[arg-type]

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify",
        route_query,
        {
            "retrieve": "retrieve",
            "sql_count": "sql_count",
            "sql_enumerate": "sql_enumerate",
            "hybrid_recency": "hybrid_recency",
        },
    )

    builder.add_conditional_edges(
        "retrieve",
        relevance_gate_fn,
        {"generate": "generate", "no_match": "no_match"},
    )

    builder.add_edge("generate", END)
    builder.add_edge("no_match", END)
    builder.add_edge("sql_count", END)
    builder.add_edge("sql_enumerate", END)
    builder.add_edge("hybrid_recency", END)

    return builder


# Module-level compiled graph — loaded by langgraph.json for Studio.
# Wrapped in try/except so importing this module (e.g. in tests or offline tools)
# doesn't crash when the DB is unavailable. Tests always call build_graph() directly
# with injected mocks and never use this module-level instance.
try:
    graph = build_graph().compile()
except Exception as _err:  # noqa: BLE001
    logger.warning("agent.graph: module-level graph unavailable (%s) — use build_graph()", _err)
    graph = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Manual smoke test (live, requires DB + OPENAI_API_KEY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    queries = [
        "What kind of tools does Bitovi recommend for E2E testing?",
        "Can you show me all Bitovi articles about DevOps?",
        "How many articles does Bitovi have about AI?",
        "What is Bitovi's latest blog post about?",
    ]
    for q in queries:
        print(f"\n{'=' * 60}\nQ: {q}")
        result = graph.invoke({"question": q})  # type: ignore[call-overload]
        print(f"type : {result.get('query_type')}")
        print(f"answer: {result.get('answer', '')[:200]}")
        srcs = result.get("sources", [])
        print(f"sources ({len(srcs)}): {json.dumps(srcs[:2], indent=2)}")
