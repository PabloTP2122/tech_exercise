"""Conditional-edge routing function for the blog RAG graph."""

from __future__ import annotations

from typing import Any

_ROUTE_MAP: dict[str, str] = {
    "semantic_qa": "retrieve",
    "count": "sql_count",
    "enumeration": "sql_enumerate",
    "recency": "hybrid_recency",
}


def route_query(state: dict[str, Any]) -> str:
    """Map query_type to the target node name for add_conditional_edges."""
    return _ROUTE_MAP[state["query_type"]]
