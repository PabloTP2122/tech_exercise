"""Hybrid retrieval helpers: keyword search, RRF fusion, context dedupe.

Single responsibility: pure retrieval-side helpers used by the semantic_qa
retrieve node.  Dense vector search stays in :class:`PGVectorStore`; this
module adds a Postgres full-text pass over the same chunks table and fuses
both rankings with Reciprocal Rank Fusion.

Design notes (kept deliberately custom — ~40 lines of SQL/fusion):
- ``websearch_to_tsquery`` with OR-joined tokens, because AND semantics would
  make natural-language questions match nothing.
- Keyword hits enrich and reorder the LLM context but NEVER open the
  relevance gate: the gate keeps scoring vector-only relevance scores, so the
  no-match invariant is unaffected.
- Keyword search fails open: any DB error returns ``[]`` and retrieval
  degrades to pure vector search (same fail-safe pattern as agent/rss.py).
"""

from __future__ import annotations

import logging
import re

import sqlalchemy as sa
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Tokens that websearch_to_tsquery treats as operators — never emit them as terms.
_TSQUERY_OPERATORS = {"or", "and", "not"}
# Bound the tsquery size for very long questions.
_MAX_QUERY_TOKENS = 16

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def build_or_query(question: str) -> str:
    """Return an OR-joined websearch_to_tsquery string for ``question``.

    Tokenizes on alphanumerics (punctuation is stripped, so user input cannot
    inject tsquery syntax) and joins with ``OR`` — a natural-language question
    rarely matches every term, and ``ts_rank_cd`` already rewards chunks that
    match more (and rarer) terms.

    Args:
        question: Raw user question.

    Returns:
        OR-joined token string, or ``""`` when no usable tokens remain.
    """
    tokens = [t for t in _TOKEN_RE.findall(question) if t.lower() not in _TSQUERY_OPERATORS]
    return " OR ".join(tokens[:_MAX_QUERY_TOKENS])


def keyword_search_chunks(
    engine: sa.Engine, table: str, question: str, limit: int
) -> list[Document]:
    """Full-text search over the chunks table, ranked by ``ts_rank_cd``.

    Args:
        engine: SQLAlchemy sync engine.
        table: Chunks table name (internal settings constant — not user input).
        question: Raw user question (tokenized via :func:`build_or_query`).
        limit: Maximum number of chunks to return.

    Returns:
        Matching chunks as Documents (with ``source_url``/``title`` metadata,
        the keys the generate-node whitelist needs), best-ranked first.
        Returns ``[]`` on empty query or any DB error (fail open to vector).
    """
    fts_query = build_or_query(question)
    if not fts_query or limit <= 0:
        return []
    # table is an internal constant from Settings — not user input.
    sql = sa.text(
        f"SELECT langchain_id, content, source_url, title"  # noqa: S608
        f" FROM {table}"
        f" WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', :q)"
        f" ORDER BY ts_rank_cd(to_tsvector('english', content),"
        f"                     websearch_to_tsquery('english', :q)) DESC"
        f" LIMIT :n"
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"q": fts_query, "n": limit}).fetchall()
        return [
            Document(
                id=str(row[0]),
                page_content=row[1] or "",
                metadata={"source_url": row[2] or "", "title": row[3] or ""},
            )
            for row in rows
        ]
    except Exception:
        logger.warning("keyword_search_chunks: query failed — falling back to vector-only")
        return []


def _doc_key(doc: Document) -> str:
    """Stable identity for fusion: prefer the store id, else url + content prefix."""
    if doc.id:
        return str(doc.id)
    return f"{doc.metadata.get('source_url', '')}|{doc.page_content[:64]}"


def rrf_fuse(
    vector_docs: list[Document], keyword_docs: list[Document], *, rrf_k: int = 60
) -> list[Document]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    Each document scores ``sum(1 / (rrf_k + rank))`` over the lists it appears
    in (rank is 1-based).  Pure function — no I/O.

    Args:
        vector_docs: Dense-retrieval ranking, best first.
        keyword_docs: Full-text ranking, best first.
        rrf_k: Standard RRF dampening constant.

    Returns:
        All distinct documents ordered by fused score (best first).  Ties keep
        vector-list order first (stable sort over insertion order).
    """
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for ranking in (vector_docs, keyword_docs):
        for rank, doc in enumerate(ranking, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            by_key.setdefault(key, doc)
    ordered = sorted(by_key, key=lambda key: scores[key], reverse=True)
    return [by_key[key] for key in ordered]


def dedupe_chunks(docs: list[Document], *, max_per_article: int) -> list[Document]:
    """Drop exact-duplicate chunks and cap chunks per article, order-stable.

    Args:
        docs: Fused chunk ranking, best first.
        max_per_article: Maximum chunks kept per ``source_url``.

    Returns:
        Filtered list preserving the input order.
    """
    seen_content: set[str] = set()
    per_article: dict[str, int] = {}
    result: list[Document] = []
    for doc in docs:
        if doc.page_content in seen_content:
            continue
        url = str(doc.metadata.get("source_url", ""))
        if per_article.get(url, 0) >= max_per_article:
            continue
        seen_content.add(doc.page_content)
        per_article[url] = per_article.get(url, 0) + 1
        result.append(doc)
    return result
