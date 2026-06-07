"""Normalized articles catalog table — one row per article.

Single responsibility: own the DDL and upsert for the ``articles`` table.
This is the relational catalog that powers count, enumeration, and recency
SQL queries at agent run time, eliminating the chunk-vs-article pitfall
that occurs when counting rows in a chunk-level vector table.

Schema
------
::

    CREATE TABLE articles (
        source_url   TEXT PRIMARY KEY,
        title        TEXT NOT NULL DEFAULT '',
        description  TEXT NOT NULL DEFAULT '',
        author       TEXT NOT NULL DEFAULT '',
        published_at TIMESTAMPTZ,       -- NULL when date cannot be parsed
        categories   TEXT NOT NULL DEFAULT ','   -- delimited ",slug,slug,"
    )
"""

import logging
from datetime import UTC, datetime

import sqlalchemy as sa
from langchain_core.documents import Document
from sqlalchemy import text

logger = logging.getLogger(__name__)

_ARTICLES_DDL = """
CREATE TABLE IF NOT EXISTS articles (
    source_url   TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    author       TEXT NOT NULL DEFAULT '',
    published_at TIMESTAMPTZ,
    categories   TEXT NOT NULL DEFAULT ','
)
"""

_UPSERT_SQL = """
INSERT INTO articles (source_url, title, description, author, published_at, categories)
VALUES (:source_url, :title, :description, :author, :published_at, :categories)
ON CONFLICT (source_url) DO UPDATE SET
    title        = EXCLUDED.title,
    description  = EXCLUDED.description,
    author       = EXCLUDED.author,
    published_at = EXCLUDED.published_at,
    categories   = EXCLUDED.categories
"""

# Date formats observed in the Bitovi blog JSON-LD (datePublished).
_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
]


def _parse_published_at(value: str) -> datetime | None:
    """Parse a ``published_at`` string into a timezone-aware datetime.

    Tries multiple formats defensively; returns ``None`` on failure so
    the row is stored with a NULL ``published_at`` rather than crashing.

    Args:
        value: Raw date string from article metadata.

    Returns:
        Timezone-aware :class:`datetime`, or ``None`` if unparseable.
    """
    if not value:
        return None
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    logger.warning("Could not parse published_at=%r — storing NULL.", value)
    return None


def ensure_articles_table(engine: sa.Engine) -> None:
    """Create the ``articles`` table if it does not already exist.

    Args:
        engine: SQLAlchemy sync engine connected to the target database.
    """
    with engine.begin() as conn:
        conn.execute(text(_ARTICLES_DDL))


def upsert_articles(engine: sa.Engine, docs: list[Document]) -> int:
    """Upsert one catalog row per article from the provided documents.

    Args:
        engine: SQLAlchemy sync engine connected to the target database.
        docs: Documents as returned by :func:`~ingest.loader.build_documents`.
              Each document's ``metadata`` must carry the six standard keys.

    Returns:
        Number of rows upserted (equals ``len(docs)`` on success).
    """
    if not docs:
        return 0

    rows = [
        {
            "source_url": doc.metadata.get("source_url", ""),
            "title": doc.metadata.get("title", ""),
            "description": doc.metadata.get("description", ""),
            "author": doc.metadata.get("author", ""),
            "published_at": _parse_published_at(doc.metadata.get("published_at", "")),
            "categories": doc.metadata.get("categories", ","),
        }
        for doc in docs
    ]

    with engine.begin() as conn:
        conn.execute(text(_UPSERT_SQL), rows)

    logger.info("Upserted %d rows into articles table.", len(rows))
    return len(rows)


def get_article_count(engine: sa.Engine) -> int:
    """Return the number of rows in the ``articles`` table (0 if absent).

    Args:
        engine: SQLAlchemy sync engine.

    Returns:
        Row count, or ``0`` if the table does not exist or the query fails.
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT count(*) FROM articles")).fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0
