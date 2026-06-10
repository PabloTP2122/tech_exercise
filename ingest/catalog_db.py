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
        categories   TEXT NOT NULL DEFAULT ''    -- delimited ",slug,slug,"; "" when untagged
    )
"""

import logging
import re
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
    categories   TEXT NOT NULL DEFAULT ''
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
            "categories": doc.metadata.get("categories", ""),
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


# ---------------------------------------------------------------------------
# Slug helpers — used by the classifier and the SQL routes
# ---------------------------------------------------------------------------

# ADR-0008 S4: read-time slug validation grammar.
# Mirrors the ingest-time _SLUG_RE in extract.py — both enforce the same grammar.
_SLUG_RE = re.compile(r"^[a-z0-9-]+$")


def escape_like(s: str) -> str:
    """Escape PostgreSQL ILIKE metacharacters for use with ``ESCAPE '\\'``.

    Escapes ``\\``, ``%``, and ``_`` so a slug value is treated as a literal
    string, not a wildcard pattern, even when bound as a parameter.

    Required by ADR-0008 S1/S2 for all count/enumeration queries:
    ``conn.execute(text("... ILIKE :pat ESCAPE '\\'"), {"pat": f"%,{escape_like(slug)},%"})``

    Args:
        s: Raw slug string to sanitize.

    Returns:
        String with ``\\``, ``%``, ``_`` backslash-escaped.
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _split_category_slugs(rows: list[str]) -> list[str]:
    """Split comma-delimited category rows into individual validated slugs.

    Applies ADR-0008 S4 read-time validation: drops any slug that does not
    match ``^[a-z0-9-]+$`` and logs a WARNING so operators can investigate
    legacy dirty rows or ingest-time whitelist gaps.

    Args:
        rows: Raw ``categories`` column values, e.g. ``[",react,angular,", ",devops,"]``.

    Returns:
        Sorted, deduplicated list of conforming slug strings.
    """
    slugs: set[str] = set()
    for row in rows:
        for part in row.split(","):
            s = part.strip()
            if not s:
                continue
            if _SLUG_RE.match(s):
                slugs.add(s)
            else:
                logger.warning("list_category_slugs: dropping non-conforming slug %r", s)
    return sorted(slugs)


def list_category_slugs(engine: sa.Engine) -> list[str]:
    """Return all distinct, validated category slugs from the articles table.

    Uses a static query (no value interpolation).  Drops any slug not matching
    ``^[a-z0-9-]+$`` via :func:`_split_category_slugs` (ADR-0008 S4 read-time
    re-validation).  Returns ``[]`` on any DB error (mirrors :func:`get_article_count`).

    Args:
        engine: SQLAlchemy sync engine connected to the target database.

    Returns:
        Sorted list of validated category slugs, or ``[]`` on error.
    """
    try:
        with engine.connect() as conn:
            rows: list[str] = list(
                conn.execute(
                    text("SELECT DISTINCT categories FROM articles WHERE categories <> ''")
                ).scalars()
            )
        return _split_category_slugs(rows)
    except Exception:
        logger.exception("list_category_slugs: DB query failed")
        return []


# ---------------------------------------------------------------------------
# Agent runtime helpers — count, enumeration, recency (routing nodes)
# ---------------------------------------------------------------------------


def count_articles_by_slug(engine: sa.Engine, slug: str) -> int:
    """Return the number of distinct articles tagged with ``slug``.

    Args:
        engine: SQLAlchemy sync engine.
        slug: Validated category slug (``^[a-z0-9-]+$``).

    Returns:
        Count of distinct ``source_url`` rows whose ``categories`` column
        contains ``,slug,``.  Returns ``0`` on any error.
    """
    pat = f"%,{escape_like(slug)},%"
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT COUNT(DISTINCT source_url) FROM articles"
                " WHERE categories ILIKE :pat ESCAPE '\\'"
            ),
            {"pat": pat},
        ).fetchone()
    return int(row[0]) if row else 0


def list_articles_by_slug(engine: sa.Engine, slug: str) -> list[dict[str, str]]:
    """Return articles tagged with ``slug``, newest first.

    Args:
        engine: SQLAlchemy sync engine.
        slug: Validated category slug.

    Returns:
        List of ``{title, url}`` dicts ordered by ``published_at DESC``.
    """
    pat = f"%,{escape_like(slug)},%"
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT title, source_url FROM articles"
                " WHERE categories ILIKE :pat ESCAPE '\\'"
                " ORDER BY published_at DESC"
            ),
            {"pat": pat},
        ).fetchall()
    return [{"title": r[0] or "", "url": r[1] or ""} for r in rows]


def get_recent_articles(engine: sa.Engine, limit: int = 3) -> list[dict[str, str]]:
    """Return the most recently published articles.

    Args:
        engine: SQLAlchemy sync engine.
        limit: Maximum number of rows to return.

    Returns:
        List of ``{title, url, published_at}`` dicts ordered newest first.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT title, source_url, published_at FROM articles"
                " ORDER BY published_at DESC LIMIT :n"
            ),
            {"n": limit},
        ).fetchall()
    return [{"title": r[0] or "", "url": r[1] or "", "published_at": str(r[2] or "")} for r in rows]
