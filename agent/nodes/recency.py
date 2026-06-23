"""Recency node: hybrid SQL + RSS overlay.

Public API (node factory)
--------------------------
make_hybrid_recency_node(engine) -> Callable[[AgentState], dict]
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from agent.prompts import render_recency
from agent.rss import fetch_recent
from api.config import get_settings
from ingest.catalog_db import _parse_published_at, get_recent_articles
from ingest.clean import clean_title

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_HUMAN_DATE_FMT = "%B %-d, %Y"  # e.g. "May 28, 2026"


def _format_sql_date(value: str) -> str:
    """Format a ``published_at`` DB string to a human-readable date.

    Tries _parse_published_at first (handles the _DATE_FORMATS list), then
    falls back to datetime.fromisoformat() which handles the "+00:00" variant
    stored by newer ingest runs (e.g. "2026-05-28 16:51:37+00:00").
    Returns the original string unchanged if all parsing attempts fail.
    """
    dt = _parse_published_at(value)
    if dt is None:
        try:
            dt = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return value
    try:
        return dt.strftime(_HUMAN_DATE_FMT)
    except Exception:
        return value


def _format_rss_date(value: str) -> str:
    """Format an RFC-2822 RSS ``pubDate`` to a human-readable date.

    Returns the original string unchanged if parsing fails (fail-safe).
    """
    try:
        return parsedate_to_datetime(value).strftime(_HUMAN_DATE_FMT)
    except Exception:
        return value


def make_hybrid_recency_node(engine: sa.Engine) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a hybrid_recency terminal node.

    SQL base is always authoritative.  RSS overlay wins only when the live
    RSS feed contains an item strictly newer than the SQL top-1 row
    (ADR-0004 §3).  For ``oldest`` direction the RSS overlay is skipped
    entirely — feeds only carry the newest items.

    Sources are populated from the winning items so the UI always has at
    least one reference link (brief requirement: "shows reference links").
    """

    def hybrid_recency(state: dict[str, Any]) -> dict[str, Any]:
        slug: str | None = state.get("category_slug")
        oldest: bool = state.get("recency_direction") == "oldest"
        limit: int = state.get("recency_limit") or 3
        sql_rows = get_recent_articles(engine, limit=limit, oldest=oldest, slug=slug)
        rss_items = [] if oldest else fetch_recent(slug, limit=limit)

        use_rss = False
        if rss_items and sql_rows:
            try:
                rss_dt = parsedate_to_datetime(rss_items[0]["pubDate"])
                sql_dt = _parse_published_at(sql_rows[0]["published_at"])
                if sql_dt is not None and rss_dt > sql_dt:
                    use_rss = True
            except Exception:
                logger.warning("hybrid_recency: could not compare dates — using SQL")

        if use_rss:
            items = [
                {
                    "title": clean_title(it["title"]),
                    "url": it["link"],
                    "date": _format_rss_date(it["pubDate"]),
                }
                for it in rss_items
            ]
        else:
            items = [
                {
                    "title": clean_title(row["title"]),
                    "url": row["url"],
                    "date": _format_sql_date(row["published_at"]),
                }
                for row in sql_rows
            ]

        sources = [{"title": it["title"], "url": it["url"]} for it in items if it["url"]]
        if not sources:
            # Reference-link parity: even an empty result ships one link.
            base = get_settings().blog_base_url.rstrip("/")
            if slug:
                sources = [
                    {
                        "title": f"All Bitovi articles about {slug}",
                        "url": f"{base}/blog/topic/{slug}/page/1",
                    }
                ]
            else:
                sources = [{"title": "Bitovi Blog", "url": f"{base}/blog"}]
        return {"answer": render_recency(items, oldest=oldest), "sources": sources}

    return hybrid_recency
