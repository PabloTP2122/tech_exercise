"""Recency node: hybrid SQL + RSS overlay.

Public API (node factory)
--------------------------
make_hybrid_recency_node(engine) -> Callable[[AgentState], dict]
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from agent.prompts import render_recency
from agent.rss import fetch_recent
from ingest.catalog_db import _parse_published_at, get_recent_articles

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def make_hybrid_recency_node(engine: sa.Engine) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a hybrid_recency terminal node.

    SQL base is always authoritative.  RSS overlay wins only when the live
    RSS feed contains an item strictly newer than the SQL top-1 row
    (ADR-0004 §3).

    Sources are populated from the winning items so the UI always has at
    least one reference link (brief requirement: "shows reference links").
    """

    def hybrid_recency(state: dict[str, Any]) -> dict[str, Any]:
        slug: str | None = state.get("category_slug")
        sql_rows = get_recent_articles(engine, limit=3)
        rss_items = fetch_recent(slug, limit=3)

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
                    "title": it["title"],
                    "url": it["link"],
                    "date": it["pubDate"],
                }
                for it in rss_items
            ]
        else:
            items = [
                {
                    "title": row["title"],
                    "url": row["url"],
                    "date": row["published_at"],
                }
                for row in sql_rows
            ]

        sources = [{"title": it["title"], "url": it["url"]} for it in items if it["url"]]
        return {"answer": render_recency(items), "sources": sources}

    return hybrid_recency
