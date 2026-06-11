"""Catalog nodes: sql_count and sql_enumerate.

Public API (node factories)
---------------------------
make_sql_count_node(engine)     -> Callable[[dict[str, Any]], dict[str, Any]]
make_sql_enumerate_node(engine) -> Callable[[dict[str, Any]], dict[str, Any]]
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from agent.prompts import render_count, render_enumeration
from api.config import get_settings
from ingest.catalog_db import (
    count_articles_by_slug,
    count_articles_by_year,
    get_article_count,
    get_recent_articles,
    list_articles_by_slug,
    list_articles_by_year,
)
from ingest.clean import clean_title

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Max articles returned by enumerate-all (no slug) to keep response manageable.
_ENUMERATE_ALL_LIMIT = 50


def make_sql_count_node(engine: sa.Engine) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a sql_count terminal node.

    Every count response includes a reference link so the UI always shows
    at least one source (brief requirement: "shows reference links").
    """

    def sql_count(state: dict[str, Any]) -> dict[str, Any]:
        slug: str | None = state.get("category_slug")
        year: int | None = state.get("year")
        if year is not None:
            n = count_articles_by_year(engine, year, slug=slug)
        elif slug:
            n = count_articles_by_slug(engine, slug)
        else:
            n = get_article_count(engine)
        base = get_settings().blog_base_url.rstrip("/")
        if slug:
            link: dict[str, str] = {
                "title": f"All Bitovi articles about {slug}",
                "url": f"{base}/blog/topic/{slug}/page/1",
            }
        else:
            link = {"title": "Bitovi Blog", "url": f"{base}/blog"}
        return {"answer": render_count(n, slug, year=year), "sources": [link]}

    return sql_count


def make_sql_enumerate_node(engine: sa.Engine) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a sql_enumerate terminal node."""

    def sql_enumerate(state: dict[str, Any]) -> dict[str, Any]:
        slug: str | None = state.get("category_slug")
        year: int | None = state.get("year")
        if year is not None:
            raw_rows = list_articles_by_year(engine, year, slug=slug)
        elif slug:
            raw_rows = list_articles_by_slug(engine, slug)
        else:
            # No slug: return recent articles up to limit (avoid unbounded response).
            raw_rows = get_recent_articles(engine, limit=_ENUMERATE_ALL_LIMIT)
        # Unescape HTML entities (e.g. &amp; → &) stored in DB titles.
        rows = [{"title": clean_title(r["title"]), "url": r["url"]} for r in raw_rows]
        sources = rows
        if not sources:
            # Reference-link parity: even an empty result ships one link.
            base = get_settings().blog_base_url.rstrip("/")
            sources = [{"title": "Bitovi Blog", "url": f"{base}/blog"}]
        return {"answer": render_enumeration(rows, slug, year=year), "sources": sources}

    return sql_enumerate
