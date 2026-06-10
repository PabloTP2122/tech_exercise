"""Live RSS recency overlay for the Bitovi blog RAG agent.

Thin SRP fetcher behind a single try/except.  Returns the newest feed items,
or [] on any failure (offline, timeout, HTTP error, malformed XML).  The caller
(the recency node) owns the SQL-vs-RSS merge and pubDate normalisation.

Public API
----------
fetch_recent(slug, limit=3) -> list[dict[str, str]]
    Fetch the newest ``limit`` items from the global or per-topic RSS feed.
    Returns RSS-native keys: title, link, description, pubDate.
    Returns [] on any error — never propagates (ADR-0004 §3).
"""

import logging
import xml.etree.ElementTree as ET

import requests

from api.config import get_settings

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; company-blog-rag/1.0; +https://github.com)"
_TIMEOUT = 10


def fetch_recent(slug: str | None, limit: int = 3) -> list[dict[str, str]]:
    """Return up to ``limit`` newest RSS items, or ``[]`` on any failure.

    Args:
        slug: Category slug for a per-topic feed, or ``None`` for the global feed.
              Must already be validated (``^[a-z0-9-]+$``) by the caller.
        limit: Maximum number of items to return.  Feed order (newest-first) is
               preserved.

    Returns:
        List of dicts with keys ``title``, ``link``, ``description``, ``pubDate``
        (verbatim RSS strings — empty string when a tag is absent).
        Returns ``[]`` on any HTTP, network, or XML parse error.
    """
    try:
        base = get_settings().blog_base_url.rstrip("/")
        url = f"{base}/blog/rss.xml" if slug is None else f"{base}/blog/topic/{slug}/rss.xml"

        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        items = root.findall(".//item")

        result: list[dict[str, str]] = []
        for item in items[:limit]:
            result.append(
                {
                    "title": item.findtext("title") or "",
                    "link": item.findtext("link") or "",
                    "description": item.findtext("description") or "",
                    "pubDate": item.findtext("pubDate") or "",
                }
            )
        return result

    except Exception:
        logger.warning("fetch_recent: failed to fetch RSS (slug=%r) — returning []", slug)
        return []
