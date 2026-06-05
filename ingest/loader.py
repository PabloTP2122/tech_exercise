import json
import logging
import re
from typing import Any, cast

from bs4 import BeautifulSoup
from langchain_community.document_loaders import SitemapLoader
from langchain_core.documents import Document

from api.config import get_settings

logger = logging.getLogger(__name__)


def _extract_ld(soup: BeautifulSoup) -> dict[str, Any]:
    try:
        tag = soup.find("script", type="application/ld+json")
        parsed: dict[str, Any] = json.loads(tag.string) if tag and tag.string else {}
        return parsed
    except Exception as exc:
        logger.warning("JSON-LD parse error: %s", exc)
        return {}


def _categories(soup: BeautifulSoup) -> str:
    """Extract topic slugs from /blog/topic/{slug} links — the real category source on Bitovi."""
    slugs: list[str] = []
    for a in soup.find_all("a", href=True):
        raw = a["href"]
        if not isinstance(raw, str):
            continue
        if "/blog/topic/" not in raw:
            continue
        slug = raw.rstrip("/").split("/blog/topic/")[-1].strip().lower()
        if slug and slug not in slugs:
            slugs.append(slug)
    return "," + ",".join(slugs) + "," if slugs else ","


def _meta_function(meta: dict[str, Any], soup: BeautifulSoup) -> dict[str, Any]:
    source_url: str = meta.get("source", "")
    ld = _extract_ld(soup)

    if not ld:
        logger.warning("No JSON-LD found for %s", source_url)

    author_raw = ld.get("author", "")
    author = author_raw.get("name", "") if isinstance(author_raw, dict) else str(author_raw)

    categories = _categories(soup)
    if categories == ",":
        logger.warning("No topic links found for %s — categories will be empty", source_url)

    return {
        **meta,
        "source_url": source_url,
        "title": ld.get("headline", ""),
        "description": ld.get("description", ""),
        "author": author,
        "published_at": ld.get("datePublished", ""),
        "categories": categories,
    }


def build_documents() -> list[Document]:
    settings = get_settings()
    base = settings.blog_base_url

    loader = SitemapLoader(
        web_path=f"{base}/sitemap.xml",
        filter_urls=[rf"{re.escape(base)}/blog/(?!topic/|page/)[\w-]+/?$"],
        meta_function=_meta_function,
        continue_on_failure=True,
    )
    loader.requests_per_second = 2  # be polite to the server

    docs = cast(list[Document], loader.load())
    logger.info("Loaded %d documents from sitemap", len(docs))
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = build_documents()
    logger.info("Total documents built: %d", len(result))
    if result:
        sample_meta = result[0].metadata
        logger.info("Sample metadata keys: %s", list(sample_meta.keys()))
        logger.info("Sample categories: %s", sample_meta.get("categories"))
