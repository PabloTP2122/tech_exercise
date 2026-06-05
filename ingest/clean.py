"""Content cleaning for Bitovi blog articles.

Strips site chrome (nav, footer, cookie banner, CTAs) from raw SitemapLoader
output by targeting the HubSpot CMS article-body selector
``#hs_cos_wrapper_post_body``.  Falls back to the original content if the
selector is absent, logging a warning so the data quality issue is visible
without silently dropping articles.
"""

import logging

from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_BODY_SELECTOR = "#hs_cos_wrapper_post_body"


def clean_html(html: str) -> str:
    """Return the clean article-body text from a full Bitovi HTML page.

    Extracts only the ``#hs_cos_wrapper_post_body`` element.  If the selector
    is absent the function returns ``html`` unchanged (caller should pass the
    raw ``page_content`` string, which may already be pre-rendered text rather
    than HTML; BeautifulSoup handles both gracefully).

    Args:
        html: Raw HTML string (or pre-rendered text) as returned by the loader.

    Returns:
        Cleaned article body text, or the original ``html`` string if the
        selector is not found.
    """
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one(_BODY_SELECTOR)
    if body is None:
        return html
    return str(body.get_text(separator="\n", strip=True))


def clean_document(doc: Document) -> Document:
    """Return a new Document with cleaned ``page_content``, preserving metadata.

    Calls :func:`clean_html` on ``doc.page_content``.  If the article-body
    selector is absent, logs a WARNING with the source URL and returns the
    original content unchanged.

    Args:
        doc: A :class:`~langchain_core.documents.Document` produced by the
             ingest loader.

    Returns:
        A new ``Document`` with cleaned ``page_content`` and the same metadata.
    """
    raw = doc.page_content
    cleaned = clean_html(raw)

    if cleaned == raw:
        source = doc.metadata.get("source_url") or doc.metadata.get("source", "<unknown>")
        logger.warning(
            "Body selector '%s' not found for %s — using full content",
            _BODY_SELECTOR,
            source,
        )

    return Document(page_content=cleaned, metadata=doc.metadata)
