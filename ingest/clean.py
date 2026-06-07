"""Content cleaning for Bitovi blog articles.

Single responsibility: given **raw HTML**, extract only the article body
(``#hs_cos_wrapper_post_body``) and return clean **Markdown** text.  Nav,
footer, cookie banners, tracking pixels, and related-post blocks are stripped.
Code blocks (``<pre>``/``<code>``) are preserved as fenced/backtick Markdown
so the embedder receives intact syntax rather than line-shattered plain text.

Called by :func:`ingest.extract.build_document` which supplies raw HTML from
:mod:`ingest.fetcher`, so the selector always finds its target.

Falls back to the original string if the selector is absent (logging a warning)
to avoid silent data-loss; callers should treat this as a data-quality signal.
"""

import logging

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from markdownify import markdownify

logger = logging.getLogger(__name__)

_BODY_SELECTOR = "#hs_cos_wrapper_post_body"


def clean_html(html: str) -> str:
    """Return the clean article-body Markdown from a full Bitovi HTML page.

    Extracts only the ``#hs_cos_wrapper_post_body`` element and converts it to
    Markdown, preserving ``<pre>``/``<code>`` as fenced/backtick spans and
    ATX-style headings.  Nav, footer, tracking pixels, and related-post blocks
    are excluded.  If the selector is absent the function returns ``html``
    unchanged and logs a warning.

    Args:
        html: Raw HTML string as returned by :func:`ingest.fetcher.fetch_html`.

    Returns:
        Cleaned article body as Markdown, or the original ``html`` string if
        the selector is not found.
    """
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one(_BODY_SELECTOR)
    if body is None:
        logger.warning("Body selector '%s' not found — using full content", _BODY_SELECTOR)
        return html
    return str(markdownify(str(body), heading_style="ATX", strip=["script", "style"]))


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
