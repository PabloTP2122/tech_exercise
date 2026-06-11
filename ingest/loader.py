"""Ingestion orchestrator — discover → fetch → extract → Document list.

Single responsibility: wire :mod:`ingest.discovery`, :mod:`ingest.fetcher`,
and :mod:`ingest.extract` together and expose the public API used by
:mod:`ingest.load_chroma` and the test suite.

Public API
----------
build_documents(filter_urls=None)
    Full ingest pipeline; returns :class:`~langchain_core.documents.Document`
    objects ready for chunking and embedding.
"""

import logging

from langchain_core.documents import Document

from ingest.discovery import discover_article_urls
from ingest.extract import build_document
from ingest.fetcher import fetch_all

logger = logging.getLogger(__name__)


def build_documents(filter_urls: list[str] | None = None) -> list[Document]:
    """Load, fetch, and extract Bitovi blog articles as Documents.

    When ``filter_urls`` is ``None`` (default) the full article set is
    discovered via :func:`ingest.discovery.discover_article_urls`.  Pass an
    explicit list to fetch a deterministic subset (e.g. one article for
    testing) — the list is used verbatim, bypassing discovery.

    Args:
        filter_urls: Optional explicit list of article URLs to fetch.  Pass
            ``None`` to ingest all articles (runs discovery + cross-check).

    Returns:
        List of :class:`~langchain_core.documents.Document` objects with all
        six metadata keys populated and clean ``page_content``.
    """
    if filter_urls is not None:
        urls = filter_urls
        logger.info("build_documents: using %d explicit URL(s) (no discovery)", len(urls))
    else:
        urls = discover_article_urls()

    docs: list[Document] = []
    for url, html in fetch_all(urls):
        if html is None:
            logger.warning("build_documents: skipping %s (fetch failed)", url)
            continue
        try:
            doc = build_document(url, html)
            docs.append(doc)
        except Exception as exc:
            logger.warning("build_documents: skipping %s (extract error: %s)", url, exc)

    logger.info("build_documents: produced %d documents", len(docs))
    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = build_documents()
    logger.info("Total documents built: %d", len(result))
    if result:
        sample_meta = result[0].metadata
        logger.info("Sample metadata keys: %s", list(sample_meta.keys()))
        logger.info("Sample categories: %s", sample_meta.get("categories"))
