"""Polite HTTP fetcher for Bitovi blog article HTML.

Single responsibility: fetch a URL and return its raw HTML string.

Public API
----------
fetch_html(url)
    Fetch one URL and return its HTML (raises on non-2xx).
fetch_all(urls, requests_per_second=2)
    Yield (url, html | None) for each URL; continues on failure.
"""

import logging
import time
from collections.abc import Iterator

import requests

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; company-blog-rag/1.0; +https://github.com)"
_TIMEOUT = 15


def fetch_html(url: str, *, timeout: int = _TIMEOUT) -> str:
    """Fetch ``url`` and return the raw HTML body.

    Args:
        url: Fully-qualified URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Raw HTML string.

    Raises:
        requests.HTTPError: For non-2xx responses.
        requests.RequestException: For network-level errors.
    """
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_all(
    urls: list[str],
    *,
    requests_per_second: float = 2.0,
) -> Iterator[tuple[str, str | None]]:
    """Yield ``(url, html)`` for each URL; yield ``(url, None)`` on error.

    Enforces a polite inter-request delay derived from ``requests_per_second``.
    Logs a WARNING for each failure but continues (mirrors the old
    ``SitemapLoader(continue_on_failure=True)`` behaviour).

    Args:
        urls: Ordered list of URLs to fetch.
        requests_per_second: Maximum fetch rate.  Defaults to 2.

    Yields:
        ``(url, html)`` on success, ``(url, None)`` on any fetch error.
    """
    if not urls:
        return

    delay = 1.0 / max(requests_per_second, 0.01)
    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(delay)
        try:
            html = fetch_html(url)
            logger.debug("Fetched %s (%d chars)", url, len(html))
            yield url, html
        except Exception as exc:
            logger.warning("fetch_all: skipping %s — %s", url, exc)
            yield url, None
