"""Article URL discovery for the Bitovi blog.

Single responsibility: determine *which* URLs to ingest, using two independent
sources — the sitemap XML and the paginated blog index — then reconcile them.

Public API
----------
list_sitemap_urls()
    Parse ``sitemap.xml`` (single HTTP request, no article body fetches).
list_paginator_urls()
    Walk ``/blog/page/N`` and collect article hrefs.
discover_article_urls(strict=True)
    Union both sources; raise :exc:`DiscoveryMismatchError` when they differ
    (default) or return the union silently (``strict=False``).
"""

import logging
import re
import time
from urllib.request import urlopen
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from api.config import get_settings

logger = logging.getLogger(__name__)

_ARTICLE_RE_TEMPLATE = r"{base}/blog/(?!topic/|page/)[\w-]+/?$"

# Maximum pages to crawl on the paginator (safety cap — Bitovi has ~50 pages).
_MAX_PAGINATOR_PAGES = 100
# Politeness delay between paginator page requests (seconds).
_PAGINATOR_DELAY = 0.5
# HTTP request timeout (seconds).
_HTTP_TIMEOUT = 15
# User-Agent sent with all requests from this module.
_UA = "Mozilla/5.0 (compatible; company-blog-rag/1.0; +https://github.com)"


class DiscoveryMismatchError(RuntimeError):
    """Raised when sitemap and paginator article sets differ (strict mode).

    Attributes:
        sitemap_only: URLs present in the sitemap but absent from the paginator.
        paginator_only: URLs present in the paginator but absent from the sitemap.
    """

    def __init__(self, sitemap_only: set[str], paginator_only: set[str]) -> None:
        self.sitemap_only = sitemap_only
        self.paginator_only = paginator_only
        lines = ["Sitemap and paginator article sets differ."]
        if sitemap_only:
            preview = ", ".join(sorted(sitemap_only)[:5])
            lines.append(f"  In sitemap only ({len(sitemap_only)}): {preview}")
        if paginator_only:
            lines.append(
                f"  In paginator only ({len(paginator_only)}): "
                + ", ".join(sorted(paginator_only)[:5])
            )
        super().__init__("\n".join(lines))


def list_sitemap_urls() -> list[str]:
    """Return all article URLs from ``sitemap.xml``.

    Parses the sitemap directly (one HTTP request, no article body fetches).
    Applies the standard ``/blog/{slug}`` article filter.

    Returns:
        Article URLs in sitemap order.  Returns ``[]`` on fetch or parse error.
    """
    settings = get_settings()
    base = settings.blog_base_url
    pattern = re.compile(_ARTICLE_RE_TEMPLATE.format(base=re.escape(base)))

    try:
        with urlopen(f"{base}/sitemap.xml", timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            tree = ElementTree.parse(resp)
    except Exception as exc:
        logger.warning("Could not fetch sitemap: %s", exc)
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: list[str] = []
    for loc in tree.findall(".//sm:loc", ns):
        url = (loc.text or "").strip()
        if pattern.match(url):
            urls.append(url)

    logger.info("list_sitemap_urls: found %d article URLs", len(urls))
    return urls


def list_paginator_urls() -> list[str]:
    """Walk ``/blog/page/N`` and return all article hrefs found.

    Stops when a page returns 404, has no new article links, or
    ``_MAX_PAGINATOR_PAGES`` is reached.  Polite: ``_PAGINATOR_DELAY`` seconds
    between requests.

    Returns:
        Deduplicated article URLs in discovery order.  Returns ``[]`` on error.
    """
    settings = get_settings()
    base = settings.blog_base_url
    pattern = re.compile(_ARTICLE_RE_TEMPLATE.format(base=re.escape(base)))

    seen: set[str] = set()
    urls: list[str] = []
    headers = {"User-Agent": _UA}

    for page_num in range(1, _MAX_PAGINATOR_PAGES + 1):
        page_url = f"{base}/blog/page/{page_num}"
        try:
            resp = requests.get(page_url, headers=headers, timeout=_HTTP_TIMEOUT)
            if resp.status_code == 404:
                logger.debug("Paginator: page %d returned 404 — stopping.", page_num)
                break
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Paginator: error fetching page %d (%s) — stopping.", page_num, exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")
        new_found = 0
        for a in soup.find_all("a", href=True):
            href: str = str(a["href"])
            # Resolve relative URLs
            if href.startswith("/"):
                href = base.rstrip("/") + href
            if pattern.match(href) and href not in seen:
                seen.add(href)
                urls.append(href)
                new_found += 1

        logger.debug(
            "Paginator: page %d → %d new article URLs (total %d)", page_num, new_found, len(urls)
        )

        if new_found == 0:
            logger.debug("Paginator: no new URLs on page %d — stopping.", page_num)
            break

        if page_num < _MAX_PAGINATOR_PAGES:
            time.sleep(_PAGINATOR_DELAY)

    logger.info("list_paginator_urls: found %d article URLs across paginator pages", len(urls))
    return urls


def discover_article_urls(*, strict: bool = True) -> list[str]:
    """Return the deduplicated union of sitemap and paginator article URLs.

    Reconciles both sources and surfaces discrepancies so sitemap gaps are
    visible before paying to embed.

    Args:
        strict: When ``True`` (default) raise :exc:`DiscoveryMismatchError` if
            the two sources disagree, enumerating the full diff.  When
            ``False`` log a WARNING and continue with the union.

    Returns:
        Deduplicated article URLs (sitemap order first, then paginator-only
        additions).

    Raises:
        DiscoveryMismatchError: When ``strict=True`` and the two sources differ.
    """
    sitemap_urls = list_sitemap_urls()
    paginator_urls = list_paginator_urls()

    sitemap_set = set(sitemap_urls)
    paginator_set = set(paginator_urls)
    sitemap_only = sitemap_set - paginator_set
    paginator_only = paginator_set - sitemap_set

    if sitemap_only or paginator_only:
        if strict:
            raise DiscoveryMismatchError(sitemap_only, paginator_only)
        logger.warning(
            "Discovery mismatch (strict=False — continuing with union): "
            "%d sitemap-only, %d paginator-only",
            len(sitemap_only),
            len(paginator_only),
        )

    # Return union: sitemap order first, then any paginator-only additions.
    union_urls: list[str] = list(sitemap_urls)
    for url in paginator_urls:
        if url not in sitemap_set:
            union_urls.append(url)

    logger.info("discover_article_urls: %d total article URLs in union", len(union_urls))
    return union_urls
