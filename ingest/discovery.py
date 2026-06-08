"""Article URL discovery for the Bitovi blog.

Single responsibility: determine *which* URLs to ingest, using two independent
sources — the sitemap XML and the paginated blog index — then reconcile them.

Public API
----------
list_sitemap_urls()
    Parse ``sitemap.xml`` (single HTTP request, no article body fetches).
list_paginator_urls()
    Walk ``/blog/page/N``, return deduplicated article URLs.
reconcile(sitemap_urls, paginator_raw_urls)
    Pure function: compute union, set diffs, and a repeat-occurrence map from a
    raw (possibly containing repeats) paginator list.
discover_article_urls(strict=True)
    Union both sources; raise :exc:`DiscoveryMismatchError` when the *sitemap*
    lists URLs the live paginator does not confirm (the suspicious direction).
    Articles present only in the paginator (not yet in the sitemap) are silently
    included in the union — expected for newly-published posts.
"""

import logging
import re
import time
from collections import Counter
from typing import NamedTuple
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

from api.config import get_settings

logger = logging.getLogger(__name__)

# Maximum pages to crawl on the paginator (safety cap — Bitovi has ~50 pages).
_MAX_PAGINATOR_PAGES = 100
# Politeness delay between paginator page requests (seconds).
_PAGINATOR_DELAY = 0.5
# HTTP request timeout (seconds).
_HTTP_TIMEOUT = 15
# User-Agent sent with all requests from this module.
_UA = "Mozilla/5.0 (compatible; company-blog-rag/1.0; +https://github.com)"

# Path-level article pattern: /blog/<slug>, excluding /topic/ and /page/ prefixes.
# Allows dots and .html suffixes (e.g. canjs-4.0, ie-11-and-angular-overview.html).
_ARTICLE_PATH_RE = re.compile(r"/blog/(?!topic/|page/)[\w.\-]+$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_url(href: str, canonical_host: str) -> str | None:
    """Normalize ``href`` to a canonical absolute URL, or ``None`` if off-site.

    Performs four normalizations:

    1. Resolves relative ``/path`` hrefs against ``canonical_host``.
    2. Forces the ``https`` scheme.
    3. Collapses the apex/www host variants (e.g. ``bitovi.com`` ↔
       ``www.bitovi.com``) to the canonical host from ``blog_base_url``.
    4. Strips query string, fragment, and trailing slash.

    Args:
        href: Raw href from an anchor tag or sitemap ``<loc>``.
        canonical_host: The normalized hostname (e.g. ``"www.bitovi.com"``)
            derived from ``settings.blog_base_url``.

    Returns:
        Canonical URL string, or ``None`` when the resolved host does not
        match the canonical host's apex domain (i.e. it is a different site).
    """
    if not href or href.startswith("#"):
        return None
    if href.startswith("/"):
        href = f"https://{canonical_host}{href}"
    p = urlparse(href)
    if not p.scheme or not p.netloc:
        return None
    host = p.netloc.lower()
    # Collapse apex ↔ www variants (e.g. bitovi.com ↔ www.bitovi.com).
    apex = canonical_host.removeprefix("www.")
    if host not in (canonical_host, apex):
        return None  # off-site
    path = p.path.rstrip("/") or "/"
    return urlunparse(("https", canonical_host, path, "", "", ""))


def _is_article_url(url: str) -> bool:
    """Return ``True`` when ``url`` has an article path (not topic/page routes).

    Uses the path-level :data:`_ARTICLE_PATH_RE` which allows dots and
    ``.html`` suffixes in slugs (e.g. ``canjs-4.0``,
    ``ie-11-and-angular-overview.html``).

    Args:
        url: Already-normalized absolute URL string.

    Returns:
        ``True`` for article paths; ``False`` for topic, page, or other paths.
    """
    return bool(_ARTICLE_PATH_RE.match(urlparse(url).path))


def _list_paginator_raw(base: str, canonical_host: str) -> list[str]:
    """Crawl all ``/blog/page/N`` pages and return every article URL found.

    Returns normalized URLs **including duplicate occurrences** (featured or
    popular-post links appear on multiple pages and repeat within a page).
    Callers that need a deduplicated set should pass the result to
    :func:`reconcile` or deduplicate via a ``seen`` set.

    Args:
        base: Blog base URL (e.g. ``"https://www.bitovi.com"``).
        canonical_host: Canonical host (e.g. ``"www.bitovi.com"``).

    Returns:
        List of normalized article URLs in crawl order, with repeats.
        Returns ``[]`` on connection error.
    """
    headers = {"User-Agent": _UA}
    raw: list[str] = []

    for page_num in range(1, _MAX_PAGINATOR_PAGES + 1):
        page_url = f"{base}/blog/page/{page_num}"
        try:
            resp = requests.get(page_url, headers=headers, timeout=_HTTP_TIMEOUT)
            if resp.status_code == 404:
                logger.debug("Paginator: page %d returned 404 — stopping.", page_num)
                break
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Paginator: error on page %d (%s) — stopping.", page_num, exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")
        page_count = 0
        for a in soup.find_all("a", href=True):
            norm = _normalize_url(str(a["href"]), canonical_host)
            if norm and _is_article_url(norm):
                raw.append(norm)
                page_count += 1

        logger.debug("Paginator: page %d → %d article links (with repeats)", page_num, page_count)

        if page_count == 0:
            logger.debug("Paginator: no article links on page %d — stopping.", page_num)
            break

        if page_num < _MAX_PAGINATOR_PAGES:
            time.sleep(_PAGINATOR_DELAY)

    return raw


# ---------------------------------------------------------------------------
# Reconcile data structure
# ---------------------------------------------------------------------------


class ReconcileReport(NamedTuple):
    """Immutable result of reconciling sitemap and paginator URL sets.

    Attributes:
        union: Ordered URL list (sitemap order first, then paginator-only
            additions in discovery order).
        sitemap_unique: Distinct URLs from the sitemap.
        paginator_unique: Distinct URLs from the paginator (after dedup).
        paginator_raw_count: Total raw links found (before dedup; includes
            featured-post repeats).
        repeated: Mapping of ``url → occurrence_count`` for URLs that appear
            more than once in the raw paginator output.  These are benign
            featured/popular-post links, not genuine duplicate articles.
        sitemap_only: URLs present in the sitemap but absent from the
            paginator.  Non-empty = suspicious; may indicate crawl failure.
        paginator_only: URLs present only in the paginator (not yet in the
            sitemap).  Non-empty = expected for newly-published posts.
    """

    union: list[str]
    sitemap_unique: set[str]
    paginator_unique: set[str]
    paginator_raw_count: int
    repeated: dict[str, int]
    sitemap_only: set[str]
    paginator_only: set[str]


# ---------------------------------------------------------------------------
# Public reconciliation function
# ---------------------------------------------------------------------------


def reconcile(sitemap_urls: list[str], paginator_raw_urls: list[str]) -> ReconcileReport:
    """Compute the union and diffs of two URL sources.

    Pure function — no network calls, no side effects.  Designed as a
    testable seam between the crawler and :func:`discover_article_urls`.

    The ``repeated`` map exposes a diagnostic tool: any URL with count > 1 in
    the raw paginator output is a featured/popular-post template link repeated
    across pages, **not** a genuine duplicate article.  The union and
    ``paginator_unique`` correctly collapse these.

    Args:
        sitemap_urls: URLs from the sitemap (order-significant; used as the
            primary ordering in the union).
        paginator_raw_urls: All article URLs from the paginator in crawl order,
            **including repeats** (as returned by :func:`_list_paginator_raw`).

    Returns:
        :class:`ReconcileReport` with all computed fields.
    """
    cnt: Counter[str] = Counter(paginator_raw_urls)
    repeated: dict[str, int] = {u: c for u, c in cnt.items() if c > 1}
    paginator_unique: set[str] = set(cnt.keys())
    sitemap_unique: set[str] = set(sitemap_urls)

    sitemap_only = sitemap_unique - paginator_unique
    paginator_only = paginator_unique - sitemap_unique

    # Union: sitemap order first, then any paginator-only additions (discovery order).
    union: list[str] = list(sitemap_urls)
    seen_in_union: set[str] = sitemap_unique.copy()
    for u in paginator_raw_urls:
        if u not in seen_in_union:
            seen_in_union.add(u)
            union.append(u)

    return ReconcileReport(
        union=union,
        sitemap_unique=sitemap_unique,
        paginator_unique=paginator_unique,
        paginator_raw_count=len(paginator_raw_urls),
        repeated=repeated,
        sitemap_only=sitemap_only,
        paginator_only=paginator_only,
    )


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class DiscoveryMismatchError(RuntimeError):
    """Raised when the sitemap lists URLs the live paginator does not confirm.

    This is the *suspicious* direction of the asymmetric reconcile gate:
    the sitemap claims an article exists, but crawling all live paginator
    pages did not find it.  This may indicate a broken paginator crawl.

    ``paginator_only`` (live articles not yet in the sitemap) is **not** an
    error — it is an expected state for newly-published posts and is always
    included in the union without raising.

    Attributes:
        sitemap_only: URLs in the sitemap but absent from the paginator.
        paginator_only: URLs found by the paginator but absent from the sitemap.
    """

    def __init__(self, sitemap_only: set[str], paginator_only: set[str]) -> None:
        self.sitemap_only = sitemap_only
        self.paginator_only = paginator_only
        lines = ["Sitemap lists URLs the live paginator did not confirm."]
        if sitemap_only:
            preview = ", ".join(sorted(sitemap_only)[:5])
            lines.append(f"  In sitemap only ({len(sitemap_only)}): {preview}")
        if paginator_only:
            lines.append(
                f"  In paginator only ({len(paginator_only)}): "
                + ", ".join(sorted(paginator_only)[:5])
            )
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_sitemap_urls() -> list[str]:
    """Return all article URLs from ``sitemap.xml``.

    Parses the sitemap directly (one HTTP request, no article body fetches).
    Normalizes each ``<loc>`` URL (forces https, collapses www/apex host
    variants, strips trailing slash) and applies the article path filter
    — which now correctly includes dotted/``.html`` slugs.

    Returns:
        Article URLs in sitemap order.  Returns ``[]`` on fetch or parse error.
    """
    settings = get_settings()
    base = settings.blog_base_url
    canonical_host = urlparse(base).netloc

    try:
        with urlopen(f"{base}/sitemap.xml", timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            tree = ElementTree.parse(resp)
    except Exception as exc:
        logger.warning("Could not fetch sitemap: %s", exc)
        return []

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: list[str] = []
    for loc in tree.findall(".//sm:loc", ns):
        raw_url = (loc.text or "").strip()
        norm = _normalize_url(raw_url, canonical_host)
        if norm and _is_article_url(norm):
            urls.append(norm)

    logger.info("list_sitemap_urls: found %d article URLs", len(urls))
    return urls


def list_paginator_urls() -> list[str]:
    """Walk ``/blog/page/N`` and return deduplicated article URLs.

    Normalizes hrefs (host, scheme, trailing slash) before collecting, so
    apex-linked articles (``bitovi.com``) are correctly equated with their
    canonical ``www.bitovi.com`` form.

    Returns:
        Deduplicated article URLs in first-seen order.
        Returns ``[]`` on connection error.
    """
    settings = get_settings()
    base = settings.blog_base_url
    canonical_host = urlparse(base).netloc

    raw = _list_paginator_raw(base, canonical_host)
    seen: set[str] = set()
    result: list[str] = []
    for u in raw:
        if u not in seen:
            seen.add(u)
            result.append(u)

    logger.info("list_paginator_urls: found %d article URLs across paginator pages", len(result))
    return result


def discover_article_urls(*, strict: bool = True) -> list[str]:
    """Return the deduplicated union of sitemap and paginator article URLs.

    Uses an **asymmetric reconcile gate**:

    - ``sitemap_only`` non-empty **+** ``strict=True`` → raises
      :exc:`DiscoveryMismatchError`.  The sitemap claiming URLs the live
      paginator did not confirm may indicate a broken crawl.
    - ``paginator_only`` non-empty → always included in the union silently
      (articles published after the last sitemap update are expected here).

    With current data (sitemap=461, paginator=462, sitemap_only=0,
    paginator_only=1) the strict gate always passes.

    Args:
        strict: When ``True`` (default), raise on ``sitemap_only`` discrepancy.
            When ``False``, log a WARNING and return the union regardless.

    Returns:
        Deduplicated article URLs — sitemap order first, then any
        paginator-only additions in discovery order.

    Raises:
        DiscoveryMismatchError: When ``strict=True`` and the sitemap contains
            URLs the paginator did not find.
    """
    settings = get_settings()
    base = settings.blog_base_url
    canonical_host = urlparse(base).netloc

    sitemap_urls = list_sitemap_urls()
    paginator_raw = _list_paginator_raw(base, canonical_host)
    report = reconcile(sitemap_urls, paginator_raw)

    if report.paginator_only:
        logger.info(
            "Discovery: %d article(s) in paginator but not in sitemap — included in union: %s",
            len(report.paginator_only),
            sorted(report.paginator_only)[:5],
        )

    if report.sitemap_only:
        if strict:
            raise DiscoveryMismatchError(report.sitemap_only, report.paginator_only)
        logger.warning(
            "Discovery mismatch (strict=False): %d sitemap-only URL(s) not found via paginator",
            len(report.sitemap_only),
        )

    logger.info("discover_article_urls: %d total article URLs in union", len(report.union))
    return report.union


# ---------------------------------------------------------------------------
# CLI entrypoint — live reconcile diagnostic
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a live reconcile and print the full diagnostic report.

    Useful as a zero-embed health check before embedding or to investigate
    sitemap/paginator divergence.  Run with::

        python -m ingest.discovery
        make discover-check
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    base = settings.blog_base_url
    canonical_host = urlparse(base).netloc

    logger.info("Fetching sitemap …")
    sitemap = list_sitemap_urls()
    logger.info("Crawling paginator …")
    paginator_raw = _list_paginator_raw(base, canonical_host)
    report = reconcile(sitemap, paginator_raw)

    print(f"\n{'=' * 60}")
    print(f"  Sitemap unique articles    : {len(report.sitemap_unique)}")
    print(f"  Paginator raw links        : {report.paginator_raw_count}")
    print(f"  Paginator unique articles  : {len(report.paginator_unique)}")
    print(f"  Union total                : {len(report.union)}")
    print(f"  Repeated (featured) URLs   : {len(report.repeated)}")
    print(f"  Sitemap-only               : {len(report.sitemap_only)}")
    print(f"  Paginator-only             : {len(report.paginator_only)}")
    print(f"{'=' * 60}")

    if report.repeated:
        print("\nTop repeated (featured-link) URLs (max 20):")
        for u, c in sorted(report.repeated.items(), key=lambda x: -x[1])[:20]:
            print(f"  {c:>3}×  {u}")

    if report.sitemap_only:
        print(f"\n⚠  Sitemap-only ({len(report.sitemap_only)}) — paginator did not confirm:")
        for u in sorted(report.sitemap_only)[:20]:
            print(f"       {u}")

    if report.paginator_only:
        print(f"\n↑  Paginator-only ({len(report.paginator_only)}) — not in sitemap (included):")
        for u in sorted(report.paginator_only)[:20]:
            print(f"       {u}")

    if not report.sitemap_only and not report.paginator_only:
        print("\n✓  Both sources agree — union is authoritative.")


if __name__ == "__main__":
    main()
