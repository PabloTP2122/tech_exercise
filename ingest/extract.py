"""HTML → structured Document extraction for Bitovi blog articles.

Single responsibility: given a raw HTML page and its URL, extract all article
metadata and the clean body text, and return a :class:`~langchain_core.documents.Document`.

Public API
----------
select_blogposting(soup, url)
    Pick the correct ``BlogPosting`` JSON-LD block from a page with multiple
    ``<script type="application/ld+json">`` blocks.
extract_metadata(soup, url)
    Fallback chain: BlogPosting → OpenGraph → ``<h1>``/``<title>``.
extract_categories(soup)
    Collect topic slugs from ``/blog/topic/{slug}`` links.
build_document(url, html)
    Orchestrate extraction and return a ready-to-embed :class:`Document`.
"""

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from ingest import clean

logger = logging.getLogger(__name__)

# Metadata keys every Document must carry.
_REQUIRED_KEYS = {
    "source_url",
    "title",
    "description",
    "author",
    "published_at",
    "categories",
    "content_origin",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_ld_blocks(soup: BeautifulSoup) -> Iterator[dict[str, Any]]:
    """Yield every JSON-LD object, flattening top-level ``@graph`` arrays.

    Used for scanning all structured-data objects on a page (e.g. FAQPage,
    Organization, BlogPosting).  :func:`select_blogposting` has its own
    graph-aware logic and does **not** use this helper — it needs to
    distinguish top-level from ``@graph`` nodes.

    Args:
        soup: Parsed page soup.

    Yields:
        Individual JSON-LD dicts (top-level or from ``@graph``).
    """
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or ""
        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.debug("_iter_ld_blocks: JSON parse error — skipping block")
            continue
        if not isinstance(data, dict):
            continue
        if "@graph" in data:
            for node in data["@graph"]:
                if isinstance(node, dict):
                    yield node
        else:
            yield data


def _ld_url_matches(candidate: dict[str, Any], url: str) -> bool:
    """Return True when a JSON-LD object's URL fields match ``url``."""
    if not url:
        return False
    mep = candidate.get("mainEntityOfPage")
    mep_id: str = mep.get("@id", "") if isinstance(mep, dict) else ""
    return url in (
        candidate.get("url", ""),
        candidate.get("@id", ""),
        mep_id,
    )


# Regexes for recovering fields from raw JSON-LD strings, even malformed ones.
_DATE_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
# Matches: "author": { ... "name": "Value" ... }  (DOTALL so it spans multiple lines)
_AUTHOR_NAME_RE = re.compile(r'"author"\s*:\s*\{[^}]*?"name"\s*:\s*"([^"]+)"', re.DOTALL)

# ADR-0008 S5: ingest-time slug whitelist — only lowercase alphanumeric + hyphen.
# All real Bitovi topic slugs already conform; non-conforming slugs are dropped at extraction.
_SLUG_RE = re.compile(r"^[a-z0-9-]+$")


def _recover_jsonld_field(soup: BeautifulSoup, pattern: re.Pattern[str]) -> str:
    """Scan all JSON-LD ``<script>`` blocks for the first match of ``pattern``.

    Iterates blocks in document order and returns ``pattern.search(raw).group(1)``
    for the first matching block, or ``""`` if no block matches.

    This shared helper powers both :func:`_recover_date_published` and
    :func:`_recover_author` — the scanning logic is identical; only the
    compiled pattern differs.

    Args:
        soup: Parsed page soup.
        pattern: Compiled regex with one capturing group for the desired value.

    Returns:
        The first captured value across all JSON-LD blocks, or ``""`` if absent.
    """
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or ""
        match = pattern.search(raw)
        if match:
            return match.group(1)
    return ""


def _recover_date_published(soup: BeautifulSoup) -> str:
    """Scan all JSON-LD blocks for ``datePublished`` via regex.

    Used as a fallback when :func:`extract_metadata` returns an empty
    ``published_at``.  Handles two distinct failure modes:

    1. **Malformed JSON-LD**: an unescaped ``"`` inside a field value (common
       in Bitovi's ``headline``) makes ``json.loads`` throw, so the whole block
       is discarded — but the date string is present in the raw source text.
    2. **URL-mismatch**: the URL-matched ``BlogPosting`` lacks ``datePublished``
       while another block on the same page (e.g. a related post or alternate
       representation) carries it.

    Scans block-by-block in document order and returns the first match, so the
    primary article's date (typically block 0 or 1) is preferred.

    Args:
        soup: Parsed page soup.

    Returns:
        The first ``datePublished`` value found in any JSON-LD block, or ``""``
        if none is present.
    """
    return _recover_jsonld_field(soup, _DATE_PUBLISHED_RE)


def _recover_author(soup: BeautifulSoup) -> str:
    """Scan all JSON-LD blocks for the ``author.name`` field via regex.

    Used as a fallback when :func:`extract_metadata` returns an empty ``author``.
    Handles the same failure modes as :func:`_recover_date_published`:

    1. **Malformed JSON-LD**: unescaped ``"`` in ``headline`` causes ``json.loads``
       to throw and the block is discarded — but the author object is still present
       in the raw source text.
    2. **URL-mismatch**: the URL-matched ``BlogPosting`` lacks an ``author`` field
       while another block on the same page carries it.

    Args:
        soup: Parsed page soup.

    Returns:
        The first ``author.name`` value found in any JSON-LD block, or ``""``
        if none is present.
    """
    return _recover_jsonld_field(soup, _AUTHOR_NAME_RE)


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------


def select_blogposting(soup: BeautifulSoup, url: str) -> dict[str, Any]:
    """Return the ``BlogPosting`` JSON-LD dict for this article URL.

    Selection strategy (in priority order):

    1. A **top-level** ``BlogPosting`` (not inside ``@graph``) whose
       ``mainEntityOfPage/@id``, ``url``, or ``@id`` matches ``url``.
    2. The first top-level ``BlogPosting`` regardless of URL match.
    3. A ``@graph`` ``BlogPosting`` whose URL fields match ``url``.
    4. The first ``@graph`` ``BlogPosting``.

    This ensures that a *related-post* ``BlogPosting`` nested inside an
    ``@graph`` block is never returned when a matching top-level block exists.

    Args:
        soup: Parsed page soup.
        url: The canonical URL of the page being processed (used to match
             ``mainEntityOfPage``/``url``/``@id`` fields).

    Returns:
        The chosen ``BlogPosting`` dict, or ``{}`` if none found.
    """
    top_level: list[dict[str, Any]] = []
    graph_level: list[dict[str, Any]] = []

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or ""
        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        if "@graph" in data:
            for node in data.get("@graph", []):
                if isinstance(node, dict) and node.get("@type") == "BlogPosting":
                    graph_level.append(node)
        elif data.get("@type") == "BlogPosting":
            top_level.append(data)

    # Priority: top-level URL-match → top-level any → graph URL-match → graph any
    for candidates in (top_level, graph_level):
        url_matched = [c for c in candidates if _ld_url_matches(c, url)]
        if url_matched:
            return url_matched[0]
        if candidates:
            return candidates[0]

    return {}


def extract_metadata(soup: BeautifulSoup, url: str) -> dict[str, str]:
    """Extract article metadata using a defensive fallback chain.

    Chain (first source that provides a non-empty title wins):

    1. ``BlogPosting`` JSON-LD (preferred — structured, machine-readable).
    2. OpenGraph meta tags (``og:title``, ``og:description``,
       ``article:published_time``, ``author`` / ``article:author``).
    3. ``<h1>`` / ``<title>`` tag (last resort).

    Always returns all four keys with string values (empty string when absent).

    Args:
        soup: Parsed page soup.
        url: Canonical URL of the page (passed to :func:`select_blogposting`).

    Returns:
        Dict with keys ``title``, ``description``, ``author``, ``published_at``.
    """
    # --- 1. BlogPosting JSON-LD ------------------------------------------
    bp = select_blogposting(soup, url)
    if bp:
        author_raw = bp.get("author", "")
        author: str = (
            author_raw.get("name", "") if isinstance(author_raw, dict) else str(author_raw)
        )
        return {
            "title": str(bp.get("headline", "")),
            "description": str(bp.get("description", "")),
            "author": author,
            "published_at": str(bp.get("datePublished", "")),
        }

    logger.warning("No BlogPosting JSON-LD found for %s — trying OpenGraph", url)

    # --- 2. OpenGraph / meta tags ----------------------------------------
    def _meta(prop: str) -> str:
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
        if tag is None:
            return ""
        from bs4 import Tag

        if not isinstance(tag, Tag):
            return ""
        return str(tag.get("content", ""))

    og_title = _meta("og:title")
    if og_title:
        return {
            "title": og_title,
            "description": _meta("og:description"),
            "author": _meta("author") or _meta("article:author"),
            "published_at": _meta("article:published_time"),
        }

    logger.warning("No OpenGraph title found for %s — falling back to <h1>/<title>", url)

    # --- 3. <h1> / <title> -----------------------------------------------
    h1 = soup.find("h1")
    title_tag = soup.find("title")
    fallback_title = (
        h1.get_text(strip=True) if h1 else (title_tag.get_text(strip=True) if title_tag else "")
    )
    return {
        "title": fallback_title,
        "description": "",
        "author": "",
        "published_at": "",
    }


def extract_categories(soup: BeautifulSoup) -> str:
    """Extract topic slugs from ``/blog/topic/{slug}`` links.

    Collects all unique topic slugs in document order and returns them as a
    delimited string ``",slug1,slug2,"``.  If no topic links are found,
    returns ``""`` (empty string — the article is legitimately untagged).

    Args:
        soup: Parsed page soup.

    Returns:
        Delimited categories string, e.g. ``",react,angular,"``, or ``""``
        when no topic links are present.
    """
    slugs: list[str] = []
    for a in soup.find_all("a", href=True):
        href: Any = a["href"]
        if not isinstance(href, str) or "/blog/topic/" not in href:
            continue
        slug = href.rstrip("/").split("/blog/topic/")[-1].strip().lower()
        if not slug:
            continue
        if not _SLUG_RE.match(slug):
            logger.warning("extract_categories: dropping non-conforming slug %r", slug)
            continue
        if slug not in slugs:
            slugs.append(slug)
    return "," + ",".join(slugs) + "," if slugs else ""


def build_document(url: str, html: str) -> Document:
    """Parse ``html`` and return a fully-populated :class:`Document`.

    Applies extraction in the following order:

    1. Parse with BeautifulSoup.
    2. Extract metadata via :func:`extract_metadata` (BlogPosting → OG → h1).
    3. Extract categories via :func:`extract_categories`.
    4. Clean the article body via :func:`ingest.clean.clean_html`.
    5. Validate the metadata contract (all 7 keys; ``categories`` may be ``""`` for
       legitimately untagged articles).

    The returned ``Document.page_content`` is the clean article body text
    (nav, footer, CTAs, and tracking pixels removed).

    Args:
        url: Canonical article URL (used as ``source_url`` and for JSON-LD
             matching).
        html: Raw HTML of the article page.

    Returns:
        A :class:`Document` ready for chunking and embedding.
    """
    soup = BeautifulSoup(html, "lxml")
    fields = extract_metadata(soup, url)
    categories = extract_categories(soup)

    # Backfill published_at when the normal extraction chain returned "".
    # Covers two cases: (1) malformed JSON-LD blocks json.loads discards;
    # (2) pages where the URL-matched BlogPosting lacks datePublished but
    # another block on the page carries it.
    published_at = fields["published_at"]
    if not published_at:
        recovered = _recover_date_published(soup)
        if recovered:
            logger.info("Recovered datePublished via regex for %s: %s", url, recovered)
            published_at = recovered

    # Backfill author when the normal extraction chain returned "".
    # Same failure modes as the date recovery: malformed JSON-LD blocks and
    # pages where the URL-matched BlogPosting is missing the author field.
    author = fields["author"]
    if not author:
        recovered_author = _recover_author(soup)
        if recovered_author:
            logger.info("Recovered author via regex for %s: %s", url, recovered_author)
            author = recovered_author

    if not categories:
        logger.warning("No topic links found for %s — categories empty", url)

    # clean_body returns the article Markdown AND the provenance label
    # ("blog_body" when the selector was found; "full_page_fallback" when not).
    body_result = clean.clean_body(html)

    metadata: dict[str, str] = {
        "source_url": url,
        "title": fields["title"],
        "description": fields["description"],
        "author": author,
        "published_at": published_at,
        "categories": categories,
        "content_origin": body_result.content_origin,
    }

    # Validate metadata contract — warn loudly, never silently emit bad data.
    missing = _REQUIRED_KEYS - set(metadata.keys())
    if missing:
        logger.error("build_document: missing metadata keys %s for %s", missing, url)

    return Document(page_content=body_result.text, metadata=metadata)
