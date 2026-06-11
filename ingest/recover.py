"""Regex-based recovery of metadata fields from raw JSON-LD blocks.

Single responsibility: when structured JSON-LD parsing fails or the
URL-matched ``BlogPosting`` lacks a field, rescue the value straight from the
raw ``<script type="application/ld+json">`` source text.

Two failure modes motivate this module (see spec/adr/0007):

1. **Malformed JSON-LD**: an unescaped ``"`` inside a field value (common in
   Bitovi's ``headline``) makes ``json.loads`` throw, so the whole block is
   discarded — but the desired string is still present in the raw source.
2. **URL-mismatch**: the URL-matched ``BlogPosting`` lacks the field while
   another block on the same page (e.g. a related post or alternate
   representation) carries it.

Public API
----------
_recover_date_published(soup)
    First ``datePublished`` value found in any JSON-LD block, or ``""``.
_recover_author(soup)
    First ``author.name`` value found in any JSON-LD block, or ``""``.
"""

import re

from bs4 import BeautifulSoup

_DATE_PUBLISHED_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
# Matches: "author": { ... "name": "Value" ... }  (DOTALL so it spans multiple lines)
_AUTHOR_NAME_RE = re.compile(r'"author"\s*:\s*\{[^}]*?"name"\s*:\s*"([^"]+)"', re.DOTALL)


def _recover_jsonld_field(soup: BeautifulSoup, pattern: re.Pattern[str]) -> str:
    """Scan all JSON-LD ``<script>`` blocks for the first match of ``pattern``.

    Iterates blocks in document order and returns ``pattern.search(raw).group(1)``
    for the first matching block, or ``""`` if no block matches — so the primary
    article's value (typically block 0 or 1) is preferred.

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

    Used as a fallback when ``extract_metadata`` returns an empty
    ``published_at``.

    Args:
        soup: Parsed page soup.

    Returns:
        The first ``datePublished`` value found in any JSON-LD block, or ``""``
        if none is present.
    """
    return _recover_jsonld_field(soup, _DATE_PUBLISHED_RE)


def _recover_author(soup: BeautifulSoup) -> str:
    """Scan all JSON-LD blocks for the ``author.name`` field via regex.

    Used as a fallback when ``extract_metadata`` returns an empty ``author``.

    Args:
        soup: Parsed page soup.

    Returns:
        The first ``author.name`` value found in any JSON-LD block, or ``""``
        if none is present.
    """
    return _recover_jsonld_field(soup, _AUTHOR_NAME_RE)
