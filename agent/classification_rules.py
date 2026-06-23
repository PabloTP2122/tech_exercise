"""Declarative classification rules for the query classifier — data only.

Single responsibility: hold the routing regex table and the stopword-slug set
so they can be read, reviewed, and extended without touching classifier logic.
``agent/classifier.py`` owns all matching/extraction code and imports these
tables.

Each :class:`Rule` carries a ``why`` field documenting the intent of the
pattern, so the table doubles as routing documentation.
"""

import re
from typing import Literal, NamedTuple

QueryType = Literal["semantic_qa", "enumeration", "count", "recency"]


class Rule(NamedTuple):
    """One ordered routing rule: first matching pattern wins."""

    pattern: re.Pattern[str]
    query_type: QueryType
    why: str


# Slugs that are also common English words and must never be matched by the fuzzy pass alone.
# Without this guard, "about" (a real topic slug) matches the preposition in "articles about AI",
# and Pass-2 difflib matches stopword "does" to "donejs" at cutoff 0.8.
STOPWORD_SLUGS: frozenset[str] = frozenset(
    {
        "about",
        "blog",
        "all",
        "does",
        "what",
        "how",
        "the",
        "is",
        "are",
        "post",
        "posts",
        "article",
        "articles",
        "any",
        "with",
        "for",
    }
)


# Ordered regex rules — first match wins; default is semantic_qa.
# count stays first: "how many latest articles" must count, not list (priority tests).
# All recency rules sit before enumeration so "list the oldest articles" sorts by date.
REGEX_RULES: tuple[Rule, ...] = (
    Rule(
        re.compile(r"\b(how many|number of|count)\b", re.IGNORECASE),
        "count",
        "Quantity questions answer from SQL COUNT, never from chunk retrieval.",
    ),
    Rule(
        re.compile(r"\b(latest|newest|most recent)\b", re.IGNORECASE),
        "recency",
        "Freshness questions answer from the articles catalog ordered by date.",
    ),
    Rule(
        re.compile(r"\b(oldest|earliest)\b", re.IGNORECASE),
        "recency",
        "Same date-ordered pipeline as latest, just ascending (oldest direction).",
    ),
    Rule(
        re.compile(r"\bfirst\s+(?:\w+\s+){0,2}(?:blog\s+)?(?:post|article)s?\b", re.IGNORECASE),
        "recency",
        'Guarded "first … post/article" pattern: catches "first article about react" '
        'without misrouting "which article should I read first?".',
    ),
    Rule(
        re.compile(r"\blast\s+(?:\d+\s+)?(?:blog\s+)?(?:post|article)s?\b", re.IGNORECASE),
        "recency",
        '"last post" / "last 5 posts" means newest-first; the count is parsed separately.',
    ),
    Rule(
        re.compile(
            r"\b(show me all|list all|list|which articles|what articles|all articles)\b",
            re.IGNORECASE,
        ),
        "enumeration",
        "Listing requests answer with the full SQL result set, not top-k chunks.",
    ),
    Rule(
        re.compile(
            r"\b(articles|posts)\s+(from|in|of|during|published)\b.*\b20[0-3]\d\b",
            re.IGNORECASE,
        ),
        "enumeration",
        'Year-scoped listings ("articles from 2023") enumerate; the year value is '
        "extracted separately and applied as a SQL date-range filter.",
    ),
)


# --- Slot-extraction patterns (applied after routing, per query type) -------

# Direction slot for recency: any of these flips the ORDER BY to ascending.
OLDEST_RE = re.compile(r"\b(oldest|earliest|first)\b", re.IGNORECASE)

# Item-count slot for recency: a standalone 1-2 digit number ("last 5 posts").
# Two-digit max naturally excludes years like 2023.
RECENCY_LIMIT_RE = re.compile(r"\b([1-9]\d?)\b")

# Year slot for count/enumeration: a 20XX calendar year.
YEAR_RE = re.compile(r"\b(20[0-3]\d)\b")
