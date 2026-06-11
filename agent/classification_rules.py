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
        re.compile(
            r"\b(show me all|list all|list|which articles|what articles|all articles)\b",
            re.IGNORECASE,
        ),
        "enumeration",
        "Listing requests answer with the full SQL result set, not top-k chunks.",
    ),
)
