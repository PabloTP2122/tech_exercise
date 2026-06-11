"""Golden-query dataset for the RAG agent.

Pure data module — no pytest imports, no network. Each :class:`GoldenCase`
captures one user question and the externally observable behaviour we promise:
which route it takes, which slot values the classifier extracts, and (for the
live tier) what the answer/sources must contain.

Two consumers in tests/test_golden_queries.py:
- TestGoldenRouting (offline, default run): asserts query_type + extracted slots.
- TestGoldenEndToEnd (``golden_live``): runs the compiled graph against the live
  DB + OpenAI and asserts answers, reference-link parity, and source URLs.

Keyword assertions are deliberately loose (any-of lists, case-insensitive) so
the live tier survives normal blog content drift.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldenCase:
    """One golden question with its expected observable behaviour.

    Attributes:
        question: Raw user question, exactly as typed in the UI.
        expected_type: Route the classifier must pick.
        expected_slug: Category slug the classifier must extract (None = no slug).
        expected_direction: Recency direction slot ("newest"/"oldest"); only
            asserted when set (slot lands with the route-edge-cases work).
        expected_limit: Recency item-count slot; only asserted when set.
        expected_year: Year-filter slot; only asserted when set.
        expected_keywords: Any-of list — at least one must appear in the answer
            (case-insensitive). Empty = no keyword assertion.
        expected_url_fragment: Substring that must appear in at least one
            source URL. None = only parity (>= 1 source) is asserted.
        is_negative: True for the known-negative case: answer must be the
            canonical no-match sentence and sources must be empty.
    """

    question: str
    expected_type: str
    expected_slug: str | None = None
    expected_direction: str | None = None
    expected_limit: int | None = None
    expected_year: int | None = None
    expected_keywords: tuple[str, ...] = field(default=())
    expected_url_fragment: str | None = None
    is_negative: bool = False


# Slug list injected into classify() for the offline tier — mirrors
# tests/test_classifier.py so routing assertions never need a live DB.
KNOWN_SLUGS: list[str] = [
    "ai",
    "angular",
    "devops",
    "frontend-engineering",
    "project-management",
    "react",
]


GOLDEN: tuple[GoldenCase, ...] = (
    # --- the four queries from the exercise brief ---
    GoldenCase(
        question="What is Bitovi's latest blog post about?",
        expected_type="recency",
    ),
    GoldenCase(
        question="Can you show me all Bitovi articles about DevOps?",
        expected_type="enumeration",
        expected_slug="devops",
        expected_url_fragment="/blog/",
    ),
    GoldenCase(
        question="How many articles does Bitovi have about AI?",
        expected_type="count",
        expected_slug="ai",
        expected_keywords=("articles about ai",),
        expected_url_fragment="/blog/topic/ai",
    ),
    GoldenCase(
        question="What kind of tools does Bitovi recommend for E2E testing?",
        expected_type="semantic_qa",
        expected_keywords=("cypress", "playwright", "selenium", "testcafe", "e2e", "end-to-end"),
    ),
    # --- paraphrases ---
    GoldenCase(
        question="What's the newest post on the Bitovi blog?",
        expected_type="recency",
    ),
    GoldenCase(
        question="List all React articles",
        expected_type="enumeration",
        expected_slug="react",
        expected_url_fragment="/blog/",
    ),
    GoldenCase(
        question="Which articles cover Angular?",
        expected_type="enumeration",
        expected_slug="angular",
        expected_url_fragment="/blog/",
    ),
    GoldenCase(
        question="Give me a count of project management posts",
        expected_type="count",
        expected_slug="project-management",
        expected_url_fragment="/blog/topic/project-management",
    ),
    GoldenCase(
        question="How does Bitovi approach testing React components?",
        expected_type="semantic_qa",
        expected_slug="react",
        expected_keywords=("react",),
    ),
    # --- known negative: must hit the no-match path, never hallucinate ---
    GoldenCase(
        question="Do you have a recipe for lasagna?",
        expected_type="semantic_qa",
        is_negative=True,
    ),
)
