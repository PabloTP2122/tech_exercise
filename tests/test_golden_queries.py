"""Golden-query harness — two tiers over tests/golden_queries.py.

TestGoldenRouting (offline, runs in the default suite):
    classify() with injected slugs — asserts query_type and extracted slots.
    The LLM slug fallback is patched out so no network is ever attempted.

TestGoldenEndToEnd (``golden_live``, deselected by default):
    Compiled graph against the live DB + OpenAI (``make golden``). Asserts:
    - query_type per case,
    - reference-link parity: >= 1 {title, url} source for every positive case,
    - expected URL fragment appears in some source,
    - at least one expected keyword appears in the answer (no LLM judge),
    - the negative case returns the canonical no-match sentence with no sources.
"""

from typing import Any
from unittest.mock import patch

import pytest

from agent.classifier import classify
from agent.prompts import NO_MATCH_RESPONSE
from tests.golden_queries import GOLDEN, KNOWN_SLUGS, GoldenCase

_IDS = [case.question for case in GOLDEN]


# ---------------------------------------------------------------------------
# Tier 1 — offline routing assertions (default run)
# ---------------------------------------------------------------------------


class TestGoldenRouting:
    @pytest.mark.parametrize("case", GOLDEN, ids=_IDS)
    def test_routing(self, case: GoldenCase) -> None:
        with patch("agent.classifier._resolve_slug_via_llm", return_value=None):
            result = classify(case.question, known_slugs=KNOWN_SLUGS)

        assert result.query_type == case.expected_type
        assert result.category_slug == case.expected_slug
        # Slot assertions are opt-in: only checked once a case declares them
        # (direction/limit/year slots land with the route-edge-cases work).
        if case.expected_direction is not None:
            assert getattr(result, "recency_direction", None) == case.expected_direction
        if case.expected_limit is not None:
            assert getattr(result, "recency_limit", None) == case.expected_limit
        if case.expected_year is not None:
            assert getattr(result, "year", None) == case.expected_year


# ---------------------------------------------------------------------------
# Tier 2 — live end-to-end (make golden; requires db-up + OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_graph() -> Any:
    from agent.graph import build_graph

    return build_graph().compile()


@pytest.mark.golden_live
class TestGoldenEndToEnd:
    @pytest.mark.parametrize("case", GOLDEN, ids=_IDS)
    def test_end_to_end(self, compiled_graph: Any, case: GoldenCase) -> None:
        result = compiled_graph.invoke({"question": case.question})

        assert result["query_type"] == case.expected_type
        answer: str = result["answer"]
        sources: list[dict[str, str]] = result["sources"]

        if case.is_negative:
            assert answer == NO_MATCH_RESPONSE
            assert sources == []
            return

        # Reference-link parity: every positive answer ships >= 1 source.
        assert len(sources) >= 1
        assert all(s.get("url") for s in sources)

        if case.expected_url_fragment is not None:
            assert any(case.expected_url_fragment in s["url"] for s in sources)

        if case.expected_keywords:
            lowered = answer.lower()
            assert any(kw.lower() in lowered for kw in case.expected_keywords)
