"""Offline tests for agent/prompts.py — no DB, no LLM."""

from agent.prompts import (
    NO_MATCH_RESPONSE,
    SEMANTIC_QA_SYSTEM_PROMPT,
    render_count,
    render_enumeration,
    render_recency,
)


class TestNoMatchResponse:
    def test_exact_sentence(self) -> None:
        assert NO_MATCH_RESPONSE == "I couldn't find information about that in the blog."


class TestSemanticQaSystemPrompt:
    def test_contains_data_source_delimiter(self) -> None:
        assert "<DATA_SOURCE>" in SEMANTIC_QA_SYSTEM_PROMPT

    def test_contains_defensive_instruction(self) -> None:
        assert "never" in SEMANTIC_QA_SYSTEM_PROMPT.lower()

    def test_format_placeholders_work(self) -> None:
        rendered = SEMANTIC_QA_SYSTEM_PROMPT.format(question="What is LangGraph?", context="ctx")
        assert "What is LangGraph?" in rendered
        assert "ctx" in rendered

    def test_contains_no_match_sentence(self) -> None:
        assert "I couldn't find information about that in the blog." in SEMANTIC_QA_SYSTEM_PROMPT


class TestRenderCount:
    def test_plural_with_slug(self) -> None:
        assert render_count(42, "ai") == "There are 42 articles about ai on the Bitovi blog."

    def test_singular_with_slug(self) -> None:
        assert render_count(1, "devops") == "There is 1 article about devops on the Bitovi blog."

    def test_zero_with_slug(self) -> None:
        assert render_count(0, "ai") == "There are no articles about ai on the Bitovi blog."

    def test_plural_no_slug(self) -> None:
        assert render_count(9, None) == "There are 9 articles on the Bitovi blog."

    def test_singular_no_slug(self) -> None:
        assert render_count(1, None) == "There is 1 article on the Bitovi blog."

    def test_zero_no_slug(self) -> None:
        assert render_count(0, None) == "There are no articles on the Bitovi blog."


class TestRenderEnumeration:
    def test_empty_rows(self) -> None:
        assert render_enumeration([]) == "No articles found."

    def test_single_row(self) -> None:
        rows = [{"title": "Intro to LangGraph", "url": "https://bitovi.com/blog/langgraph"}]
        result = render_enumeration(rows)
        assert result == "1. Intro to LangGraph — https://bitovi.com/blog/langgraph"

    def test_multiple_rows_numbered_and_ordered(self) -> None:
        rows = [
            {"title": "First", "url": "https://bitovi.com/1"},
            {"title": "Second", "url": "https://bitovi.com/2"},
            {"title": "Third", "url": "https://bitovi.com/3"},
        ]
        lines = render_enumeration(rows).splitlines()
        assert lines[0].startswith("1.")
        assert lines[1].startswith("2.")
        assert lines[2].startswith("3.")
        assert "First" in lines[0]
        assert "Third" in lines[2]

    def test_order_preserved(self) -> None:
        rows = [
            {"title": "B Article", "url": "https://bitovi.com/b"},
            {"title": "A Article", "url": "https://bitovi.com/a"},
        ]
        result = render_enumeration(rows)
        assert result.index("B Article") < result.index("A Article")


class TestRenderRecency:
    def test_empty_items(self) -> None:
        assert render_recency([]) == "No recent articles found."

    def test_single_item(self) -> None:
        items = [{"title": "Latest Post", "url": "https://bitovi.com/latest", "date": "2026-06-01"}]
        result = render_recency(items)
        assert "Latest Post" in result
        assert "2026-06-01" in result
        assert "https://bitovi.com/latest" in result

    def test_multiple_items_first_is_prominent(self) -> None:
        items = [
            {"title": "Newest", "url": "https://bitovi.com/new", "date": "2026-06-08"},
            {"title": "Older", "url": "https://bitovi.com/old", "date": "2026-05-01"},
        ]
        result = render_recency(items)
        assert result.index("Newest") < result.index("Older")
        assert "Older" in result
        assert "2026-05-01" in result

    def test_multiple_items_contains_other_section(self) -> None:
        items = [
            {"title": "A", "url": "https://bitovi.com/a", "date": "2026-06-08"},
            {"title": "B", "url": "https://bitovi.com/b", "date": "2026-06-07"},
            {"title": "C", "url": "https://bitovi.com/c", "date": "2026-06-06"},
        ]
        result = render_recency(items)
        assert "Other recent posts" in result
        assert "B" in result
        assert "C" in result
