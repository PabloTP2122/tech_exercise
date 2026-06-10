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
    def test_empty_rows_no_slug(self) -> None:
        assert render_enumeration([]) == "No articles found."

    def test_empty_rows_with_slug(self) -> None:
        assert render_enumeration([], slug="devops") == "No articles found."

    def test_with_slug_summary_sentence(self) -> None:
        rows = [
            {"title": "A", "url": "https://bitovi.com/a"},
            {"title": "B", "url": "https://bitovi.com/b"},
        ]
        result = render_enumeration(rows, slug="devops")
        assert "2 articles" in result
        assert "devops" in result
        assert "complete list" in result

    def test_without_slug_recent_articles(self) -> None:
        rows = [{"title": "X", "url": "https://bitovi.com/x"}]
        result = render_enumeration(rows, slug=None)
        assert "1 recent article" in result
        assert "complete list" in result

    def test_singular_noun(self) -> None:
        rows = [{"title": "Only One", "url": "https://bitovi.com/one"}]
        result = render_enumeration(rows, slug="ai")
        assert "1 article" in result  # slug path: "1 article tagged ai"
        assert "articles" not in result

    def test_plural_noun(self) -> None:
        rows = [
            {"title": "A", "url": "https://bitovi.com/a"},
            {"title": "B", "url": "https://bitovi.com/b"},
        ]
        result = render_enumeration(rows, slug="ai")
        assert "2 articles" in result

    def test_no_inline_url_in_answer(self) -> None:
        """URLs must not appear inline — they belong in the REFERENCES list."""
        rows = [{"title": "Article", "url": "https://bitovi.com/blog/article"}]
        result = render_enumeration(rows, slug="react")
        assert "http" not in result

    def test_default_slug_none(self) -> None:
        """Calling without slug= kwarg uses None default — no crash."""
        rows = [{"title": "T", "url": "https://bitovi.com/t"}]
        result = render_enumeration(rows)
        assert "1 recent article" in result


class TestRenderRecency:
    def test_empty_items(self) -> None:
        assert render_recency([]) == "No recent articles found."

    def test_single_item_contains_title_and_date(self) -> None:
        items = [
            {"title": "Latest Post", "url": "https://bitovi.com/latest", "date": "June 1, 2026"}
        ]
        result = render_recency(items)
        assert "Latest Post" in result
        assert "June 1, 2026" in result

    def test_single_item_no_inline_url(self) -> None:
        """URL must not appear inline — it belongs in the REFERENCES list."""
        items = [
            {"title": "Latest Post", "url": "https://bitovi.com/latest", "date": "June 1, 2026"}
        ]
        result = render_recency(items)
        assert "http" not in result

    def test_single_item_no_markdown_bold(self) -> None:
        """No `**title**` literal — frontend renders Markdown, not the template."""
        items = [{"title": "Bold Title", "url": "https://bitovi.com/b", "date": "May 1, 2026"}]
        result = render_recency(items)
        assert "**" not in result

    def test_multiple_items_lead_title_in_answer(self) -> None:
        """Lead item title must appear in the answer; secondary items go to REFERENCES only."""
        items = [
            {"title": "Newest", "url": "https://bitovi.com/new", "date": "June 8, 2026"},
            {"title": "Older", "url": "https://bitovi.com/old", "date": "May 1, 2026"},
        ]
        result = render_recency(items)
        assert "Newest" in result

    def test_multiple_items_no_inline_urls(self) -> None:
        items = [
            {"title": "A", "url": "https://bitovi.com/a", "date": "June 8, 2026"},
            {"title": "B", "url": "https://bitovi.com/b", "date": "June 7, 2026"},
        ]
        result = render_recency(items)
        assert "http" not in result

    def test_multiple_items_trailing_sentence(self) -> None:
        items = [
            {"title": "A", "url": "https://bitovi.com/a", "date": "June 8, 2026"},
            {"title": "B", "url": "https://bitovi.com/b", "date": "June 7, 2026"},
            {"title": "C", "url": "https://bitovi.com/c", "date": "June 6, 2026"},
        ]
        result = render_recency(items)
        assert "other recent posts" in result.lower()
