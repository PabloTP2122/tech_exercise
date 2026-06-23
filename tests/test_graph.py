"""Tests for agent/graph.py, nodes, routes, and SQL helpers.

All tests are fully offline — no DB, no LLM, no network.  Deps are injected
via the node-factory pattern and patched where the node queries the DB or LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from langchain_core.documents import Document

from agent.graph import build_graph
from agent.nodes.catalog import make_sql_count_node, make_sql_enumerate_node
from agent.nodes.recency import make_hybrid_recency_node
from agent.nodes.semantic_qa import (
    _QAResult,
    make_generate_node,
    make_no_match_node,
    make_relevance_gate_fn,
    make_retrieve_node,
)
from agent.prompts import NO_MATCH_RESPONSE
from agent.routes.route import route_query
from ingest.catalog_db import (
    count_articles_by_slug,
    count_articles_by_year,
    escape_like,
    get_recent_articles,
    list_articles_by_slug,
    list_articles_by_year,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "https://www.bitovi.com"


def _doc(title: str = "Test Article", url: str = "https://www.bitovi.com/blog/test") -> Document:
    return Document(
        page_content="Some content.",
        metadata={"title": title, "source_url": url},
    )


def _state(**kwargs: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "question": "test question",
        "query_type": "semantic_qa",
        "category_slug": None,
        "docs": [],
        "scores": [],
        "answer": "",
        "sources": [],
    }
    defaults.update(kwargs)
    return defaults


def _mock_engine(count: int = 0, rows: list[Any] | None = None) -> MagicMock:
    """Return a mock engine whose connect() yields a mock connection."""
    conn = MagicMock()
    result = MagicMock()
    result.fetchone.return_value = (count,)
    result.fetchall.return_value = rows or []
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value = result
    engine = MagicMock(spec=sa.Engine)
    engine.connect.return_value = conn
    return engine


# ---------------------------------------------------------------------------
# retrieve node
# ---------------------------------------------------------------------------


class TestMakeRetrieveNode:
    def test_returns_docs_and_scores(self) -> None:
        doc = _doc()
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]
        node = make_retrieve_node(vs)
        result = node(_state())
        assert result["docs"] == [doc]
        assert result["scores"] == [0.9]

    def test_empty_results(self) -> None:
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = []
        node = make_retrieve_node(vs)
        result = node(_state())
        assert result["docs"] == []
        assert result["scores"] == []

    def test_passes_question_to_vector_store(self) -> None:
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = []
        node = make_retrieve_node(vs)
        node(_state(question="E2E testing tools"))
        vs.similarity_search_with_relevance_scores.assert_called_once_with("E2E testing tools", k=4)

    def test_top_k_is_configurable(self) -> None:
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = []
        node = make_retrieve_node(vs, top_k=7)
        node(_state(question="E2E testing tools"))
        vs.similarity_search_with_relevance_scores.assert_called_once_with("E2E testing tools", k=7)

    def test_hybrid_fuses_keyword_docs_but_scores_stay_vector_only(self) -> None:
        """Keyword hits enrich docs; scores stay vector-only (no-match invariant)."""
        vec_doc = _doc(url="https://www.bitovi.com/blog/vec")
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [(vec_doc, 0.9)]
        kw_doc = Document(
            id="kw-1",
            page_content="keyword chunk",
            metadata={"source_url": "https://www.bitovi.com/blog/kw", "title": "KW"},
        )
        with patch(
            "agent.nodes.semantic_qa.keyword_search_chunks", return_value=[kw_doc]
        ) as kw_mock:
            node = make_retrieve_node(vs, engine=MagicMock(spec=sa.Engine))
            result = node(_state(question="E2E testing tools"))
        kw_mock.assert_called_once()
        urls = {d.metadata["source_url"] for d in result["docs"]}
        assert urls == {"https://www.bitovi.com/blog/vec", "https://www.bitovi.com/blog/kw"}
        assert result["scores"] == [0.9]

    def test_keyword_k_zero_disables_hybrid(self) -> None:
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = []
        with patch("agent.nodes.semantic_qa.keyword_search_chunks") as kw_mock:
            node = make_retrieve_node(vs, engine=MagicMock(spec=sa.Engine), keyword_k=0)
            node(_state())
        kw_mock.assert_not_called()

    def test_dedupe_caps_chunks_per_article(self) -> None:
        url = "https://www.bitovi.com/blog/big"
        docs = [
            Document(page_content=f"chunk {i}", metadata={"source_url": url, "title": "Big"})
            for i in range(4)
        ]
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [(d, 0.9) for d in docs]
        node = make_retrieve_node(vs, max_per_article=2)
        result = node(_state())
        assert len(result["docs"]) == 2
        assert len(result["scores"]) == 4  # scores untouched by dedupe


# ---------------------------------------------------------------------------
# relevance gate
# ---------------------------------------------------------------------------


class TestRelevanceGateFn:
    def test_high_score_routes_to_generate(self) -> None:
        fn = make_relevance_gate_fn(threshold=0.75)
        assert fn(_state(scores=[0.8])) == "generate"

    def test_low_score_routes_to_no_match(self) -> None:
        fn = make_relevance_gate_fn(threshold=0.75)
        assert fn(_state(scores=[0.5])) == "no_match"

    def test_exactly_at_threshold_routes_to_generate(self) -> None:
        fn = make_relevance_gate_fn(threshold=0.75)
        assert fn(_state(scores=[0.75])) == "generate"

    def test_empty_scores_routes_to_no_match(self) -> None:
        fn = make_relevance_gate_fn(threshold=0.75)
        assert fn(_state(scores=[])) == "no_match"

    def test_multiple_scores_uses_max(self) -> None:
        fn = make_relevance_gate_fn(threshold=0.75)
        assert fn(_state(scores=[0.5, 0.3, 0.9])) == "generate"


# ---------------------------------------------------------------------------
# no_match node
# ---------------------------------------------------------------------------


class TestNoMatchNode:
    def test_sets_no_match_response(self) -> None:
        node = make_no_match_node()
        result = node(_state())
        assert result["answer"] == NO_MATCH_RESPONSE
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# generate node
# ---------------------------------------------------------------------------


class TestMakeGenerateNode:
    def test_returns_answer(self) -> None:
        llm = _make_structured_llm("Answer text.", cited_urls=["https://www.bitovi.com/blog/test"])
        node = make_generate_node(llm)
        result = node(_state(docs=[_doc()], scores=[0.9]))
        assert result["answer"] == "Answer text."

    def test_sources_extracted_from_docs(self) -> None:
        doc = _doc("My Article", "https://www.bitovi.com/blog/my-article")
        llm = _make_structured_llm(cited_urls=["https://www.bitovi.com/blog/my-article"])
        node = make_generate_node(llm)
        result = node(_state(docs=[doc], scores=[0.9]))
        assert result["sources"] == [
            {"title": "My Article", "url": "https://www.bitovi.com/blog/my-article"}
        ]

    def test_doc_without_source_url_excluded_from_sources(self) -> None:
        llm = _make_structured_llm()
        node = make_generate_node(llm)
        doc = Document(page_content="content", metadata={"title": "T"})  # no source_url
        result = node(_state(docs=[doc], scores=[0.9]))
        assert result["sources"] == []

    def test_structured_output_called_once(self) -> None:
        llm = _make_structured_llm(cited_urls=["https://www.bitovi.com/blog/test"])
        node = make_generate_node(llm)
        node(_state(docs=[_doc()], scores=[0.9]))
        llm.with_structured_output.assert_called_once()
        llm.with_structured_output.return_value.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# sql_count node
# ---------------------------------------------------------------------------


class TestMakeSqlCountNode:
    def test_with_slug_returns_topic_page_link(self) -> None:
        engine = _mock_engine(count=5)
        with patch("agent.nodes.catalog.count_articles_by_slug", return_value=5):
            node = make_sql_count_node(engine)
            result = node(_state(query_type="count", category_slug="devops"))
        assert len(result["sources"]) == 1
        assert result["sources"][0]["url"].endswith("/blog/topic/devops/page/1")

    def test_no_slug_returns_blog_root_link(self) -> None:
        with patch("agent.nodes.catalog.get_article_count", return_value=100):
            node = make_sql_count_node(_mock_engine())
            result = node(_state(query_type="count", category_slug=None))
        assert result["sources"][0]["url"].endswith("/blog")

    def test_answer_contains_count(self) -> None:
        with patch("agent.nodes.catalog.count_articles_by_slug", return_value=7):
            node = make_sql_count_node(_mock_engine())
            result = node(_state(query_type="count", category_slug="ai"))
        assert "7" in result["answer"]

    def test_sources_non_empty(self) -> None:
        with patch("agent.nodes.catalog.count_articles_by_slug", return_value=3):
            node = make_sql_count_node(_mock_engine())
            result = node(_state(query_type="count", category_slug="react"))
        assert len(result["sources"]) >= 1
        assert "title" in result["sources"][0]
        assert "url" in result["sources"][0]


# ---------------------------------------------------------------------------
# sql_enumerate node
# ---------------------------------------------------------------------------


class TestMakeSqlEnumerateNode:
    def test_returns_rows_as_sources(self) -> None:
        rows = [
            {"title": "Article A", "url": "https://www.bitovi.com/blog/a"},
            {"title": "Article B", "url": "https://www.bitovi.com/blog/b"},
        ]
        with patch("agent.nodes.catalog.list_articles_by_slug", return_value=rows):
            node = make_sql_enumerate_node(_mock_engine())
            result = node(_state(query_type="enumeration", category_slug="devops"))
        assert result["sources"] == rows

    def test_answer_is_summary_sentence_not_list(self) -> None:
        """Answer must be a summary sentence — not a numbered list with inline URLs."""
        rows = [{"title": "X", "url": "https://www.bitovi.com/blog/x"}]
        with patch("agent.nodes.catalog.list_articles_by_slug", return_value=rows):
            node = make_sql_enumerate_node(_mock_engine())
            result = node(_state(query_type="enumeration", category_slug="devops"))
        assert "devops" in result["answer"]
        assert "complete list" in result["answer"]
        assert "http" not in result["answer"]  # no inline URLs

    def test_empty_rows_returns_no_articles_message(self) -> None:
        with patch("agent.nodes.catalog.list_articles_by_slug", return_value=[]):
            node = make_sql_enumerate_node(_mock_engine())
            result = node(_state(query_type="enumeration", category_slug="devops"))
        assert "No articles" in result["answer"]

    def test_html_entities_unescaped_in_sources(self) -> None:
        """&amp; in DB titles must become & in sources (never raw HTML entities)."""
        rows = [{"title": "CI/CD &amp; DevOps", "url": "https://www.bitovi.com/blog/cicd"}]
        with patch("agent.nodes.catalog.list_articles_by_slug", return_value=rows):
            node = make_sql_enumerate_node(_mock_engine())
            result = node(_state(query_type="enumeration", category_slug="devops"))
        assert result["sources"][0]["title"] == "CI/CD & DevOps"
        assert "&amp;" not in result["sources"][0]["title"]


# ---------------------------------------------------------------------------
# hybrid_recency node
# ---------------------------------------------------------------------------


_SQL_ROW = {
    "title": "SQL Article",
    "url": "https://www.bitovi.com/blog/sql-art",
    "published_at": "2026-01-01 00:00:00",
}
_RSS_ITEM_NEWER = {
    "title": "RSS Article",
    "link": "https://www.bitovi.com/blog/rss-art",
    "description": "d",
    "pubDate": "Mon, 02 Jun 2026 10:00:00 GMT",
}
_RSS_ITEM_OLDER = {
    "title": "Old RSS",
    "link": "https://www.bitovi.com/blog/old",
    "description": "d",
    "pubDate": "Mon, 01 Jan 2024 00:00:00 GMT",
}


class TestMakeHybridRecencyNode:
    def test_sql_wins_when_rss_empty(self) -> None:
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert result["sources"][0]["url"] == _SQL_ROW["url"]

    def test_rss_wins_when_newer(self) -> None:
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[_RSS_ITEM_NEWER]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert result["sources"][0]["url"] == _RSS_ITEM_NEWER["link"]

    def test_sql_wins_when_rss_older(self) -> None:
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[_RSS_ITEM_OLDER]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert result["sources"][0]["url"] == _SQL_ROW["url"]

    def test_sources_non_empty(self) -> None:
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert len(result["sources"]) >= 1
        assert "title" in result["sources"][0]
        assert "url" in result["sources"][0]

    def test_rss_date_parse_error_falls_back_to_sql(self) -> None:
        bad_rss = [
            {"title": "Bad", "link": "https://x.com/b", "description": "d", "pubDate": "not-a-date"}
        ]
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=bad_rss),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert result["sources"][0]["url"] == _SQL_ROW["url"]

    def test_answer_has_human_date_not_iso(self) -> None:
        """Answer must contain a human date, not a raw ISO timestamp."""
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert "January 1, 2026" in result["answer"]
        assert "2026-01-01 00:00:00" not in result["answer"]

    def test_answer_has_no_inline_url(self) -> None:
        """URLs must not appear in the answer prose — only in sources."""
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert "http" not in result["answer"]

    def test_answer_has_no_markdown_bold(self) -> None:
        """No **title** literals — frontend renders Markdown, not the template."""
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert "**" not in result["answer"]

    def test_html_entities_unescaped_in_sources(self) -> None:
        """&amp; in DB titles must become & in sources."""
        row_with_entity = {**_SQL_ROW, "title": "CI/CD &amp; DevOps"}
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[row_with_entity]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert result["sources"][0]["title"] == "CI/CD & DevOps"
        assert "&amp;" not in result["sources"][0]["title"]


# ---------------------------------------------------------------------------
# route_query
# ---------------------------------------------------------------------------


class TestRouteQuery:
    @pytest.mark.parametrize(
        ("query_type", "expected_node"),
        [
            ("semantic_qa", "retrieve"),
            ("count", "sql_count"),
            ("enumeration", "sql_enumerate"),
            ("recency", "hybrid_recency"),
        ],
    )
    def test_routes(self, query_type: str, expected_node: str) -> None:
        assert route_query({"query_type": query_type}) == expected_node


# ---------------------------------------------------------------------------
# Graph-level invoke tests (all deps mocked)
# ---------------------------------------------------------------------------


def _make_mock_graph() -> Any:
    """Build a graph with all external deps mocked — no DB, no LLM, no network."""
    doc = _doc("E2E Article", "https://www.bitovi.com/blog/e2e")

    vs = MagicMock()
    vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]

    llm = _make_structured_llm(
        answer="Bitovi recommends Cypress for E2E testing.",
        cited_urls=["https://www.bitovi.com/blog/e2e"],
    )

    engine = _mock_engine(count=10, rows=[("Article", "https://www.bitovi.com/blog/art", None)])

    return build_graph(
        engine=engine,
        vector_store=vs,
        llm=llm,
        known_slugs=["devops", "ai", "react"],
    ).compile()


def _make_structured_llm(
    answer: str = "Generated answer.",
    cited_urls: list[str] | None = None,
) -> MagicMock:
    """Return a mock LLM with ``with_structured_output`` wired for generate-node tests."""
    qa_result = _QAResult(answer=answer, cited_urls=cited_urls or [])
    structured = MagicMock()
    structured.invoke.return_value = qa_result
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    # fail-closed fallback path — only reached if with_structured_output().invoke raises
    response = MagicMock()
    response.content = answer
    llm.invoke.return_value = response
    return llm


class TestGraphInvoke:
    @pytest.fixture(autouse=True)
    def _patch_recency(self) -> Any:
        """Patch DB-level recency helpers for all graph-invoke tests."""
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            yield

    def test_semantic_qa_returns_answer_and_sources(self) -> None:
        g = _make_mock_graph()
        result = g.invoke({"question": "What tools for E2E testing?"})
        assert result["answer"]
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) >= 1

    def test_count_returns_answer_and_topic_link(self) -> None:
        with patch("agent.nodes.catalog.count_articles_by_slug", return_value=3):
            g = _make_mock_graph()
            result = g.invoke({"question": "How many articles about devops?"})
        assert result["answer"]
        assert len(result["sources"]) >= 1
        assert "/blog/topic/devops/page/1" in result["sources"][0]["url"]

    def test_enumeration_returns_answer_and_sources(self) -> None:
        rows = [{"title": "DevOps Post", "url": "https://www.bitovi.com/blog/do"}]
        with patch("agent.nodes.catalog.list_articles_by_slug", return_value=rows):
            g = _make_mock_graph()
            result = g.invoke({"question": "Show me all articles about devops"})
        assert result["answer"]
        assert len(result["sources"]) >= 1

    def test_recency_returns_answer_and_sources(self) -> None:
        g = _make_mock_graph()
        result = g.invoke({"question": "What is the latest blog post?"})
        assert result["answer"]
        assert len(result["sources"]) >= 1

    def test_semantic_qa_calls_structured_output(self) -> None:
        """with_structured_output should be called (not plain invoke) for semantic_qa."""
        doc = _doc()
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]
        llm = _make_structured_llm(cited_urls=["https://www.bitovi.com/blog/test"])
        engine = _mock_engine(count=0)
        g: Any = build_graph(engine=engine, vector_store=vs, llm=llm, known_slugs=[]).compile()
        g.invoke({"question": "What is Bitovi?"})
        llm.with_structured_output.assert_called_once()
        llm.invoke.assert_not_called()

    def test_count_does_not_call_llm(self) -> None:
        vs = MagicMock()
        llm = MagicMock()
        engine = _mock_engine(count=5)
        with patch("agent.nodes.catalog.get_article_count", return_value=5):
            g: Any = build_graph(engine=engine, vector_store=vs, llm=llm, known_slugs=[]).compile()
            g.invoke({"question": "How many articles are there?"})
        llm.invoke.assert_not_called()

    def test_recency_does_not_call_llm(self) -> None:
        vs = MagicMock()
        llm = MagicMock()
        engine = _mock_engine()
        g: Any = build_graph(engine=engine, vector_store=vs, llm=llm, known_slugs=[]).compile()
        g.invoke({"question": "What is the latest post?"})
        llm.invoke.assert_not_called()

    def test_all_query_types_have_sources(self) -> None:
        """Every query type must return ≥1 source (reference-link parity requirement)."""
        queries = [
            ("What tools for E2E testing?", "semantic_qa"),
            ("How many articles about devops?", "count"),
            ("Show me all articles about devops", "enumeration"),
            ("What is the latest post?", "recency"),
        ]
        with (
            patch("agent.nodes.catalog.count_articles_by_slug", return_value=3),
            patch(
                "agent.nodes.catalog.list_articles_by_slug",
                return_value=[{"title": "X", "url": "https://www.bitovi.com/blog/x"}],
            ),
        ):
            g = _make_mock_graph()
            for question, _ in queries:
                result = g.invoke({"question": question})
                assert len(result.get("sources", [])) >= 1, (
                    f"Expected ≥1 source for question: {question!r}"
                )
                for src in result["sources"]:
                    assert "title" in src, f"Missing 'title' in source: {src}"
                    assert "url" in src, f"Missing 'url' in source: {src}"

    def test_relevance_gate_no_match(self) -> None:
        """Low relevance score (< threshold=0.35) → NO_MATCH_RESPONSE, sources=[]."""
        doc = _doc()
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.20)]  # below 0.35
        llm = MagicMock()
        engine = _mock_engine()
        g: Any = build_graph(engine=engine, vector_store=vs, llm=llm, known_slugs=[]).compile()
        result = g.invoke({"question": "Some obscure question"})
        assert result["answer"] == NO_MATCH_RESPONSE
        assert result["sources"] == []
        llm.invoke.assert_not_called()

    def test_relevance_gate_strong_match_routes_to_generate(self) -> None:
        """Relevance ≈0.60 (typical on-topic match) must reach generate, not no_match."""
        doc = _doc("Cypress E2E", "https://www.bitovi.com/blog/e2e")
        vs = MagicMock()
        vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.60)]
        llm = _make_structured_llm(
            answer="Cypress is recommended.",
            cited_urls=["https://www.bitovi.com/blog/e2e"],
        )
        engine = _mock_engine()
        g: Any = build_graph(engine=engine, vector_store=vs, llm=llm, known_slugs=[]).compile()
        result = g.invoke({"question": "What tools for E2E testing?"})
        assert result["answer"] == "Cypress is recommended."
        llm.with_structured_output.assert_called_once()
        llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# SQL security tests — ADR-0008 S1/S2 (escaped bound params, no injection)
# ---------------------------------------------------------------------------


class TestSqlSecurityEscaping:
    def _capture_pattern(self, fn: Any, slug: str) -> str:
        """Call fn with a mock engine and capture the ILIKE :pat value."""
        captured: list[dict[str, str]] = []

        conn = MagicMock()
        result = MagicMock()
        result.fetchone.return_value = (0,)
        result.fetchall.return_value = []

        def capture_execute(stmt: Any, params: dict[str, str] | None = None) -> MagicMock:
            if params:
                captured.append(params)
            return result

        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute = capture_execute

        engine = MagicMock(spec=sa.Engine)
        engine.connect.return_value = conn

        fn(engine, slug)
        assert captured, "No params were captured — execute was not called with params"
        return str(captured[0].get("pat", ""))

    def _capture_sql_text(self, fn: Any, slug: str) -> str:
        """Capture the rendered SQL string passed to conn.execute."""
        captured_sql: list[str] = []

        conn = MagicMock()
        result = MagicMock()
        result.fetchone.return_value = (0,)
        result.fetchall.return_value = []

        def capture_execute(stmt: Any, params: Any = None) -> MagicMock:
            captured_sql.append(str(stmt))
            return result

        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute = capture_execute

        engine = MagicMock(spec=sa.Engine)
        engine.connect.return_value = conn

        fn(engine, slug)
        assert captured_sql, "execute was not called"
        return captured_sql[0]

    def test_count_percent_in_slug_escaped(self) -> None:
        pat = self._capture_pattern(count_articles_by_slug, "a%")
        assert "\\%" in pat
        assert pat == "%,a\\%,%"

    def test_count_underscore_in_slug_escaped(self) -> None:
        pat = self._capture_pattern(count_articles_by_slug, "a_b")
        assert "\\_" in pat
        assert pat == "%,a\\_b,%"

    def test_list_percent_in_slug_escaped(self) -> None:
        pat = self._capture_pattern(list_articles_by_slug, "x%y")
        assert pat == "%,x\\%y,%"

    def test_list_underscore_in_slug_escaped(self) -> None:
        pat = self._capture_pattern(list_articles_by_slug, "a_b")
        assert pat == "%,a\\_b,%"

    def test_sql_injection_attempt_is_bound_param(self) -> None:
        """Dangerous string must survive as a bound value — engine executes without error."""
        dangerous = "a;DROP TABLE articles;--"
        pat = self._capture_pattern(count_articles_by_slug, dangerous)
        # escape_like does not escape semicolons or dashes (only %, _, \)
        # but the value is ONLY in the bound parameter, never interpolated into SQL.
        assert "DROP TABLE" in pat  # confirms it's in the param, not the SQL string
        assert pat.startswith("%,") and pat.endswith(",%")

    def test_escape_like_backslash(self) -> None:
        assert escape_like("a\\b") == "a\\\\b"

    def test_escape_like_percent(self) -> None:
        assert escape_like("a%b") == "a\\%b"

    def test_escape_like_underscore(self) -> None:
        assert escape_like("a_b") == "a\\_b"

    def test_count_escape_clause_is_single_backslash(self) -> None:
        """ESCAPE clause must be exactly one backslash; two backslashes crash Postgres."""
        sql = self._capture_sql_text(count_articles_by_slug, "devops")
        assert "ESCAPE '\\'" in sql, f"Expected single-backslash ESCAPE in: {sql!r}"
        assert "ESCAPE '\\\\'" not in sql, f"Found 2-char ESCAPE (arity bug) in: {sql!r}"

    def test_list_escape_clause_is_single_backslash(self) -> None:
        """Same arity check for list_articles_by_slug."""
        sql = self._capture_sql_text(list_articles_by_slug, "devops")
        assert "ESCAPE '\\'" in sql, f"Expected single-backslash ESCAPE in: {sql!r}"
        assert "ESCAPE '\\\\'" not in sql, f"Found 2-char ESCAPE (arity bug) in: {sql!r}"


# ---------------------------------------------------------------------------
# Generate node — no double-wrap of DATA_SOURCE delimiters
# ---------------------------------------------------------------------------


class TestGenerateNodeNoDoubleWrap:
    def test_context_does_not_nest_data_source_tags(self) -> None:
        """Content already has <DATA_SOURCE> from ingest; generate must not re-wrap it."""
        wrapped_content = "<DATA_SOURCE>\nSome article text.\n</DATA_SOURCE>"
        doc = Document(
            page_content=wrapped_content,
            metadata={"title": "Test", "source_url": "https://www.bitovi.com/blog/t"},
        )

        captured_prompts: list[str] = []

        def fake_structured_invoke(messages: Any) -> _QAResult:
            for m in messages:
                captured_prompts.append(str(m.content))
            return _QAResult(answer="answer", cited_urls=["https://www.bitovi.com/blog/t"])

        structured = MagicMock()
        structured.invoke = fake_structured_invoke
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        node = make_generate_node(llm)
        node({"docs": [doc], "scores": [0.9], "question": "test?"})

        full_prompt = " ".join(captured_prompts)
        # Nested: <DATA_SOURCE>...<DATA_SOURCE> must NOT appear in the context block
        assert "<DATA_SOURCE>\n<DATA_SOURCE>" not in full_prompt, (
            "double-wrap detected: wrap_as_data called on already-wrapped content"
        )
        # The original context wrapper must still be present (not stripped)
        assert "<DATA_SOURCE>" in full_prompt


# ---------------------------------------------------------------------------
# Generate node — structured-output behaviour
# ---------------------------------------------------------------------------

_URL_A = "https://www.bitovi.com/blog/article-a"
_URL_B = "https://www.bitovi.com/blog/article-b"
_URL_FAKE = "https://example.com/not-in-retrieved"


class TestGenerateNodeStructuredOutput:
    """Structured-output whitelist, parity fallback, no-match guard, fail-closed."""

    def _doc_with_url(self, url: str, title: str = "Article") -> Document:
        return Document(page_content="content", metadata={"title": title, "source_url": url})

    def _node(self, answer: str, cited_urls: list[str], docs: list[Document]) -> dict[str, Any]:
        llm = _make_structured_llm(answer=answer, cited_urls=cited_urls)
        return make_generate_node(llm)(_state(docs=docs, scores=[0.9]))

    def test_subset_citation(self) -> None:
        """Model cites 1 of 2 retrieved URLs → sources contain only that URL."""
        docs = [self._doc_with_url(_URL_A, "A"), self._doc_with_url(_URL_B, "B")]
        result = self._node("Answer.", [_URL_A], docs)
        assert result["sources"] == [{"title": "A", "url": _URL_A}]

    def test_out_of_set_url_dropped(self) -> None:
        """URL not in the retrieved set is discarded; parity fallback provides the real source."""
        docs = [self._doc_with_url(_URL_A, "A")]
        result = self._node("Answer.", [_URL_FAKE], docs)
        # _URL_FAKE is not in whitelist → dropped; parity fallback uses _URL_A
        assert result["sources"] == [{"title": "A", "url": _URL_A}]
        for src in result["sources"]:
            assert src["url"] != _URL_FAKE

    def test_empty_cited_urls_triggers_parity_fallback(self) -> None:
        """Empty cited_urls with a real answer → all retrieved docs become sources."""
        docs = [self._doc_with_url(_URL_A, "A"), self._doc_with_url(_URL_B, "B")]
        result = self._node("Answer.", [], docs)
        urls = {s["url"] for s in result["sources"]}
        assert urls == {_URL_A, _URL_B}

    def test_no_match_response_returns_empty_sources(self) -> None:
        """Canonical no-match answer must produce sources=[] regardless of cited_urls."""
        docs = [self._doc_with_url(_URL_A, "A")]
        result = self._node(NO_MATCH_RESPONSE, [_URL_A], docs)
        assert result["sources"] == []
        assert result["answer"].strip() == NO_MATCH_RESPONSE

    def test_fail_closed_on_structured_output_error(self) -> None:
        """structured.invoke raising → fall back to plain llm.invoke; sources = all retrieved."""
        doc = self._doc_with_url(_URL_A, "Fallback Article")
        structured = MagicMock()
        structured.invoke.side_effect = RuntimeError("parse error")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        response = MagicMock()
        response.content = "fallback answer"
        llm.invoke.return_value = response

        result = make_generate_node(llm)(_state(docs=[doc], scores=[0.9]))
        assert result["answer"] == "fallback answer"
        assert result["sources"] == [{"title": "Fallback Article", "url": _URL_A}]
        llm.invoke.assert_called_once()

    def test_url_not_in_answer_prose(self) -> None:
        """Regression: no http-scheme URL should appear inside the answer string."""
        docs = [self._doc_with_url(_URL_A, "A")]
        result = self._node("Bitovi is a consulting firm.", [_URL_A], docs)
        assert "http" not in result["answer"]

    def test_question_is_wrapped_in_prompt(self) -> None:
        """The user question must be wrapped in <DATA_SOURCE> before prompt interpolation."""
        captured: list[str] = []

        def fake_structured_invoke(messages: Any) -> _QAResult:
            for m in messages:
                captured.append(str(m.content))
            return _QAResult(answer="Answer.", cited_urls=[_URL_A])

        structured = MagicMock()
        structured.invoke = fake_structured_invoke
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        doc = self._doc_with_url(_URL_A, "A")
        make_generate_node(llm)(_state(docs=[doc], question="What is Bitovi?", scores=[0.9]))

        full_prompt = " ".join(captured)
        assert "<DATA_SOURCE>" in full_prompt
        assert "What is Bitovi?" in full_prompt


# ---------------------------------------------------------------------------
# recency: direction (oldest), parsed limit, slug filter, parity fallback
# ---------------------------------------------------------------------------


class TestRecencyDirectionAndLimit:
    def test_oldest_skips_rss_overlay(self) -> None:
        """RSS feeds only carry newest items — oldest must never fetch the feed."""
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent") as rss_mock,
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency", recency_direction="oldest"))
        rss_mock.assert_not_called()
        assert "oldest post" in result["answer"]

    def test_direction_limit_and_slug_passed_to_sql(self) -> None:
        engine = _mock_engine()
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]) as sql_mock,
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(engine)
            node(
                _state(
                    query_type="recency",
                    recency_direction="oldest",
                    recency_limit=5,
                    category_slug="react",
                )
            )
        sql_mock.assert_called_once_with(engine, limit=5, oldest=True, slug="react")

    def test_newest_default_still_fetches_rss(self) -> None:
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[_SQL_ROW]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]) as rss_mock,
        ):
            node = make_hybrid_recency_node(_mock_engine())
            node(_state(query_type="recency", recency_limit=5))
        rss_mock.assert_called_once_with(None, limit=5)

    def test_empty_rows_with_slug_falls_back_to_topic_page(self) -> None:
        """Reference-link parity: empty result still ships one source."""
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency", category_slug="react"))
        assert result["sources"] == [
            {
                "title": "All Bitovi articles about react",
                "url": f"{BASE}/blog/topic/react/page/1",
            }
        ]

    def test_empty_rows_without_slug_falls_back_to_blog_root(self) -> None:
        with (
            patch("agent.nodes.recency.get_recent_articles", return_value=[]),
            patch("agent.nodes.recency.fetch_recent", return_value=[]),
        ):
            node = make_hybrid_recency_node(_mock_engine())
            result = node(_state(query_type="recency"))
        assert result["sources"] == [{"title": "Bitovi Blog", "url": f"{BASE}/blog"}]


# ---------------------------------------------------------------------------
# catalog nodes: year filter + enumeration parity fallback
# ---------------------------------------------------------------------------


class TestCatalogYearFilter:
    def test_count_with_year_uses_year_query(self) -> None:
        engine = _mock_engine()
        with patch("agent.nodes.catalog.count_articles_by_year", return_value=17) as year_mock:
            node = make_sql_count_node(engine)
            result = node(_state(query_type="count", year=2023))
        year_mock.assert_called_once_with(engine, 2023, slug=None)
        assert "17 articles published in 2023" in result["answer"]
        assert result["sources"] == [{"title": "Bitovi Blog", "url": f"{BASE}/blog"}]

    def test_count_with_year_and_slug(self) -> None:
        engine = _mock_engine()
        with patch("agent.nodes.catalog.count_articles_by_year", return_value=5) as year_mock:
            node = make_sql_count_node(engine)
            result = node(_state(query_type="count", year=2023, category_slug="react"))
        year_mock.assert_called_once_with(engine, 2023, slug="react")
        assert "about react published in 2023" in result["answer"]

    def test_enumerate_with_year_uses_year_query(self) -> None:
        engine = _mock_engine()
        rows = [{"title": "A", "url": f"{BASE}/blog/a"}]
        with patch("agent.nodes.catalog.list_articles_by_year", return_value=rows) as year_mock:
            node = make_sql_enumerate_node(engine)
            result = node(_state(query_type="enumeration", year=2023))
        year_mock.assert_called_once_with(engine, 2023, slug=None)
        assert "published in 2023" in result["answer"]
        assert result["sources"] == rows

    def test_enumerate_empty_result_keeps_parity(self) -> None:
        """Reference-link parity: empty enumeration still ships one source."""
        engine = _mock_engine()
        with patch("agent.nodes.catalog.list_articles_by_slug", return_value=[]):
            node = make_sql_enumerate_node(engine)
            result = node(_state(query_type="enumeration", category_slug="react"))
        assert result["answer"] == "No articles found."
        assert result["sources"] == [{"title": "Bitovi Blog", "url": f"{BASE}/blog"}]


# ---------------------------------------------------------------------------
# catalog_db SQL shape: direction-aware recency + year bounds
# ---------------------------------------------------------------------------


class TestRecencyAndYearSql:
    def _capture_sql(self, call: Any) -> tuple[str, dict[str, Any]]:
        """Run call against a mock engine; return (sql_text, params) of first execute."""
        captured: list[tuple[str, dict[str, Any]]] = []

        conn = MagicMock()
        result = MagicMock()
        result.fetchone.return_value = (0,)
        result.fetchall.return_value = []

        def capture_execute(stmt: Any, params: dict[str, Any] | None = None) -> MagicMock:
            captured.append((str(stmt), params or {}))
            return result

        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute = capture_execute
        engine = MagicMock(spec=sa.Engine)
        engine.connect.return_value = conn

        call(engine)
        assert captured, "execute was never called"
        return captured[0]

    def test_newest_orders_desc_nulls_last(self) -> None:
        sql, _ = self._capture_sql(lambda e: get_recent_articles(e))
        assert "DESC NULLS LAST" in sql

    def test_oldest_orders_asc_nulls_last(self) -> None:
        sql, _ = self._capture_sql(lambda e: get_recent_articles(e, oldest=True))
        assert "ASC NULLS LAST" in sql

    def test_slug_filter_uses_escaped_ilike(self) -> None:
        sql, params = self._capture_sql(lambda e: get_recent_articles(e, slug="re_act"))
        assert "ILIKE :pat ESCAPE" in sql
        assert params["pat"] == "%,re\\_act,%"

    def test_count_by_year_uses_half_open_tz_bounds(self) -> None:
        sql, params = self._capture_sql(lambda e: count_articles_by_year(e, 2023))
        assert "published_at >= :start" in sql
        assert "published_at < :end" in sql
        assert params["start"] == datetime(2023, 1, 1, tzinfo=UTC)
        assert params["end"] == datetime(2024, 1, 1, tzinfo=UTC)

    def test_list_by_year_with_slug_combines_filters(self) -> None:
        sql, params = self._capture_sql(lambda e: list_articles_by_year(e, 2024, slug="ai"))
        assert "published_at >= :start" in sql
        assert "ILIKE :pat ESCAPE" in sql
        assert params["pat"] == "%,ai,%"
