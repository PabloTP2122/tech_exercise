"""Unit tests for agent/retrieval.py — all offline (mocked engine, no network).

Coverage:
- build_or_query: tokenization, operator-token stripping, punctuation safety.
- keyword_search_chunks: SQL shape, Document mapping, fail-open on errors.
- rrf_fuse: ordering, overlap boost, disjoint/empty inputs.
- dedupe_chunks: exact-duplicate drop, per-article cap, order stability.
"""

from typing import Any
from unittest.mock import MagicMock

import sqlalchemy as sa
from langchain_core.documents import Document

from agent.retrieval import build_or_query, dedupe_chunks, keyword_search_chunks, rrf_fuse


def _mock_engine(rows: list[Any] | None = None) -> MagicMock:
    """Return a mock engine whose connect() yields a mock connection."""
    conn = MagicMock()
    result = MagicMock()
    result.fetchall.return_value = rows or []
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value = result
    engine = MagicMock(spec=sa.Engine)
    engine.connect.return_value = conn
    return engine


def _doc(
    content: str, url: str = "https://www.bitovi.com/blog/a", id_: str | None = None
) -> Document:
    return Document(id=id_, page_content=content, metadata={"source_url": url, "title": "T"})


# ---------------------------------------------------------------------------
# build_or_query
# ---------------------------------------------------------------------------


class TestBuildOrQuery:
    def test_joins_tokens_with_or(self) -> None:
        assert build_or_query("E2E testing tools") == "E2E OR testing OR tools"

    def test_strips_punctuation(self) -> None:
        assert build_or_query("What's testing? (tools!)") == "What's OR testing OR tools"

    def test_drops_tsquery_operator_tokens(self) -> None:
        assert build_or_query("react or angular and not vue") == "react OR angular OR vue"

    def test_empty_question(self) -> None:
        assert build_or_query("?!.,") == ""

    def test_caps_token_count(self) -> None:
        question = " ".join(f"word{i}" for i in range(40))
        assert len(build_or_query(question).split(" OR ")) == 16


# ---------------------------------------------------------------------------
# keyword_search_chunks
# ---------------------------------------------------------------------------


class TestKeywordSearchChunks:
    def test_maps_rows_to_documents(self) -> None:
        rows = [("id-1", "chunk text", "https://www.bitovi.com/blog/a", "Title A")]
        engine = _mock_engine(rows)
        docs = keyword_search_chunks(engine, "chunks", "testing tools", 4)
        assert len(docs) == 1
        assert docs[0].id == "id-1"
        assert docs[0].page_content == "chunk text"
        assert docs[0].metadata == {
            "source_url": "https://www.bitovi.com/blog/a",
            "title": "Title A",
        }

    def test_sql_uses_websearch_to_tsquery_and_bound_params(self) -> None:
        engine = _mock_engine()
        keyword_search_chunks(engine, "chunks", "testing tools", 4)
        conn = engine.connect.return_value
        sql_arg, params = conn.execute.call_args[0]
        assert "websearch_to_tsquery" in str(sql_arg)
        assert "ts_rank_cd" in str(sql_arg)
        assert params == {"q": "testing OR tools", "n": 4}

    def test_empty_query_returns_empty_without_db_call(self) -> None:
        engine = _mock_engine()
        assert keyword_search_chunks(engine, "chunks", "?!", 4) == []
        engine.connect.assert_not_called()

    def test_zero_limit_returns_empty(self) -> None:
        engine = _mock_engine()
        assert keyword_search_chunks(engine, "chunks", "testing", 0) == []

    def test_db_error_fails_open_to_empty(self) -> None:
        engine = MagicMock(spec=sa.Engine)
        engine.connect.side_effect = RuntimeError("db down")
        assert keyword_search_chunks(engine, "chunks", "testing", 4) == []

    def test_none_columns_become_empty_strings(self) -> None:
        rows = [("id-1", None, None, None)]
        docs = keyword_search_chunks(_mock_engine(rows), "chunks", "testing", 4)
        assert docs[0].page_content == ""
        assert docs[0].metadata == {"source_url": "", "title": ""}


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------


class TestRrfFuse:
    def test_doc_in_both_lists_outranks_single_list_docs(self) -> None:
        shared = _doc("shared", id_="s")
        v_only = _doc("vector only", id_="v")
        k_only = _doc("keyword only", id_="k")
        fused = rrf_fuse([v_only, shared], [shared, k_only])
        assert fused[0].id == "s"

    def test_disjoint_lists_keep_all_docs(self) -> None:
        v = [_doc("a", id_="a"), _doc("b", id_="b")]
        k = [_doc("c", id_="c")]
        assert {d.id for d in rrf_fuse(v, k)} == {"a", "b", "c"}

    def test_empty_keyword_list_preserves_vector_order(self) -> None:
        v = [_doc("a", id_="a"), _doc("b", id_="b")]
        assert [d.id for d in rrf_fuse(v, [])] == ["a", "b"]

    def test_both_empty(self) -> None:
        assert rrf_fuse([], []) == []

    def test_fallback_key_without_id(self) -> None:
        """Docs without store ids fuse on url + content prefix (no crash, no dupes)."""
        a1 = _doc("same chunk", url="https://x/1")
        a2 = _doc("same chunk", url="https://x/1")
        fused = rrf_fuse([a1], [a2])
        assert len(fused) == 1


# ---------------------------------------------------------------------------
# dedupe_chunks
# ---------------------------------------------------------------------------


class TestDedupeChunks:
    def test_drops_exact_duplicate_content(self) -> None:
        docs = [_doc("same"), _doc("same"), _doc("other")]
        assert len(dedupe_chunks(docs, max_per_article=9)) == 2

    def test_caps_chunks_per_article(self) -> None:
        url = "https://www.bitovi.com/blog/big"
        docs = [_doc(f"chunk {i}", url=url) for i in range(4)] + [
            _doc("from another", url="https://www.bitovi.com/blog/other")
        ]
        kept = dedupe_chunks(docs, max_per_article=2)
        assert len([d for d in kept if d.metadata["source_url"] == url]) == 2
        assert len(kept) == 3

    def test_preserves_order(self) -> None:
        docs = [
            _doc("a", url="https://x/1"),
            _doc("b", url="https://x/2"),
            _doc("c", url="https://x/1"),
        ]
        assert [d.page_content for d in dedupe_chunks(docs, max_per_article=2)] == ["a", "b", "c"]

    def test_empty_input(self) -> None:
        assert dedupe_chunks([], max_per_article=2) == []
