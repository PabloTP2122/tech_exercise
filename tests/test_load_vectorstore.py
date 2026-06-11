"""Offline unit tests for ingest/load_vectorstore.py helpers."""

from langchain_core.documents import Document

from ingest.load_vectorstore import METADATA_COLUMN_NAMES, build_chunks, wrap_as_data

_SAMPLE = "Run `git diff` to inspect the changes."

# Full 7-key metadata matching the ingest contract (extract.build_document).
_BASE_META: dict[str, str] = {
    "source_url": "https://www.bitovi.com/blog/test-article",
    "title": "Test Article Title",
    "description": "A test description.",
    "author": "Test Author",
    "published_at": "2025-01-15 12:00:00",
    "categories": ",react,angular,",
    "content_origin": "blog_body",
}

# > 800 chars to guarantee RecursiveCharacterTextSplitter produces > 1 chunk.
_LONG_BODY = "This is a long test article paragraph. " * 35  # ~1365 chars

# < 800 chars → exactly one chunk.
_SHORT_BODY = "This is a short article."


def _doc(body: str, meta: dict[str, str] | None = None) -> Document:
    return Document(page_content=body, metadata=meta if meta is not None else dict(_BASE_META))


def test_wrap_adds_open_delimiter() -> None:
    assert wrap_as_data(_SAMPLE).startswith("<DATA_SOURCE>\n")


def test_wrap_adds_close_delimiter() -> None:
    assert wrap_as_data(_SAMPLE).endswith("\n</DATA_SOURCE>")


def test_wrap_preserves_text_verbatim() -> None:
    """Body text must be unchanged inside the delimiters."""
    result = wrap_as_data(_SAMPLE)
    assert _SAMPLE in result


def test_wrap_empty_string() -> None:
    assert wrap_as_data("") == "<DATA_SOURCE>\n\n</DATA_SOURCE>"


class TestBuildChunks:
    """Offline tests for build_chunks — the vector-view heart."""

    def test_short_doc_produces_one_chunk(self) -> None:
        assert len(build_chunks([_doc(_SHORT_BODY)])) == 1

    def test_long_doc_produces_multiple_chunks(self) -> None:
        assert len(build_chunks([_doc(_LONG_BODY)])) > 1

    def test_every_chunk_starts_with_contextual_header(self) -> None:
        expected = (
            f"{_BASE_META['title']} · {_BASE_META['categories']} · {_BASE_META['source_url']}\n\n"
        )
        for chunk in build_chunks([_doc(_LONG_BODY)]):
            assert chunk.page_content.startswith(expected), (
                f"Header missing.\nExpected prefix: {expected!r}\n"
                f"Got start: {chunk.page_content[:120]!r}"
            )

    def test_every_chunk_body_is_wrapped_in_data_source(self) -> None:
        for chunk in build_chunks([_doc(_SHORT_BODY)]):
            assert "<DATA_SOURCE>" in chunk.page_content
            assert "</DATA_SOURCE>" in chunk.page_content

    def test_all_seven_metadata_keys_carried_onto_chunks(self) -> None:
        """All 7 ingest-contract metadata keys must survive split + header."""
        for chunk in build_chunks([_doc(_SHORT_BODY)]):
            assert set(_BASE_META.keys()) <= set(chunk.metadata.keys())

    def test_chunk_metadata_covers_pgvectorstore_columns(self) -> None:
        """Every declared typed PGVectorStore column must appear in chunk metadata.

        Fails if a column is added to ``_METADATA_COLUMNS`` without a matching
        key in the Document metadata — the format-drift guard.
        """
        for chunk in build_chunks([_doc(_SHORT_BODY)]):
            missing = set(METADATA_COLUMN_NAMES) - set(chunk.metadata.keys())
            assert not missing, f"Chunk metadata missing PG columns: {missing}"

    def test_start_index_present_on_all_chunks(self) -> None:
        """add_start_index=True means every chunk carries a start_index."""
        for chunk in build_chunks([_doc(_LONG_BODY)]):
            assert "start_index" in chunk.metadata

    def test_two_docs_keep_independent_metadata(self) -> None:
        """Chunks from different docs must not cross-contaminate metadata."""
        meta_a = {**_BASE_META, "source_url": "https://example.com/blog/a", "title": "A"}
        meta_b = {**_BASE_META, "source_url": "https://example.com/blog/b", "title": "B"}
        chunks = build_chunks([_doc(_SHORT_BODY, meta_a), _doc(_SHORT_BODY, meta_b)])
        urls = {c.metadata["source_url"] for c in chunks}
        assert urls == {"https://example.com/blog/a", "https://example.com/blog/b"}
