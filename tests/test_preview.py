"""Offline unit tests for ingest/preview.py — emulate_vector_view + find_problems."""

from langchain_core.documents import Document

from ingest.preview import emulate_vector_view, find_problems

_META: dict[str, str] = {
    "source_url": "https://www.bitovi.com/blog/sample",
    "title": "Sample Article",
    "description": "Sample description.",
    "author": "Jane Doe",
    "published_at": "2025-03-01 10:00:00",
    "categories": ",react,angular,",
    "content_origin": "blog_body",
}

_DOC = Document(page_content="Short body text for previewing.", metadata=_META)


class TestEmulateVectorView:
    """emulate_vector_view produces faithful records of the embed-input format."""

    def test_one_record_per_document(self) -> None:
        records = emulate_vector_view([_DOC])
        assert len(records) == 1

    def test_record_source_url_matches_doc(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        assert record["source_url"] == _META["source_url"]

    def test_record_includes_content_origin(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        assert record["content_origin"] == "blog_body"

    def test_record_categories_preserved(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        assert record["categories"] == _META["categories"]

    def test_at_least_one_chunk_produced(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        assert record["num_chunks"] >= 1

    def test_chunk_embedded_text_starts_with_header(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        expected = f"{_META['title']} · {_META['categories']} · {_META['source_url']}"
        for chunk in record["chunks"]:
            assert chunk["embedded_text"].startswith(expected)

    def test_chunk_embedded_text_contains_data_source_delimiters(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        for chunk in record["chunks"]:
            assert "<DATA_SOURCE>" in chunk["embedded_text"]
            assert "</DATA_SOURCE>" in chunk["embedded_text"]

    def test_chunk_embedded_char_len_matches_text(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        for chunk in record["chunks"]:
            assert chunk["embedded_char_len"] == len(chunk["embedded_text"])

    def test_raw_char_len_matches_doc_page_content(self) -> None:
        record = emulate_vector_view([_DOC])[0]
        assert record["raw_char_len"] == len(_DOC.page_content)

    def test_clean_char_len_equals_raw_char_len(self) -> None:
        """Cleaning happens at fetch time; both lens reflect the clean body."""
        record = emulate_vector_view([_DOC])[0]
        assert record["clean_char_len"] == record["raw_char_len"]

    def test_two_docs_produce_two_records(self) -> None:
        meta_b = {**_META, "source_url": "https://www.bitovi.com/blog/other"}
        doc_b = Document(page_content="Another body.", metadata=meta_b)
        records = emulate_vector_view([_DOC, doc_b])
        assert len(records) == 2
        urls = {r["source_url"] for r in records}
        assert urls == {_META["source_url"], meta_b["source_url"]}


class TestFindProblems:
    """find_problems flags each corpus quality defect independently."""

    def _clean_record(self) -> dict:  # type: ignore[type-arg]
        """Return a minimal clean record (no problems)."""
        return {
            "source_url": "https://www.bitovi.com/blog/clean",
            "categories": ",react,",
            "content_origin": "blog_body",
            "chunks": [
                {
                    "chunk_index": 0,
                    "embedded_text": (
                        "Title · ,react, · https://www.bitovi.com/blog/clean\n\n"
                        "<DATA_SOURCE>\nBody text.\n</DATA_SOURCE>"
                    ),
                }
            ],
        }

    def test_clean_record_produces_no_problems(self) -> None:
        assert find_problems([self._clean_record()]) == []

    def test_empty_categories_flagged(self) -> None:
        rec = self._clean_record()
        rec["categories"] = ","
        problems = find_problems([rec])
        assert any("EMPTY_CATEGORIES" in p for p in problems)

    def test_missing_content_origin_flagged(self) -> None:
        rec = self._clean_record()
        rec["content_origin"] = ""
        problems = find_problems([rec])
        assert any("MISSING_CONTENT_ORIGIN" in p for p in problems)

    def test_absent_content_origin_key_flagged(self) -> None:
        rec = self._clean_record()
        del rec["content_origin"]
        problems = find_problems([rec])
        assert any("MISSING_CONTENT_ORIGIN" in p for p in problems)

    def test_missing_data_source_open_flagged(self) -> None:
        rec = self._clean_record()
        rec["chunks"][0]["embedded_text"] = "No delimiters here at all."
        problems = find_problems([rec])
        assert any("MISSING_DATA_SOURCE" in p for p in problems)

    def test_missing_data_source_close_flagged(self) -> None:
        rec = self._clean_record()
        rec["chunks"][0]["embedded_text"] = "<DATA_SOURCE>\nBody without close."
        problems = find_problems([rec])
        assert any("MISSING_DATA_SOURCE_CLOSE" in p for p in problems)

    def test_multiple_defects_all_reported(self) -> None:
        rec = self._clean_record()
        rec["categories"] = ","
        rec["content_origin"] = ""
        problems = find_problems([rec])
        assert any("EMPTY_CATEGORIES" in p for p in problems)
        assert any("MISSING_CONTENT_ORIGIN" in p for p in problems)

    def test_empty_records_list_produces_no_problems(self) -> None:
        assert find_problems([]) == []
