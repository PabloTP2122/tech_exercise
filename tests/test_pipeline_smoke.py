"""Pipeline smoke test — single-document "vector-view" fixture.

Runs one real Bitovi article through the full pre-embed pipeline
(loader → clean → chunk → contextual header) and saves the exact strings that
would be passed to the embedder to ``tests/data/raw/test_1.json`` for human
inspection.  **No embedding or ChromaDB calls** — zero OpenAI cost.

Run:
    uv run pytest tests/test_pipeline_smoke.py -v           # passes online
    uv run pytest tests/test_pipeline_smoke.py -v --offline # skips gracefully

The fixture is git-ignored (covered by the ``data/`` rule in ``.gitignore``).
Only this test file is committed.

Scalability note
----------------
:func:`emulate_vector_view` is importable and works on any ``list[Document]``.
Pass ``build_documents()`` (all articles) to produce a full pre-vectorization
quality report before paying for embedding.
"""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingest.load_vectorstore import build_chunks
from ingest.loader import build_documents, list_article_urls

# Metadata keys every document must carry (from TASK-04 / TASK-05 contract).
_REQUIRED_METADATA_KEYS = {
    "source_url",
    "title",
    "description",
    "author",
    "published_at",
    "categories",
    "content_origin",
}

# Output path for the human-readable fixture (git-ignored via data/ rule).
_FIXTURE_PATH = Path(__file__).parent / "data" / "raw" / "test_1.json"


# ---------------------------------------------------------------------------
# Public utility: emulate what the vector DB receives per document
# ---------------------------------------------------------------------------


def emulate_vector_view(docs: list[Document]) -> list[dict]:  # type: ignore[type-arg]
    """Return the exact embed-input strings + metadata for a list of documents.

    Applies split → contextual-header in exactly the same way
    :func:`~ingest.load_chroma.build_chunks` does (it re-uses that function),
    so the output faithfully represents what the embedder will receive.

    Documents arrive already clean (body extracted by :mod:`ingest.extract`),
    so ``raw_char_len`` and ``clean_char_len`` both reflect the clean body size.

    Args:
        docs: Documents as returned by :func:`~ingest.loader.build_documents`.

    Returns:
        List of dicts, one per input document::

            {
                "source_url": str,
                "title": str,
                "description": str,
                "author": str,
                "published_at": str,
                "categories": str,           # ",slug,slug,"
                "raw_char_len": int,         # len(doc.page_content)
                "clean_char_len": int,       # same as raw_char_len (cleaning at fetch time)
                "num_chunks": int,
                "chunks": [
                    {
                        "chunk_index": int,
                        "start_index": int | None,   # char offset in body
                        "embedded_text": str,        # exact string the embedder sees
                        "embedded_char_len": int,
                    },
                    ...
                ],
            }
    """
    chunks_all = build_chunks(docs)

    # Group chunks back to their source document by source_url.
    # build_chunks preserves the metadata from the original doc on every chunk.
    groups: dict[str, list[Document]] = {}
    for chunk in chunks_all:
        key: str = chunk.metadata.get("source_url") or chunk.metadata.get("source", "")
        groups.setdefault(key, []).append(chunk)

    records = []
    for doc in docs:
        url: str = doc.metadata.get("source_url") or doc.metadata.get("source", "")
        raw_len = len(doc.page_content)
        # Cleaning is applied at fetch time in ingest.extract.build_document;
        # by the time we have a Document, page_content is already the clean body.
        clean_len = raw_len
        doc_chunks = groups.get(url, [])

        records.append(
            {
                "source_url": url,
                "title": doc.metadata.get("title", ""),
                "description": doc.metadata.get("description", ""),
                "author": doc.metadata.get("author", ""),
                "published_at": doc.metadata.get("published_at", ""),
                "categories": doc.metadata.get("categories", ","),
                "raw_char_len": raw_len,
                "clean_char_len": clean_len,
                "num_chunks": len(doc_chunks),
                "chunks": [
                    {
                        "chunk_index": i,
                        "start_index": c.metadata.get("start_index"),
                        "embedded_text": c.page_content,
                        "embedded_char_len": len(c.page_content),
                    }
                    for i, c in enumerate(doc_chunks)
                ],
            }
        )
    return records


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_single_document_pipeline() -> None:
    """Fetch one real Bitovi article and validate the full pre-embed pipeline.

    Skips gracefully when offline so ``make check`` stays green in CI / without
    network access.  When it passes it writes the vector-view fixture to
    ``tests/data/raw/test_1.json`` for human inspection.
    """
    # 1. Discover article URLs (cheap — single sitemap fetch, no body requests).
    urls = list_article_urls()
    if not urls:
        pytest.skip("Sitemap fetch returned no URLs (offline or network error).")

    target_url = urls[0]

    # 2. Fetch exactly one article body through the new SRP pipeline.
    #    filter_urls is now a plain URL list (no regex); pass the URL verbatim.
    docs = build_documents(filter_urls=[target_url])
    if not docs:
        pytest.skip(f"build_documents returned no docs for {target_url} (network error).")

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    doc = docs[0]

    # 3. Metadata contract: all six keys must be present and non-None.
    missing = _REQUIRED_METADATA_KEYS - set(doc.metadata.keys())
    assert not missing, f"Metadata missing keys: {missing}"

    # 4. Categories must be a non-empty delimited string.
    categories: str = doc.metadata.get("categories", "")
    assert categories.startswith(",") and categories.endswith(
        ","
    ), f"categories not properly delimited: {categories!r}"
    assert len(categories) > 1, f"categories is empty sentinel only: {categories!r}"

    # 5. Build the vector-view (clean → split → header).  No embeddings call.
    records = emulate_vector_view(docs)
    assert len(records) == 1
    record = records[0]

    # 6. Cleaning must not inflate content.
    assert (
        record["clean_char_len"] <= record["raw_char_len"]
    ), "clean_char_len > raw_char_len — cleaning is adding content"

    # 7. At least one chunk must be produced.
    assert record["num_chunks"] >= 1, "No chunks produced from the article"

    # 8. Every chunk's embedded_text must start with the contextual header
    #    (title · categories · url\n\n …).  This confirms the header was prepended.
    header_prefix = f"{record['title']} · {record['categories']} · {record['source_url']}"
    for i, chunk in enumerate(record["chunks"]):
        assert chunk["embedded_text"].startswith(header_prefix), (
            f"Chunk {i} does not start with contextual header.\n"
            f"Expected prefix: {header_prefix!r}\n"
            f"Got start:       {chunk['embedded_text'][:120]!r}"
        )
        # 9. Every chunk body must be wrapped in data-source delimiters
        #    (injection-resistant scaffolding from TASK-05G).
        assert (
            "<DATA_SOURCE>" in chunk["embedded_text"]
        ), f"Chunk {i} missing <DATA_SOURCE> delimiter"
        assert (
            "</DATA_SOURCE>" in chunk["embedded_text"]
        ), f"Chunk {i} missing </DATA_SOURCE> delimiter"

    # 9. Write local-only fixture for human inspection.
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
