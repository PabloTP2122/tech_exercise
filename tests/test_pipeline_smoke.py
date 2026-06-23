"""Pipeline smoke test — single-document "vector-view" fixture.

Runs one real Bitovi article through the full pre-embed pipeline
(loader → clean → chunk → contextual header) and saves the exact strings that
would be passed to the embedder to ``tests/data/raw/test_1.json`` for human
inspection.  **No embedding or DB calls** — zero OpenAI cost.

Run:
    uv run pytest tests/test_pipeline_smoke.py -v           # passes online
    make eyeball                                             # same, via Makefile

The fixture is git-ignored (covered by the ``data/`` rule in ``.gitignore``).
Only this test file is committed.

For the full-corpus preview (all ~431 articles, ~4 min):
    make eyeball-all   # → uv run python -m ingest.preview

:func:`~ingest.preview.emulate_vector_view` is importable and works on any
``list[Document]`` — see ``ingest/preview.py`` for the scalable CLI.
"""

import json
from pathlib import Path

import pytest

from ingest.discovery import list_sitemap_urls
from ingest.loader import build_documents
from ingest.preview import emulate_vector_view

# Metadata keys every document must carry (ingest pipeline contract).
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


def test_single_document_pipeline() -> None:
    """Fetch one real Bitovi article and validate the full pre-embed pipeline.

    Skips gracefully when offline so ``make check`` stays green in CI / without
    network access.  When it passes it writes the vector-view fixture to
    ``tests/data/raw/test_1.json`` for human inspection.
    """
    # 1. Discover article URLs (cheap — single sitemap fetch, no body requests).
    urls = list_sitemap_urls()
    if not urls:
        pytest.skip("Sitemap fetch returned no URLs (offline or network error).")

    target_url = urls[0]

    # 2. Fetch exactly one article body through the SRP pipeline.
    docs = build_documents(filter_urls=[target_url])
    if not docs:
        pytest.skip(f"build_documents returned no docs for {target_url} (network error).")

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    doc = docs[0]

    # 3. Metadata contract: all required keys must be present and non-None.
    missing = _REQUIRED_METADATA_KEYS - set(doc.metadata.keys())
    assert not missing, f"Metadata missing keys: {missing}"

    # 4. Categories must be a non-empty delimited string.
    categories: str = doc.metadata.get("categories", "")
    assert categories.startswith(",") and categories.endswith(","), (
        f"categories not properly delimited: {categories!r}"
    )
    assert len(categories) > 1, f"categories is empty sentinel only: {categories!r}"

    # 5. Build the vector-view (clean → split → header).  No embeddings call.
    records = emulate_vector_view(docs)
    assert len(records) == 1
    record = records[0]

    # 6. Cleaning must not inflate content.
    assert record["clean_char_len"] <= record["raw_char_len"], (
        "clean_char_len > raw_char_len — cleaning is adding content"
    )

    # 7. At least one chunk must be produced.
    assert record["num_chunks"] >= 1, "No chunks produced from the article"

    # 8. content_origin provenance key must be present.
    assert record["content_origin"] == "blog_body", (
        f"Unexpected content_origin: {record['content_origin']!r}"
    )

    # 9. Every chunk's embedded_text must start with the contextual header
    #    (title · categories · url\n\n …).  This confirms the header was prepended.
    header_prefix = f"{record['title']} · {record['categories']} · {record['source_url']}"
    for i, chunk in enumerate(record["chunks"]):
        assert chunk["embedded_text"].startswith(header_prefix), (
            f"Chunk {i} does not start with contextual header.\n"
            f"Expected prefix: {header_prefix!r}\n"
            f"Got start:       {chunk['embedded_text'][:120]!r}"
        )
        # 10. Every chunk body must be wrapped in data-source delimiters.
        assert "<DATA_SOURCE>" in chunk["embedded_text"], (
            f"Chunk {i} missing <DATA_SOURCE> delimiter"
        )
        assert "</DATA_SOURCE>" in chunk["embedded_text"], (
            f"Chunk {i} missing </DATA_SOURCE> delimiter"
        )

    # 11. Write local-only fixture for human inspection.
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
