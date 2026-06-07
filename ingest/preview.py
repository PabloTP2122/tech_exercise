"""Pre-embed vector-view: inspect exactly what the embedder will receive.

No embedding calls, no DB connections — pure pipeline inspection tool.  Used
by ``make eyeball-all`` to validate the full corpus before paying for OpenAI
embeddings.  The 1-doc smoke target (``make eyeball``) uses
``tests/test_pipeline_smoke.py`` directly.

Usage
-----
Full corpus (~431 articles, network, ~4 min, no embed):
    uv run python -m ingest.preview
    uv run python -m ingest.preview --limit 5
    make eyeball-all
"""

import argparse
import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from ingest.load_vectorstore import build_chunks
from ingest.loader import build_documents, list_article_urls

logger = logging.getLogger(__name__)

# Output path for the full-corpus report (git-ignored via tests/data/ rule).
_FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "data" / "raw"
_FIXTURE_PATH = _FIXTURE_DIR / "vector_view.json"


def emulate_vector_view(docs: list[Document]) -> list[dict]:  # type: ignore[type-arg]
    """Return the exact embed-input strings + stored metadata for a document list.

    Applies the same split → header → data-source-wrap transform as
    :func:`~ingest.load_vectorstore.build_chunks`, so the output faithfully
    represents what PGVectorStore will receive at embed time.

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
                "categories": str,          # ",slug,slug,"
                "content_origin": str,      # "blog_body"
                "raw_char_len": int,        # len(doc.page_content)
                "clean_char_len": int,      # same (cleaning applied at fetch time)
                "num_chunks": int,
                "chunks": [
                    {
                        "chunk_index": int,
                        "start_index": int | None,   # char offset in source body
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
                "content_origin": doc.metadata.get("content_origin", ""),
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


def find_problems(records: list[dict]) -> list[str]:  # type: ignore[type-arg]
    """Scan vector-view records and return a list of corpus quality problems.

    Checks each article record for:

    - Empty ``categories`` (only the ``","`` sentinel) — breaks catalog routes.
    - Missing or empty ``content_origin`` — injection-scaffolding regression.
    - Any chunk whose ``embedded_text`` is missing ``<DATA_SOURCE>`` or
      ``</DATA_SOURCE>`` delimiters — delimiter regression.

    Args:
        records: Records as returned by :func:`emulate_vector_view`.

    Returns:
        List of human-readable problem strings.  Empty list = clean corpus.
    """
    problems: list[str] = []
    for rec in records:
        url = rec.get("source_url", "<unknown>")
        if rec.get("categories", ",") == ",":
            problems.append(f"EMPTY_CATEGORIES  {url}")
        if not rec.get("content_origin"):
            problems.append(f"MISSING_CONTENT_ORIGIN  {url}")
        for chunk in rec.get("chunks", []):
            text: str = chunk.get("embedded_text", "")
            idx = chunk.get("chunk_index", "?")
            if "<DATA_SOURCE>" not in text:
                problems.append(f"MISSING_DATA_SOURCE  chunk={idx}  {url}")
            if "</DATA_SOURCE>" not in text:
                problems.append(f"MISSING_DATA_SOURCE_CLOSE  chunk={idx}  {url}")
    return problems


def main(limit: int | None = None) -> None:
    """Run the corpus vector-view, write a JSON report, and print a summary.

    Args:
        limit: If set, process only the first ``limit`` URLs from the sitemap.
               ``None`` processes all discovered articles (~431).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    urls = list_article_urls()
    subset = urls[:limit] if limit is not None else urls
    logger.info("Processing %d / %d articles …", len(subset), len(urls))

    docs = build_documents(filter_urls=subset)
    if not docs:
        logger.error("No documents returned — check network and BLOG_BASE_URL.")
        return

    records = emulate_vector_view(docs)
    total_chunks = sum(r["num_chunks"] for r in records)
    avg_chunks = total_chunks / len(records) if records else 0.0

    print(f"\n{'=' * 60}")
    print(f"  Articles processed : {len(records)}")
    print(f"  Total chunks       : {total_chunks}")
    print(f"  Avg chunks/article : {avg_chunks:.1f}")
    print(f"{'=' * 60}\n")

    problems = find_problems(records)
    if problems:
        print(f"⚠  {len(problems)} problem(s) found:")
        for p in problems:
            print(f"   {p}")
    else:
        print("✓  No problems found — corpus looks clean.")

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote vector-view report to %s", _FIXTURE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-embed vector-view: inspect corpus without embedding."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N articles (default: all).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
