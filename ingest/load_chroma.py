"""ChromaDB loader: chunk → embed → persist.

Usage
-----
Checkpoint-aware (cost guard):
    uv run python -m ingest.load_chroma        # skips if collection already populated
    make ingest

Force rebuild (delete + re-embed):
    uv run python -m ingest.load_chroma --force
    make load
"""

import argparse
import logging
import time

import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.config import Settings, get_settings
from ingest.clean import clean_document
from ingest.loader import build_documents

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000  # stay well below the chromadb 5461-record ceiling


def build_chunks(docs: list[Document]) -> list[Document]:
    """Clean → split → prepend contextual header.

    Produces the exact strings that will be passed to the embedder — i.e. the
    "vector-view" of the corpus.  Public so the smoke-test can reuse this
    transform to inspect embed inputs without actually calling OpenAI.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
    )

    cleaned = [clean_document(d) for d in docs]
    chunks = splitter.split_documents(cleaned)

    # Prepend contextual header so each chunk is self-describing in embedding space.
    # The header carries title + topic slugs + URL so short/ambiguous chunks are
    # retrievable even when their text has no explicit subject.
    #
    # Example:
    #   "Angular 17 Standalone Components · ,angular,frontend, · https://www.bitovi.com/blog/…
    #
    #   With standalone components you can now bootstrap…"
    enriched: list[Document] = []
    for chunk in chunks:
        meta = chunk.metadata
        title: str = meta.get("title") or ""
        categories: str = meta.get("categories") or ""
        url: str = meta.get("source_url") or meta.get("source") or ""
        header = f"{title} · {categories} · {url}\n\n"
        enriched.append(Document(page_content=header + chunk.page_content, metadata=meta))

    logger.info(
        "Split %d articles → %d chunks (blog_base_url=%s)",
        len(docs),
        len(enriched),
        settings.blog_base_url,
    )
    return enriched


def _get_client_and_collection_count(settings: Settings) -> tuple[ClientAPI, int]:
    """Return the ChromaDB client and the existing collection count (0 if absent)."""
    client: ClientAPI = chromadb.PersistentClient(path=settings.chroma_db_path)
    try:
        col = client.get_collection(settings.collection_name)
        return client, col.count()
    except Exception:
        return client, 0


def main(*, force: bool = False) -> None:
    """Chunk, embed, and persist articles to ChromaDB.

    Args:
        force: If True, delete the existing collection and rebuild from scratch.
               If False (default), skip embedding when the collection is already
               populated (cost guard).
    """
    settings = get_settings()
    t0 = time.monotonic()

    client, existing_count = _get_client_and_collection_count(settings)

    if existing_count > 0 and not force:
        logger.info(
            "Collection '%s' already contains %d chunks — skipping (use --force to rebuild).",
            settings.collection_name,
            existing_count,
        )
        return

    if existing_count > 0 and force:
        logger.info(
            "Force-rebuild: deleting collection '%s' (%d existing chunks).",
            settings.collection_name,
            existing_count,
        )
        client.delete_collection(settings.collection_name)

    # --- Load → clean → chunk → embed → persist ----------------------------
    docs = build_documents()
    logger.info("Loaded %d articles from sitemap.", len(docs))

    chunks = build_chunks(docs)

    # OpenAIEmbeddings picks up OPENAI_API_KEY from the environment automatically;
    # passing it explicitly is not supported in newer langchain-openai versions.
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    # Build the vector store; collection_metadata sets cosine distance space so
    # ranking is correct for L2-normalized text-embedding-3-small vectors.
    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_path,
        collection_metadata={"hnsw:space": "cosine"},
        client=client,
    )

    # Add in batches to respect the chromadb 5461-record limit and smooth
    # OpenAI embedding API rate limits.
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        # filter_complex_metadata drops any metadata values that ChromaDB cannot
        # store (lists, nested dicts) — our schema is flat strings so this is
        # a defensive no-op that prevents hard-to-debug runtime errors.
        safe_batch = filter_complex_metadata(batch)
        vector_store.add_documents(safe_batch)
        logger.info(
            "  Batch %d/%d: added %d chunks.",
            i // _BATCH_SIZE + 1,
            -(-len(chunks) // _BATCH_SIZE),  # ceiling division
            len(safe_batch),
        )

    elapsed = time.monotonic() - t0
    logger.info(
        "Done. %d articles → %d chunks persisted to '%s' in %.1fs.",
        len(docs),
        len(chunks),
        settings.chroma_db_path,
        elapsed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Chunk, embed, and persist blog to ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing collection and rebuild from scratch.",
    )
    args = parser.parse_args()
    main(force=args.force)
