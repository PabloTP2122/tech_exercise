"""pgvector loader: chunk → embed → persist via PGVectorStore.

Usage
-----
Checkpoint-aware (cost guard):
    uv run python -m ingest.load_vectorstore        # skips if already populated
    make ingest

Force rebuild (drop + re-embed):
    uv run python -m ingest.load_vectorstore --force
    make load
"""

import argparse
import logging
import time

import sqlalchemy as sa
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.config import Settings, get_settings
from ingest.catalog_db import ensure_articles_table, upsert_articles
from ingest.loader import build_documents

_METADATA_COLUMNS: list[Column] = [
    Column("source_url", "TEXT"),
    Column("categories", "TEXT"),
    Column("title", "TEXT"),
    Column("author", "TEXT"),
    Column("published_at", "TEXT"),
    Column("description", "TEXT"),
    Column("content_origin", "TEXT"),
]
# Names only — used by PGVectorStore.create_sync (which takes str, not Column).
_METADATA_COLUMN_NAMES: list[str] = [c.name for c in _METADATA_COLUMNS]

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000
_VECTOR_SIZE = 1536  # text-embedding-3-small output dimension


def wrap_as_data(text: str) -> str:
    """Wrap ``text`` in data-source delimiters for injection-resistant embedding.

    Places the chunk body inside ``<DATA_SOURCE>…</DATA_SOURCE>`` boundaries
    so downstream generation prompts can treat it unambiguously as DATA, not
    instructions.  Text is preserved byte-for-byte inside the delimiters.

    Args:
        text: Chunk body text to wrap.

    Returns:
        ``"<DATA_SOURCE>\\n{text}\\n</DATA_SOURCE>"``.
    """
    return f"<DATA_SOURCE>\n{text}\n</DATA_SOURCE>"


def build_chunks(docs: list[Document]) -> list[Document]:
    """Split documents and prepend a contextual header to each chunk.

    Documents arrive already clean (body extracted by :mod:`ingest.extract`),
    so this function only splits and enriches.  Produces the exact strings
    passed to the embedder — the "vector-view" of the corpus.  Public so
    the smoke-test can inspect embed inputs without calling OpenAI.

    Each chunk is formatted as::

        {title} · {categories} · {url}

        <DATA_SOURCE>
        {body}
        </DATA_SOURCE>

    The header carries provenance (title, topics, URL) so short or ambiguous
    chunks are self-describing in embedding space.  The ``<DATA_SOURCE>``
    wrapper signals the prompt layer to treat the body as DATA, not instructions.

    Args:
        docs: Clean documents from :func:`~ingest.loader.build_documents`.

    Returns:
        List of chunk :class:`Document` objects with header-prepended,
        delimiter-wrapped ``page_content`` ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    # Prepend contextual header so each chunk is self-describing in embedding
    # space.  Short or ambiguous chunks ("It supports parallel runs…") become
    # retrievable because the header carries title + topic slugs + URL.
    enriched: list[Document] = []
    for chunk in chunks:
        meta = chunk.metadata
        title: str = meta.get("title") or ""
        categories: str = meta.get("categories") or ""
        url: str = meta.get("source_url") or meta.get("source") or ""
        header = f"{title} · {categories} · {url}\n\n"
        body = wrap_as_data(chunk.page_content)
        enriched.append(Document(page_content=header + body, metadata=meta))

    logger.info("Split %d articles → %d chunks.", len(docs), len(enriched))
    return enriched


def _make_sync_engine(settings: Settings) -> sa.Engine:
    """Return a SQLAlchemy sync engine for DDL and catalog operations.

    Args:
        settings: Application settings carrying the database DSN.

    Returns:
        A connected :class:`sqlalchemy.Engine`.
    """
    return sa.create_engine(settings.database_url)


def _get_existing_chunk_count(sync_engine: sa.Engine, table_name: str) -> int:
    """Return the number of rows in the chunks table, or 0 if absent.

    Args:
        sync_engine: SQLAlchemy sync engine.
        table_name: Name of the PGVectorStore-managed chunks table.

    Returns:
        Row count, or ``0`` when the table does not exist or query fails.
    """
    try:
        with sync_engine.connect() as conn:
            # table_name is an internal constant from Settings — not user input.
            row = conn.execute(
                sa.text(f"SELECT count(*) FROM {table_name}")  # noqa: S608
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def _build_pg_engine(settings: Settings) -> PGEngine:
    """Construct a :class:`PGEngine` from the application DSN.

    Args:
        settings: Application settings.

    Returns:
        A :class:`PGEngine` instance ready for ``init_vectorstore_table``
        and :class:`PGVectorStore` construction.
    """
    return PGEngine.from_connection_string(url=settings.database_url)


def _build_vector_store(pg_engine: PGEngine, settings: Settings) -> PGVectorStore:
    """Construct a :class:`PGVectorStore` bound to the chunks table.

    Default distance strategy is COSINE_DISTANCE (correct for
    ``text-embedding-3-small`` normalized vectors).

    Args:
        pg_engine: Initialized :class:`PGEngine`.
        settings: Application settings (embedding model, table name).

    Returns:
        A :class:`PGVectorStore` ready for :meth:`add_documents` /
        :meth:`similarity_search_with_score`.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,  # SecretStr; accepted directly by LangChain
    )
    return PGVectorStore.create_sync(
        engine=pg_engine,
        embedding_service=embeddings,
        table_name=settings.collection_name,
        metadata_columns=_METADATA_COLUMN_NAMES,
    )


def main(*, force: bool = False) -> None:
    """Chunk, embed, and persist articles to pgvector.

    Checkpoint-aware: if the chunks table already has rows and ``force`` is
    ``False``, the function returns immediately (cost guard).  With
    ``force=True`` the existing table is dropped and rebuilt from scratch.

    Args:
        force: If ``True``, drop the chunks table and rebuild.
               If ``False`` (default), skip when already populated.
    """
    settings = get_settings()
    t0 = time.monotonic()

    sync_engine = _make_sync_engine(settings)
    pg_engine = _build_pg_engine(settings)

    # Ensure the articles catalog table exists (idempotent).
    ensure_articles_table(sync_engine)

    existing_count = _get_existing_chunk_count(sync_engine, settings.collection_name)

    if existing_count > 0 and not force:
        logger.info(
            "Table '%s' already contains %d chunks — skipping (use --force to rebuild).",
            settings.collection_name,
            existing_count,
        )
        return

    if existing_count > 0 and force:
        logger.info(
            "Force-rebuild: dropping table '%s' (%d existing chunks).",
            settings.collection_name,
            existing_count,
        )
        pg_engine.drop_table(settings.collection_name)

    # (Re-)create the PGVectorStore-managed table with typed metadata columns.
    pg_engine.init_vectorstore_table(
        table_name=settings.collection_name,
        vector_size=_VECTOR_SIZE,
        metadata_columns=_METADATA_COLUMNS,  # type: ignore[arg-type]
    )

    # --- Load → catalog upsert → chunk → embed → persist -------------------
    docs = build_documents()
    logger.info("Loaded %d articles from sitemap.", len(docs))

    # Upsert one catalog row per article (for SQL count/enum/recency queries).
    upsert_articles(sync_engine, docs)

    chunks = build_chunks(docs)
    vector_store = _build_vector_store(pg_engine, settings)

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        vector_store.add_documents(batch)
        logger.info(
            "  Batch %d/%d: added %d chunks.",
            i // _BATCH_SIZE + 1,
            -(-len(chunks) // _BATCH_SIZE),  # ceiling division
            len(batch),
        )

    elapsed = time.monotonic() - t0
    logger.info(
        "Done. %d articles → %d chunks persisted in %.1fs.",
        len(docs),
        len(chunks),
        elapsed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Chunk, embed, and persist blog to pgvector.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop the existing table and rebuild from scratch.",
    )
    args = parser.parse_args()
    main(force=args.force)
