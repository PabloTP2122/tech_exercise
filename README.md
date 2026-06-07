# Company Blog RAG Q&A Agent

A query-type-aware Retrieval-Augmented Generation agent that answers questions about a company blog,
grounded exclusively in the blog's own articles, with source links on every response.

Built with FastAPI, LangGraph, pgvector (PostgreSQL), and Angular. Python 3.12, managed with `uv`.

---

## What makes this different from a basic RAG

A naive similarity-search RAG silently fails on three of the four query types this blog needs to
answer. This agent classifies each question and routes it to the strategy that can actually answer it:

| Question | Type | Strategy |
|----------|------|----------|
| "What tools does the blog recommend for E2E testing?" | semantic QA | similarity search → grounded answer |
| "Show me all articles about DevOps" | enumeration | metadata filter → complete list |
| "How many articles about AI?" | count | metadata count → accurate number |
| "What is the latest blog post about?" | recency | live RSS feed → genuinely newest article |

Cosine similarity has no concept of recency, completeness, or counting. Routing fixes that.

---

## Architecture

```
INGEST (offline)
  /sitemap.xml → discovery (sitemap + paginator cross-check)
              → fetcher (raw HTML, politeness 2 req/s)
              → extract (JSON-LD title/author/date + topic-link categories)
              → clean (body selector, nav/footer stripped)
              → chunk (800 / 100 overlap, contextual header)
              → PostgreSQL articles table  (one row/article — catalog)
              → PostgreSQL chunks table    (pgvector, cosine)

AGENT (per query, LangGraph StateGraph)
  classify_query  →  ┌ semantic_qa → similarity search (k=4) → generate
                     ├ enumeration → SQL articles WHERE categories ILIKE
                     ├ count       → SQL COUNT(*) WHERE categories ILIKE
                     └ recency     → SQL ORDER BY published_at + RSS overlay
                                  →  grounded answer with source citations

SERVE
  FastAPI  POST /ask · GET /health
  Angular  chat UI → /api/ask
```

---

## Quick Start

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/), Docker, Node.js + Angular CLI (for the UI), an OpenAI API key.

```bash
# 1. Install Python deps and git hooks
make install

# 2. Configure environment
cp .env.example .env
#   then edit .env: set OPENAI_API_KEY

# 3. Start the database
make db-up

# 4. Ingest the blog into pgvector (one-time, re-runnable)
make ingest

# 5. Start the API
make dev                      # http://localhost:8000

# 6. Start the UI (separate terminal)
cd frontend
ng serve --proxy-config proxy.conf.json   # http://localhost:4200
```

Ask a question at `http://localhost:4200`, or hit the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What tools does the blog recommend for E2E testing?"}'
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `BLOG_BASE_URL` | ✅ | `https://www.company.com` | Blog root (used for sitemap + RSS) |
| `DATABASE_URL` | | `postgresql+psycopg://rag:rag@localhost:5432/blog_rag` | PostgreSQL DSN (matches docker-compose defaults) |
| `COLLECTION_NAME` | | `chunks` | pgvector table name for article chunks |
| `LLM_MODEL` | | `gpt-4o-mini` | Generation model |
| `CLASSIFIER_MODEL` | | `gpt-4o-mini` | Query classification model |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | Embedding model |

---

## Commands

```bash
make help          # list all commands
make install       # install deps + pre-commit hooks
make db-up         # start PostgreSQL + pgvector (Docker)
make db-down       # stop containers (preserves data volume)
make ingest        # fetch, embed, and load the blog into pgvector
make load          # force rebuild: drop table + re-embed
make dev           # run the FastAPI server
make test          # run the test suite
make check         # lint + format-check + typecheck + test (full gate)
make clean         # remove Python caches + stop containers and wipe DB volume
```

---

## How Accuracy & Relevance Are Achieved

1. **Strict grounding** — the model answers only from retrieved context; outside knowledge is forbidden.
2. **No-hallucination fallback** — with no relevant context, it replies "I couldn't find information about that in the blog" rather than inventing one.
3. **Source citations** — every answer links the articles it used, so each claim is verifiable.
4. **Query-type routing** — counts return real counts, lists return complete lists, "latest" returns the newest post.
5. **Vocabulary-normalized classification** — the classifier maps user wording (e.g. "Agile") to the actual category slug (`project-management`) so metadata filters don't silently miss.
6. **Context-preserving chunking** — 100-token overlap prevents losing answers that straddle a chunk boundary.
7. **Live recency** — recency queries read the blog's RSS feed at query time, reflecting current state, not the last ingest.

---

## Design Decisions

**Loaders over a custom scraper.** `SitemapLoader` + `WebBaseLoader` provide URL discovery and HTML
fetching in a few lines. A hand-written scraper would add ~80 lines with no quality advantage. More
code is not a proxy for effort.

**Topic links for category metadata.** Categories come from `a[href*="/blog/topic/"]` links in
each article body — not JSON-LD (Bitovi's `BlogPosting` block carries no `keywords` field).
All discovered slugs are stored so the catalog route works for any topic, not just the main nav set.

**Query routing instead of plain RAG.** Three of the four target query types fail under pure
similarity search: enumeration returns partial lists, counting returns invented numbers, and recency
returns the chunk most similar to the word "latest" rather than the newest article. A classification
node routes each query to a strategy that answers it correctly.

**Hybrid recency.** SQL `ORDER BY published_at DESC` is the deterministic base (always works, no
network dependency, no item cap). A live RSS fetch is an optional overlay — if it succeeds and finds
a newer post than the SQL result, that item is preferred. Falls back to SQL silently when offline.

**Delimited string for categories.** Categories are stored as `,react,angular,` so a SQL
`ILIKE '%,react,%'` filter matches precisely without colliding with substrings like `react-native`.
pgvector's native filter operators (`$ilike`) handle this natively on typed metadata columns.

**`gpt-4o-mini`.** Fast and inexpensive for both classification and generation, which keeps end-to-end
latency low enough for an interactive demo.

**Angular over a server-rendered template.** The blog covers Angular as a core topic; the UI uses
Angular 17 standalone components to stay coherent with that stack.

---

## Known Limitations

- **Category slug list is auto-discovered** from `SELECT DISTINCT categories FROM articles` at
  classifier startup — no hardcoded list.
- **RSS feeds cap at 10 items per feed** — used as a freshness overlay only; the SQL base covers
  the full article history.
- **No conversational memory** — each question is answered independently, by design.
- **No automated RAG evaluation** — RAGAS metrics (faithfulness, context recall, answer relevancy)
  are noted as the next step but not implemented.

---

## Project Structure

```
ingest/    discovery.py · fetcher.py · extract.py · clean.py   # SRP scraper pipeline
           loader.py · load_vectorstore.py · catalog_db.py      # orchestration + DB
agent/     graph.py · classifier.py · rss.py · prompts.py       # LangGraph routing agent
api/       main.py · config.py · models.py                      # FastAPI surface
frontend/  Angular 17 chat UI
tests/     test_pipeline_smoke.py · test_extract.py · test_discovery.py
spec/      prd.md · progress.txt                                # build plan + progress tracker
```

---

## Tech Stack

Python 3.12 · uv · ruff · mypy · pre-commit · FastAPI · LangGraph · LangChain ·
PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`, Docker) · SQLAlchemy · psycopg3 ·
OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) · Angular 17

---

## Running Quality Checks

```bash
make check
```

Runs ruff (lint), ruff (format check), mypy (strict typing), and pytest. The same checks run on every
commit via pre-commit hooks.
