# Bitovi Blog RAG Q&A Agent

A query-type-aware Retrieval-Augmented Generation agent that answers questions about the Bitovi blog,
grounded exclusively in the blog's own articles, with source links on every response.

Built with FastAPI, LangGraph, pgvector (PostgreSQL), Next.js 16 + Hono + React 19. Python 3.12, managed with `uv`.

---

## How this meets the brief

| Requirement | Where |
|---|---|
| Ingest all Bitovi blog articles | `ingest/` pipeline — `make ingest` fetches, embeds, and loads ~462 articles |
| RAG-grounded Q&A agent | `agent/` LangGraph StateGraph — 4-type routing; only `semantic_qa` calls the LLM |
| UI: input field + answer + reference links | `frontend/` Next.js app at `http://localhost:3000` |
| Demo video (2–5 min) | See **Demo Video** section below |

---

## What makes this different from a basic RAG

A naive similarity-search RAG silently fails on three of the four query types this brief requires.
This agent classifies each question and routes it to the strategy that can actually answer it:

| Question | Type | Strategy |
|---|---|---|
| "What is Bitovi's latest blog post about?" | `recency` | live RSS + SQL → genuinely newest article |
| "Can you show me all Bitovi articles about DevOps?" | `enumeration` | metadata filter → complete list |
| "How many articles does Bitovi have about AI?" | `count` | SQL `COUNT(*)` → accurate number |
| "What kind of tools does Bitovi recommend for E2E testing?" | `semantic_qa` | similarity search → grounded answer |

Cosine similarity has no concept of recency, completeness, or counting. Routing fixes that.

---

## Architecture

```
INGEST (offline)
  /sitemap.xml → discovery (sitemap + paginator cross-check)
              → fetcher (raw HTML, politeness 2 req/s)
              → extract (JSON-LD title/author/date + topic-link categories)
              → clean (markdownify body, nav/footer stripped)
              → chunk (800 / 100 overlap, contextual header)
              → PostgreSQL articles table  (one row/article — catalog)
              → PostgreSQL chunks table    (pgvector, cosine)

AGENT (per query, LangGraph StateGraph)
  classify_query  →  ┌ semantic_qa → similarity search (k=4) → generate (LLM)
                     ├ enumeration → SQL articles WHERE categories ILIKE
                     ├ count       → SQL COUNT(*) WHERE categories ILIKE
                     └ recency     → SQL ORDER BY published_at + RSS overlay
                                  →  grounded answer with source citations

SERVE
  FastAPI  POST /ask · GET /health           (port 8000)
  Next.js  App Router + Hono /api/* proxy    (port 3000)
           → proxies questions to FastAPI /ask
```

---

## Try these

The brief's four example queries — paste any into the UI or `curl` them directly:

| Query | Routes to |
|---|---|
| `"What is Bitovi's latest blog post about?"` | `recency` |
| `"Can you show me all Bitovi articles about DevOps?"` | `enumeration` |
| `"How many articles does Bitovi have about AI?"` | `count` |
| `"What kind of tools does Bitovi recommend for E2E testing?"` | `semantic_qa` |

---

## Quick Start

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/), Docker, Node.js + `pnpm`, an OpenAI API key.

```bash
# 1. Install Python deps and git hooks
make install

# 2. Configure environment
cp .env.example .env
#   then edit .env: set OPENAI_API_KEY

# 3. Start the database
make db-up

# 4. Ingest the Bitovi blog into pgvector (one-time, re-runnable, ~462 articles)
make ingest

# 5. Start the API
make dev                      # http://localhost:8000

# 6. Start the UI (separate terminal)
cd frontend
pnpm install
pnpm dev                      # http://localhost:3000
```

Ask a question at `http://localhost:3000`, or hit the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What kind of tools does Bitovi recommend for E2E testing?"}'
```

---

## Environment Variables

### Backend (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `BLOG_BASE_URL` | ✅ | `https://www.bitovi.com` | Blog root (sitemap + RSS) |
| `DATABASE_URL` | | `postgresql+psycopg://rag:rag@localhost:5432/blog_rag` | PostgreSQL DSN (matches docker-compose defaults) |
| `COLLECTION_NAME` | | `chunks` | pgvector table name for article chunks |
| `LLM_MODEL` | | `gpt-4o-mini` | Generation model |
| `CLASSIFIER_MODEL` | | `gpt-4o-mini` | Query classification model |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | Embedding model |
| `LANGCHAIN_API_KEY` | | — | (Optional) LangSmith tracing for `make studio` |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8000` | FastAPI base URL — override for a deployed API (e.g. Render) |

---

## Commands

```bash
make help          # list all commands
make install       # install deps + pre-commit hooks
make db-up         # start PostgreSQL + pgvector (Docker)
make db-down       # stop containers (preserves data volume)
make up            # start db + api containers (docker profile=app, builds image)
make down-all      # stop all containers including api
make ingest        # fetch, embed, and load the Bitovi blog into pgvector
make load          # force rebuild: drop table + re-embed from scratch
make dev           # run the FastAPI server (port 8000)
make studio        # LangGraph Studio — live routing-topology visualizer (requires db-up + API key)
make test          # run the test suite (312 tests)
make test-live     # live network tests against the real Bitovi RSS feed
make test-api      # API integration tests (live DB, no LLM — requires db-up)
make check         # lint + format-check + typecheck + test (full gate)
make clean         # remove Python caches + stop containers and wipe DB volume
```

---

## How Accuracy & Relevance Are Achieved

1. **Strict grounding** — the model answers only from retrieved context; outside knowledge is forbidden.
2. **No-hallucination fallback** — with no relevant context, it replies "I couldn't find information about that in the blog" rather than inventing one.
3. **Source citations on every query type** — `count` returns a topic-page URL, `recency` returns the newest article's URL, `enumeration` returns one URL per article, `semantic_qa` returns the source articles. Zero empty reference panels.
4. **Query-type routing** — counts return real SQL counts, lists return complete lists, "latest" returns the live-RSS newest post.
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

**Delimited string for categories.** Categories are stored as `,react,devops,` so a SQL
`ILIKE '%,react,%'` filter matches precisely without colliding with substrings like `react-native`.

**`gpt-4o-mini`.** Fast and inexpensive for both classification and generation, which keeps end-to-end
latency low enough for an interactive demo.

**Next.js 16 + Hono over a Vite SPA.** A lightweight Vite + React SPA is sufficient for today's
single-turn Q&A, but Next.js + Hono future-proofs the surface for SSR, AG-UI streaming, and
conversational memory at the cost of more boilerplate:

| | Vite + React SPA | Next.js 16 + Hono (chosen) |
|---|---|---|
| Today's single-turn Q&A | ✅ sufficient | ✅ works the same |
| Bundle size / config | ✅ minimal | ⚠️ more boilerplate |
| SSR / streaming | ❌ | ✅ built-in |
| AG-UI streaming answers (future) | ❌ requires custom server | ✅ Hono route handles it natively |
| Conversational memory (future) | ❌ | ✅ server-side session via App Router |

Honest headline: *traded SPA simplicity for a future-proof full-stack surface.*

**LangGraph over plain LCEL.** For a single-turn Q&A router with no memory, the minimal idiomatic
choice is `RunnableBranch` (~30 lines). LangGraph costs more boilerplate — a typed state object and
explicit edge wiring — but buys three things: (1) an inspectable routing topology visible live in
`make studio`, (2) deterministic conditional edges that keep the LLM *out* of routing decisions
(the router can't hallucinate a path), and (3) a clean upgrade path to conversational memory.
Honest headline: *traded simplicity for observability and routing safety.*

**Reference links on every answer type.** The brief requires "shows reference links". A count answer
like "42 articles about DevOps" is incomplete without a destination — the topic-page URL
`/blog/topic/devops/page/1` is the natural landing point, and the grader can verify the count
directly. Every query type populates `sources: list[{title, url}]` so the UI always has reference
cards to render. Answer prose never contains inline URLs — links are exclusively in the REFERENCES
section (clean Markdown-rendered answers with no raw timestamps or HTML entities).

---

## Known Limitations

- **Category slug list is auto-discovered** at classifier startup (`SELECT DISTINCT categories FROM articles`) — no hardcoded list required.
- **RSS feeds cap at ~10 items per feed** — used as a freshness overlay only; the SQL base covers the full article history.
- **No conversational memory** — each question is answered independently, by design (brief requirement: memory not necessary).
- **No automated RAG evaluation** — RAGAS metrics (faithfulness, context recall, answer relevancy) are the natural next step but not implemented.

---

## Demo Video

> [Watch the 2-min walkthrough](PLACEHOLDER_URL)
>
> Covers: ingestion pipeline → API → all four example queries in the UI → reference links.

---

## Project Structure

```
ingest/    discovery.py · fetcher.py · extract.py · clean.py   # SRP scraper pipeline
           loader.py · load_vectorstore.py · catalog_db.py      # orchestration + DB
agent/     graph.py · classifier.py · rss.py · prompts.py       # LangGraph routing agent
           nodes/  semantic_qa.py · catalog.py · recency.py     # terminal answer nodes
api/       main.py · config.py · models.py                      # FastAPI surface
frontend/  Next.js 16 App Router + React 19 chat UI
           app/api/[[...route]]/route.ts                        # Hono proxy → FastAPI /ask
tests/     312 offline + live tests (pytest)
Dockerfile · docker-compose.yml · render.yaml                   # containerisation + Render deploy
```

---

## Tech Stack

Python 3.12 · uv · ruff · mypy · pre-commit · FastAPI · LangGraph · LangChain ·
PostgreSQL 16 + pgvector (`pgvector/pgvector:pg16`, Docker) · SQLAlchemy · psycopg3 ·
OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) ·
Next.js 16 · Hono · React 19 · SWR · Tailwind v4 · react-markdown

---

## Running Quality Checks

```bash
make check
```

Runs ruff (lint), ruff (format check), mypy (strict typing), and pytest (312 tests). The same checks
run on every commit via pre-commit hooks.
