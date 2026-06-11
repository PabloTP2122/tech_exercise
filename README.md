# Bitovi Blog RAG Q&A Agent

A query-type-aware Retrieval-Augmented Generation agent that answers questions about the Bitovi blog,
grounded exclusively in the blog's own articles, with source links on every response.

Built with FastAPI, LangGraph, pgvector (PostgreSQL), Next.js 16 + Hono + React 19. Python 3.12, managed with `uv`.

---

## How this meets the requirements

| Requirement | Where |
|---|---|
| Ingest all Bitovi blog articles | `ingest/` pipeline — `make ingest` fetches, embeds, and loads ~462 articles (Real ~460) |
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
  classify_query  →  ┌ semantic_qa → hybrid retrieval (vector top-k + keyword
                     │               full-text, RRF-fused, deduped) → generate (LLM)
                     ├ enumeration → SQL articles WHERE categories ILIKE [AND year range]
                     ├ count       → SQL COUNT(*) WHERE categories ILIKE [AND year range]
                     └ recency     → SQL ORDER BY published_at ASC|DESC + RSS overlay
                                  →  grounded answer with source citations

SERVE
  FastAPI  POST /ask · GET /health           (port 8000)
  Next.js  App Router + Hono /api/* proxy    (port 3000)
           → proxies questions to FastAPI /ask
```

---

## A request, step by step

What happens when you type **"How many articles does Bitovi have about AI?"** in the UI:

1. The Next.js page POSTs to its Hono proxy (`frontend/app/api/[[...route]]/route.ts`), which forwards to FastAPI `POST /ask` (`api/main.py`).
2. The compiled LangGraph runs `classify` (`agent/classifier.py`): the ordered regex table in `agent/classification_rules.py` matches `how many` → `count` (zero LLM calls), and the fuzzy slug pass matches `ai`. Slots like a year filter or recency direction are extracted here too.
3. `route_query` (`agent/routes/route.py`) sends the state to the `sql_count` node (`agent/nodes/catalog.py`), which runs `SELECT COUNT(DISTINCT source_url) … WHERE categories ILIKE ',ai,'` against the `articles` catalog — no chunks, no LLM, no way to hallucinate a number.
4. The node renders a deterministic sentence (`agent/prompts.py:render_count`) and attaches the topic-page URL as the reference link.
5. FastAPI returns `{answer, sources, query_type}`; the UI renders the answer and one reference card per source.

The only path that calls the LLM is `semantic_qa`: hybrid retrieval (`agent/retrieval.py` + pgvector) → relevance gate (below `SIMILARITY_THRESHOLD` → canonical "I couldn't find…" with no sources) → generate with structured output, whose cited URLs are whitelisted against the retrieved docs.

## An article, step by step

How a blog post becomes searchable (`make ingest`, one-time):

1. **Discover** (`ingest/discovery.py`) — sitemap + paginator cross-check yields ~462 article URLs.
2. **Fetch** (`ingest/fetcher.py`) — polite HTTP (2 req/s).
3. **Extract** (`ingest/extract.py`) — title/author/date from JSON-LD with OpenGraph/HTML fallbacks (regex recovery for malformed JSON-LD lives in `ingest/recover.py`); categories from `/blog/topic/` links.
4. **Clean** (`ingest/clean.py`) — HTML → Markdown, nav/footer stripped.
5. **Chunk + embed** (`ingest/load_vectorstore.py`) — 800-char chunks with 100 overlap; each chunk gets a contextual header and `<DATA_SOURCE>` delimiters, then is embedded into the pgvector `chunks` table. A chunk as embedded literally looks like:

   ```
   Page Title · ,react,devops, · https://www.bitovi.com/blog/page-slug

   <DATA_SOURCE>
   …chunk body in markdown…
   </DATA_SOURCE>
   ```

6. **Catalog** (`ingest/catalog_db.py`) — one row per article (`title`, `source_url`, `published_at`, `categories`) powering the count/enumeration/recency SQL routes.

## Glossary

| Term | Meaning |
|---|---|
| **Chunk** | One 800-char slice of an article (100-char overlap), prefixed with a contextual header so it stays self-describing in isolation |
| **Vector-view** | The exact strings the embedder receives — inspectable without paying for embeddings via `make eyeball-all` |
| **Query type** | One of 4 routes: `semantic_qa` (LLM), `count`, `enumeration`, `recency` (all SQL, no LLM) |
| **Slot** | A value the classifier extracts alongside the route: category slug, year, recency direction/limit |
| **Relevance gate** | Conditional edge: if no retrieved chunk scores ≥ `SIMILARITY_THRESHOLD`, answer the canonical no-match sentence instead of calling the LLM |
| **Hybrid retrieval / RRF** | Vector similarity + Postgres full-text search, merged with Reciprocal Rank Fusion; keyword hits enrich context but never open the relevance gate |
| **Stopword slug** | A topic slug that is also a common English word ("about", "does") — guarded so it can't be matched by accident |
| **Reference-link parity** | Invariant: every answer (including counts and empty results) ships ≥ 1 `{title, url}` source for the UI |
| **No-match response** | The exact sentence "I couldn't find information about that in the blog." — shared constant, returned with zero sources |

## Where to start reading

| If you want to… | Start at |
|---|---|
| Follow a question through the system | `agent/graph.py` (`build_graph`) |
| Add or tune a routing rule | `agent/classification_rules.py` (data-only table with a `why` per rule) |
| Change how answers are worded | `agent/prompts.py` (deterministic renderers + the one LLM prompt) |
| Touch retrieval quality | `agent/retrieval.py` + `agent/nodes/semantic_qa.py` |
| Change what gets ingested | `ingest/load_vectorstore.py` (`main`) |
| See what the agent promises to answer | `tests/golden_queries.py` |

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Every answer is "I couldn't find information…" | DB is empty — run `make db-up` then `make ingest` |
| `curl /ask` connection refused | API not running (`make dev`) or DB container down (`make db-up`) |
| Counts return 0 for a real topic | Slug mismatch — check `SELECT DISTINCT categories FROM articles`; the classifier only matches discovered slugs |
| "Latest post" looks stale | RSS overlay unreachable (it fails silently to SQL); re-run `make ingest` to refresh the catalog |
| Want to rule hybrid search in/out of a regression | Set `KEYWORD_TOP_K=0` to fall back to pure vector retrieval |
| Want an end-to-end quality signal | `make golden` runs the golden-query harness against the live stack |

---

## Try these

The brief's four example queries — paste any into the UI or `curl` them directly:

| Query | Routes to |
|---|---|
| `"What is Bitovi's latest blog post about?"` | `recency` |
| `"Can you show me all Bitovi articles about DevOps?"` | `enumeration` |
| `"How many articles does Bitovi have about AI?"` | `count` |
| `"What kind of tools does Bitovi recommend for E2E testing?"` | `semantic_qa` |

Edge cases that are also answered deterministically:

| Query | Routes to |
|---|---|
| `"What was Bitovi's first blog post?"` | `recency` (oldest direction) |
| `"Show me the last 5 posts"` | `recency` (parsed item count, capped at 10) |
| `"How many articles did Bitovi publish in 2023?"` | `count` (year filter) |
| `"Show me all articles from 2023"` | `enumeration` (year filter) |

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
| `SIMILARITY_THRESHOLD` | | `0.35` | Relevance gate floor (0–1); below it the agent answers "I couldn't find…" |
| `RETRIEVAL_TOP_K` | | `4` | Vector hits fetched per semantic question |
| `KEYWORD_TOP_K` | | `4` | Full-text hits fused via RRF; set `0` to disable hybrid search |
| `MAX_CHUNKS_PER_ARTICLE` | | `2` | Context-diversity cap applied after fusion |
| `CORS_ALLOW_ORIGINS` | | `http://localhost:3000,…` | Comma-separated allowed origins (set at deploy time) |
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
make test          # run the offline test suite (395 tests)
make test-live     # live network tests against the real Bitovi RSS feed
make test-api      # API integration tests (live DB, no LLM — requires db-up)
make golden        # golden-query harness end-to-end (live DB + OpenAI — requires db-up)
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
8. **Hybrid retrieval** — Postgres full-text search catches exact terms ("Cypress", "E2E") that embeddings can blur; results are RRF-fused with the vector ranking, and a per-article cap keeps the context diverse. Keyword hits never bypass the relevance gate, so the no-hallucination fallback is unaffected.
9. **Golden-query harness** — `tests/golden_queries.py` pins the routes, sources, and key phrases for 14 representative questions (including a known-negative); the routing tier runs in every `make check`, the end-to-end tier via `make golden`.

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
- **No conversational memory** — each question is answered independently, by design (requirements: memory not necessary).
- **Evaluation is assertion-based, not metric-based** — the golden-query harness (`make golden`) asserts routes, sources, and key phrases end-to-end; RAGAS-style metrics (faithfulness, context recall) remain the natural next step.

---

## Demo Video

> [Watch the 2-min walkthrough](PLACEHOLDER_URL)
>
> Covers: ingestion pipeline → API → all four example queries in the UI → reference links.

---

## Project Structure

```
ingest/    discovery.py · fetcher.py · extract.py · recover.py · clean.py  # SRP scraper pipeline
           loader.py · load_vectorstore.py · catalog_db.py      # orchestration + DB
agent/     graph.py · classifier.py · classification_rules.py   # LangGraph routing agent
           retrieval.py · rss.py · prompts.py                   # hybrid search + rendering
           nodes/  semantic_qa.py · catalog.py · recency.py     # terminal answer nodes
api/       main.py · config.py · models.py                      # FastAPI surface
frontend/  Next.js 16 App Router + React 19 chat UI
           app/api/[[...route]]/route.ts                        # Hono proxy → FastAPI /ask
tests/     395 offline + live tests (pytest) · golden_queries.py  # incl. golden-query harness
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

Runs ruff (lint), ruff (format check), mypy (strict typing), and pytest (395 offline tests). The same
checks run on every commit via pre-commit hooks.
