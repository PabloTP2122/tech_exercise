.PHONY: install dev db-up db-down up down-all ingest load test test-live test-api golden lint format format-check typecheck check eyeball eyeball-all discover-check preflight studio clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install deps + pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install

db-up: ## Start PostgreSQL + pgvector container (detached)
	docker compose up -d db

db-down: ## Stop containers (preserves data volume)
	docker compose down

up: ## Start db + api containers (requires docker; builds image)
	docker compose --profile app up -d --build

down-all: ## Stop all containers including api
	docker compose --profile app down

dev: ## Start FastAPI dev server on :8000  (requires db-up)
	uv run uvicorn api.main:app --reload --port 8000

ingest: ## Fetch articles -> embed -> load pgvector (skips if already populated)
	uv run python -m ingest.load_vectorstore

load: ## Force rebuild: drop table + re-embed from scratch
	uv run python -m ingest.load_vectorstore --force

test: ## Run pytest
	uv run pytest -v

test-live: ## Run live network tests (real Bitovi feed)
	uv run pytest -v -m live

test-api: ## Run API integration tests (live DB, no LLM — requires db-up)
	uv run pytest -v -m api_live

golden: ## Run golden-query harness end-to-end (live DB + OPENAI_API_KEY — requires db-up)
	uv run pytest -v -m golden_live tests/test_golden_queries.py

eyeball: ## Eyeball 1-doc vector-view: fetch one article, run the pipeline, write test_1.json
	uv run pytest tests/test_pipeline_smoke.py -v

eyeball-all: ## Full ~462-doc vector-view report (network, ~5 min, no embed/DB)
	uv run python -m ingest.preview

discover-check: ## Live reconcile report: sitemap vs paginator (network, ~2 min, no embed/DB)
	uv run python -m ingest.discovery

preflight: check eyeball ## Full quality gate + 1-doc eyeball — run before embedding

lint: ## Ruff lint (check only)
	uv run ruff check .

format: ## Ruff format (auto-fix)
	uv run ruff format .

format-check: ## Ruff format (check only)
	uv run ruff format --check .

typecheck: ## mypy
	uv run mypy .

check: lint format-check typecheck test ## Full quality gate

studio: ## Launch LangGraph Studio against the compiled graph (requires db-up + OPENAI_API_KEY)
	uv run langgraph dev

clean: ## Remove Python caches + stop containers and wipe DB volume
	docker compose down -v
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/
