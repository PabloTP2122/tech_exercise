.PHONY: install dev ingest load test lint format format-check typecheck check clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install deps + pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install

dev: ## Start FastAPI dev server on :8000
	uv run uvicorn api.main:app --reload --port 8000

ingest: ## Fetch articles -> embed -> load ChromaDB (skips if already populated)
	uv run python -m ingest.load_chroma

load: ## Force rebuild: delete collection + re-embed from scratch
	uv run python -m ingest.load_chroma --force

test: ## Run pytest
	uv run pytest -v

lint: ## Ruff lint (check only)
	uv run ruff check .

format: ## Ruff format (auto-fix)
	uv run ruff format .

format-check: ## Ruff format (check only)
	uv run ruff format --check .

typecheck: ## mypy
	uv run mypy .

check: lint format-check typecheck test ## Full quality gate

clean: ## Remove caches and the vector store
	rm -rf chroma_db/ .mypy_cache/ .ruff_cache/ .pytest_cache/
