# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.12-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install production dependencies first (layer is cached until lock-file changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY --from=builder /app/.venv /app/.venv
COPY api/ api/
COPY agent/ agent/
COPY ingest/ ingest/

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app

USER appuser

# exec form + explicit shell so ${PORT:-8000} expands at runtime AND uvicorn gets
# PID 1 (receives SIGTERM directly for graceful shutdown).
CMD ["/bin/sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
