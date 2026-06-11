"""FastAPI application — /ask and /health endpoints.

Lifecycle: lifespan builds engine + compiled graph once into app.state.
Import-time opens no DB connection — all I/O lives inside lifespan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, cast

import sqlalchemy as sa
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from agent.graph import build_graph
from api.config import get_settings
from api.models import AskRequest, AskResponse, HealthResponse, SourceRef
from ingest.catalog_db import get_article_count


def _client_ip(request: Request) -> str:
    # Render (and most proxies) set X-Forwarded-For to the real client IP.
    # Fall back to the direct TCP peer when running locally without a proxy.
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_ip)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = sa.create_engine(settings.database_url)
    app.state.engine = engine
    app.state.graph = build_graph(engine=engine).compile()
    yield
    engine.dispose()


async def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again in a moment."},
    )


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_allow_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_graph(request: Request) -> Any:
    return request.app.state.graph


def get_engine(request: Request) -> sa.Engine:
    return cast(sa.Engine, request.app.state.engine)


GraphDep = Annotated[Any, Depends(get_graph)]
EngineDep = Annotated[sa.Engine, Depends(get_engine)]


@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, req: AskRequest, graph: GraphDep) -> AskResponse:
    result = await graph.ainvoke({"question": req.question})
    return AskResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result.get("sources", [])],
        query_type=result["query_type"],
    )


@app.get("/health", response_model=HealthResponse)
async def health(engine: EngineDep) -> HealthResponse:
    return HealthResponse(status="ok", collection_size=get_article_count(engine))
