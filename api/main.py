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

from agent.graph import build_graph
from api.config import get_settings
from api.models import AskRequest, AskResponse, HealthResponse, SourceRef
from ingest.catalog_db import get_article_count


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = sa.create_engine(settings.database_url)
    app.state.engine = engine
    app.state.graph = build_graph(engine=engine).compile()
    yield
    engine.dispose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
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
async def ask(req: AskRequest, graph: GraphDep) -> AskResponse:
    result = await graph.ainvoke({"question": req.question})
    return AskResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result.get("sources", [])],
        query_type=result["query_type"],
    )


@app.get("/health", response_model=HealthResponse)
async def health(engine: EngineDep) -> HealthResponse:
    return HealthResponse(status="ok", collection_size=get_article_count(engine))
