"""Tests for api/main.py — offline TestClient suite + @pytest.mark.api_live integration test."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_engine, get_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_graph() -> Any:
    g = MagicMock()
    g.ainvoke = AsyncMock(
        return_value={
            "answer": "Use Cypress for E2E testing.",
            "sources": [{"title": "E2E Guide", "url": "https://bitovi.com/blog/e2e"}],
            "query_type": "semantic_qa",
        }
    )
    return g


@pytest.fixture()
def offline_client(fake_graph: Any) -> Generator[TestClient, None, None]:
    """TestClient without lifespan (skips DB/LLM) + dependency overrides."""
    app.dependency_overrides[get_graph] = lambda: fake_graph
    app.dependency_overrides[get_engine] = lambda: MagicMock()
    yield TestClient(app)  # no `with` — lifespan NOT triggered
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /ask — offline
# ---------------------------------------------------------------------------


class TestAskEndpoint:
    def test_happy_path_returns_200(self, offline_client: TestClient) -> None:
        resp = offline_client.post("/ask", json={"question": "What tools for E2E testing?"})
        assert resp.status_code == 200

    def test_response_has_required_keys(self, offline_client: TestClient) -> None:
        resp = offline_client.post("/ask", json={"question": "What tools for E2E testing?"})
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "query_type" in body

    def test_response_body_matches_fake_graph(self, offline_client: TestClient) -> None:
        resp = offline_client.post("/ask", json={"question": "What tools for E2E testing?"})
        body = resp.json()
        assert body["answer"] == "Use Cypress for E2E testing."
        assert body["query_type"] == "semantic_qa"
        assert body["sources"] == [{"title": "E2E Guide", "url": "https://bitovi.com/blog/e2e"}]

    def test_empty_question_returns_422(self, offline_client: TestClient) -> None:
        resp = offline_client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_whitespace_question_returns_422(self, offline_client: TestClient) -> None:
        resp = offline_client.post("/ask", json={"question": "   "})
        assert resp.status_code == 422

    def test_question_over_500_chars_returns_422(self, offline_client: TestClient) -> None:
        resp = offline_client.post("/ask", json={"question": "a" * 501})
        assert resp.status_code == 422

    def test_cache_hit_returns_same_response(self, offline_client: TestClient) -> None:
        import api.main as main_module

        main_module._cache.clear()
        q = "What tools for E2E testing?"
        r1 = offline_client.post("/ask", json={"question": q})
        r2 = offline_client.post("/ask", json={"question": q})
        assert r1.json() == r2.json()

    def test_recency_not_cached(self, offline_client: TestClient, fake_graph: Any) -> None:
        import api.main as main_module

        main_module._cache.clear()
        fake_graph.ainvoke = AsyncMock(
            return_value={
                "answer": "Latest post.",
                "sources": [],
                "query_type": "recency",
            }
        )
        offline_client.post("/ask", json={"question": "latest post?"})
        assert "latest post?" not in main_module._cache


# ---------------------------------------------------------------------------
# /health — offline
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, offline_client: TestClient) -> None:
        with patch("api.main.get_article_count", return_value=462):
            resp = offline_client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, offline_client: TestClient) -> None:
        with patch("api.main.get_article_count", return_value=462):
            resp = offline_client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"
        assert isinstance(body["collection_size"], int)

    def test_health_returns_correct_count(self, offline_client: TestClient) -> None:
        with patch("api.main.get_article_count", return_value=42):
            resp = offline_client.get("/health")
        assert resp.json()["collection_size"] == 42


# ---------------------------------------------------------------------------
# /health — live integration (make test-api only, requires db-up, no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.api_live
def test_health_live() -> None:
    """Real lifespan against a live DB. DB-only — no LLM, no OpenAI key needed."""
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["collection_size"] > 0
