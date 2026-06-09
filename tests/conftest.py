"""Shared pytest configuration for offline test suite.

All tests in this project are fully offline (no real DB, no real LLM calls).
LangChain clients are monkeypatched where needed.  The only requirement for
``api.config.Settings`` to load is that ``OPENAI_API_KEY`` is non-empty.
This module provides that value so the test suite never needs a real key.
"""

import os

import pytest

from api.config import get_settings


@pytest.fixture(autouse=True, scope="session")
def _fake_openai_key() -> None:
    """Ensure OPENAI_API_KEY is set before any test touches get_settings().

    Uses ``setdefault`` so a real key already in the environment (CI, dev
    shell) is never overwritten.  After setting, the lru_cache is cleared
    so that any test-ordering edge case gets a fresh Settings load.
    """
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-offline-tests")
    get_settings.cache_clear()
