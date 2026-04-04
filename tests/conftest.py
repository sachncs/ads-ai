"""Shared pytest fixtures for the ads.ai test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Injects a dummy API key so tests never require real credentials."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock-test-key")
