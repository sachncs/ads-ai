"""Tests for configuration and environment validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ads_ai.config import Settings


class TestConfig:
    """Tests for the Settings and environment logic."""

    def test_settings_default_values(self) -> None:
        """Should have sensible defaults for model names and logs."""
        settings = Settings(_env_file=None) # type: ignore
        assert settings.default_text_model == "gemini-3.1-pro-preview"
        assert settings.log_level == "INFO"

    def test_settings_env_override(self) -> None:
        """Should allow overriding via environment variables."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.gemini_api_key == "test-key"
            assert settings.log_level == "DEBUG"

    def test_model_names_validation(self) -> None:
        """Should validate model name formats if needed (placeholder for future)."""
        # Currently no strict regex on model names, but we can verify they are strings
        settings = Settings(default_text_model="custom-model")
        assert settings.default_text_model == "custom-model"
