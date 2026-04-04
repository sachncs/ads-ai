"""Application configuration loaded from environment variables.

Settings are resolved in the following priority order:
1. Explicit environment variables.
2. Values defined in a ``.env`` file at the project root.
3. Hard-coded defaults in this module.
"""

from __future__ import annotations

import logging

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "gemini-3.1-pro-preview"
"""Default model for complex reasoning and generative agent tasks."""

DEFAULT_EVALUATION_MODEL = "gemini-3.1-flash-lite-preview"
"""Default model for lightweight evaluation agent tasks."""

DEFAULT_VIDEO_MODEL = "veo-3.1-generate-preview"
"""Default model for video generation tasks."""

DEFAULT_MAX_RETRIES = 3
"""Maximum attempts for Gemini API transient failures."""

DEFAULT_RETRY_DELAY = 5
"""Initial delay (seconds) before retrying a failed request."""

DEFAULT_LOG_LEVEL = "INFO"
"""Fallback log level when not specified via environment."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Centralised application settings bound to environment variables.

    Attributes:
        gemini_api_key: Google GenAI API key.  Required at runtime.
        default_text_model: Model identifier for generative tasks.
        default_evaluation_model: Model identifier for evaluation tasks.
        log_level: Python logging level string.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    default_text_model: str = DEFAULT_TEXT_MODEL
    default_evaluation_model: str = DEFAULT_EVALUATION_MODEL
    default_video_model: str = DEFAULT_VIDEO_MODEL
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: int = DEFAULT_RETRY_DELAY
    log_level: str = DEFAULT_LOG_LEVEL


def get_settings() -> Settings:
    """Creates and returns a new ``Settings`` instance.

    Returns:
        A fully resolved ``Settings`` object.
    """
    return Settings()


settings = get_settings()
