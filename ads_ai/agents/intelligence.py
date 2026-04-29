"""URL Intelligence Agent module."""

from __future__ import annotations

import ipaddress
import logging
from urllib.parse import urlparse

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import ExtractedInputs
from ads_ai.utils import agent_timing, build_url_intelligence_prompt

logger = logging.getLogger(__name__)

MAX_URL_LENGTH = 2048
"""Maximum allowed URL length."""


class URLValidationError(ValueError):
    """Raised when a URL fails security or format validation."""


class URLIntelligenceAgent(BaseAgent):
    """Parses product URLs into structured advertising inputs.

    This agent uses the LLM to analyze the content of a landing page and
    extract the necessary brand and product information to seed the
    pipeline.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the URLIntelligenceAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    @staticmethod
    def _validate_url(url: str) -> None:
        """Validates a URL for security before passing it to the LLM.

        Args:
            url: The URL to validate.

        Raises:
            URLValidationError: If the URL is malformed, uses a non-HTTP(S)
                scheme, targets a private IP, or exceeds the length limit.
        """
        if len(url) > MAX_URL_LENGTH:
            raise URLValidationError(
                f"URL exceeds maximum length of {MAX_URL_LENGTH} characters.")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise URLValidationError(
                f"URL scheme must be http or https; got '{parsed.scheme}'.")

        hostname = parsed.hostname
        if not hostname:
            raise URLValidationError("URL is missing a valid hostname.")

        try:
            addr = ipaddress.ip_address(hostname)
        except ValueError:
            return

        if addr.is_private or addr.is_loopback or addr.is_reserved:
            raise URLValidationError(
                f"URL targets a restricted IP address: {hostname}.")

    def parse_url(self, url: str) -> ExtractedInputs:
        """Extracts brand and product data from the provided URL.

        Args:
            url: The landing page URL to analyze.

        Returns:
            An ``ExtractedInputs`` instance containing the parsed data.

        Raises:
            URLValidationError: If the URL fails security validation.
            google.api_core.exceptions.InternalServerError: If URL access fails.
            ValueError: If response parsing fails.
        """
        self._validate_url(url)
        logger.info("parse_url started url=%s", url)
        with agent_timing(logger, "parse_url", "url", url):
            prompt = build_url_intelligence_prompt(url=url)
            report = self.generate(prompt, response_schema=ExtractedInputs)

        return report
