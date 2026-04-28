"""URL Intelligence Agent module."""

from __future__ import annotations

import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import ExtractedInputs

logger = logging.getLogger(__name__)


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

    def parse_url(self, url: str) -> ExtractedInputs:
        """Extracts brand and product data from the provided URL.

        Args:
            url: The landing page URL to analyze.

        Returns:
            An ``ExtractedInputs`` instance containing the parsed data.

        Raises:
            google.api_core.exceptions.InternalServerError: If URL access fails.
            ValueError: If response parsing fails.
        """
        logger.info("parse_url started url=%s", url)
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Senior Business Intelligence Analyst & Competitive Researcher.
        Objective: Conduct a technical extraction of product value propositions,
        competitive advantages, and market positioning from a provided URL.

        INPUTS:
        - Product URL: {url}

        EXECUTION STEPS:
        1. FEATURE-TO-BENEFIT MAPPING: Extract raw features and translate them into
           quantifiable consumer benefits.
        2. COMPETITIVE DIFFERENTIATION: Identify the "Only-ness" factor—what can this
           product do that others cannot?
        3. TARGET MARKET IDENTIFICATION: Infer the primary vertical and sub-sectors.
        4. OBJECTION SCAN: Identify potential reasons for purchase friction (e.g., price,
           complexity, trust).
        5. SEMANTIC EXTRACTION: Capture 5 high-intent keywords and 3 "Emotional Hooks"
           found in the source material.

        CONSTRAINTS & RULES:
        - ZERO HALLUCINATION: Do NOT invent features. If a benefit isn't explicitly
          supported by the text, flag it as "Inferred."
        - PRECISION: Use exact numbers and data points (prices, speeds, stats) if present.
        - NO UNCERTAINTY: If a field cannot be inferred, state "Information not available."
        - OUTPUT DISCIPLINE: Return result as structured ExtractedInputs JSON object.
        """
            report = self.generate(prompt, response_schema=ExtractedInputs)
            elapsed = time.perf_counter() - start
            logger.info(
                "parse_url completed url=%s elapsed=%.3fs",
                url,
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "parse_url failed url=%s elapsed=%.3fs",
                url,
                elapsed,
            )
            raise
