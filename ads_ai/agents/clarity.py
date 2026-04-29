"""Message Clarity Agent module."""

from __future__ import annotations

import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    AudienceSegments,
    ClarityEvaluation,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class MessageClarityAgent(BaseAgent):
    """Evaluates message clarity and comprehension.

    This agent analyzes ad scripts to ensure the core message is
    communicated quickly, clearly, and without ambiguity to the target
    audience segments.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the MessageClarityAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-flash-lite-preview")

    def evaluate(
        self,
        script: AdScript,
        strategy: StrategyBrief,
        personas: AudienceSegments,
    ) -> ClarityEvaluation:
        """Evaluates whether an ad communicates its message clearly.

        Args:
            script: Ad variant script to evaluate.
            strategy: Master strategy document for consistency check.
            personas: Audience personas for comprehension simulation.

        Returns:
            A ``ClarityEvaluation`` instance with scores and recommendations.

        Raises:
            google.api_core.exceptions.InternalServerError: If evaluation fails.
            ValueError: If response parsing fails.
        """
        logger.info("evaluate started concept=%s", script.concept_title)
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Senior Communications Consultant & Linguistic Auditor.
        Objective: Perform a rigorous audit of an ad script to ensure 100% message clarity
        and zero cognitive friction for the target audience.

        INPUTS:
        - Ad Variant Script: {script.model_dump_json()}
        - Intended Strategy: {strategy.model_dump_json()}
        - Target Audience Personas: {personas.model_dump_json()}

        EXECUTION STEPS:
        1. MESSAGE EXTRACTION: Identify the single most prominent message a viewer will
           take away.
        2. STRATEGIC GAP ANALYSIS: Compare the "Extracted Message" against the "Intended
           Value Prop" from the strategy. Identify any drift.
        3. COGNITIVE AUDIT:
           - Comprehensibility: Can a 10-year-old understand it?
           - Cognitive Load: Are there too many competing claims or complex words?
           - Signal vs. Noise: Is the core offer buried under secondary information?
        4. PERSONA COMPREHENSION SIMULATION: For each persona, write their "3-second summary"
           of what the ad is about.
        5. AMBIGUITY DETECTION: Identify specific words, phrases, or visual sequences
           that could be misinterpreted.
        6. SCORING: Provide objective scores (0-100) based on the "Time-to-Clarity"
           (how many seconds until the offer is known).

        CONSTRAINTS & RULES:
        - NO SUBJECTIVITY: Base scores on linguistic complexity and narrative structure,
          not personal preference.
        - FAIL FAST: If the core message is not clear within the first 5 seconds, the
          Clarity Score must be below 60.
        - OUTPUT DISCIPLINE: Return results as a structured ClarityEvaluation JSON object.
        """
            report = self.generate(prompt, response_schema=ClarityEvaluation)
            elapsed = time.perf_counter() - start
            logger.info(
                "evaluate completed concept=%s elapsed=%.3fs",
                script.concept_title,
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "evaluate failed concept=%s elapsed=%.3fs",
                script.concept_title,
                elapsed,
            )
            raise
