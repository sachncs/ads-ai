"""Creative Diagnostics Agent module."""

from __future__ import annotations

import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    AudienceSegments,
    DiagnosticsEvaluation,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class DiagnosticsAgent(BaseAgent):
    """Evaluates structural integrity and pacing.

    This agent performs a technical "teardown" of ad scripts, analyzing
    narrative flow, hook strength, and potential viewer drop-off points.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the DiagnosticsAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-flash-lite-preview")

    def evaluate(
        self,
        script: AdScript,
        strategy: StrategyBrief,
        personas: AudienceSegments,
        platform_constraints: str = "",
    ) -> DiagnosticsEvaluation:
        """Evaluates structural integrity and narrative effectiveness.

        Args:
            script: Ad variant script to evaluate.
            strategy: Strategy brief for narrative alignment.
            personas: Audience personas for drop-off simulation.
            platform_constraints: Textual constraints for specific platforms.

        Returns:
            A ``DiagnosticsEvaluation`` instance with structural analysis.

        Raises:
            google.api_core.exceptions.InternalServerError: If diagnostics fails.
            ValueError: If response parsing fails.
        """
        logger.info("evaluate started concept=%s", script.concept_title)
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Senior Film Editor & Narrative Architect.
        Objective: Conduct a technical teardown of an ad script to optimize its pacing,
        narrative flow, and viewer retention mechanics.

        INPUTS:
        - Ad Variant Script: {script.model_dump_json()}
        - Strategic Strategy: {strategy.model_dump_json()}
        - Audience Personas: {personas.model_dump_json()}
        - Platform Constraints: {platform_constraints}

        EXECUTION STEPS:
        1. STRUCTURAL CLASSIFICATION: Map the script to a known high-performance structure
           (e.g., Problem-Agitate-Solve, Inverted Pyramid, The Loop).
        2. PACING AUDIT: Identify "Dead Zones"—any sequence longer than 3 seconds without
           a new visual or narrative hook.
        3. HOOK STRENGTH TEST: Evaluate the opening 3 seconds. Does it create an "Open Loop"
           in the viewer's mind?
        4. RETENTION RISK MAPPING: Identify exactly where a persona is likely to drop off.
           Map this to the "Drop-off Risk Timeline."
        5. REDUNDANCY EXTRACTION: Identify specific lines or scenes that repeat information
           already conveyed.
        6. TRANSITION EVALUATION: Are the jumps between scenes logical or jarring?
        7. PLATFORM SUITABILITY: Does the structure match the platform behavior (e.g., 9:16
           vertical storytelling for TikTok)?

        CONSTRAINTS & RULES:
        - PRECISION: When recommending removals, specify the EXACT scene or line number.
        - RETENTION BIAS: Prioritize "Shortening" over "Elaborating." The goal is maximum
          density with minimum friction.
        - OUTPUT DISCIPLINE: Return results as a structured DiagnosticsEvaluation JSON object.
        """
            report = self.generate(prompt,
                                   response_schema=DiagnosticsEvaluation)
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
