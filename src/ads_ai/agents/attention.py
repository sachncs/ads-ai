"""Attention Heuristic Agent module."""

from __future__ import annotations

import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    AttentionEvaluation,
    AudienceSegments,
    DiagnosticsEvaluation,
)

logger = logging.getLogger(__name__)


class AttentionHeuristicAgent(BaseAgent):
    """Estimates attention capture and retention.

    This agent uses visual and narrative heuristics to predict how
    likely an ad is to stop the scroll and sustain attention across
    different platforms.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the AttentionHeuristicAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-flash-lite-preview")

    def evaluate(
        self,
        script: AdScript,
        diagnostics: DiagnosticsEvaluation | None,
        personas: AudienceSegments,
        platform_context: str = "short-form feed",
    ) -> AttentionEvaluation:
        """Estimates the likelihood that an ad will capture attention.

        Args:
            script: Ad variant script to evaluate.
            diagnostics: Structural diagnostics for pacing context.
            personas: Audience personas for attention modeling.
            platform_context: The behavioral context (e.g., "story feed").

        Returns:
            An ``AttentionEvaluation`` instance with heuristic scores.

        Raises:
            google.api_core.exceptions.InternalServerError: If analysis fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "evaluate started concept=%s platform_context=%s",
            script.concept_title,
            platform_context,
        )
        start = time.perf_counter()
        try:
            diag_json = (diagnostics.model_dump_json()
                         if diagnostics else "N/A (Independent Evaluation)")
            prompt = f"""
        Role: Senior Attention Architect & Media Psychologist.
        Objective: Use visual and narrative heuristics to predict an ad's ability to
        "Stop the Scroll" and maintain high viewer density on social feeds.

        INPUTS:
        - Ad Variant Script: {script.model_dump_json()}
        - Creative Diagnostics: {diag_json}
        - Audience Personas: {personas.model_dump_json()}
        - Platform Context: {platform_context}

        EXECUTION STEPS:
        1. SCROLL-STOP AUDIT (0-3 SECONDS): Identify the specific "Pattern Interrupter"
           used. Is it visual, auditory, or narrative?
        2. NOVELTY ASSESSMENT: Compare the hook against current platform cliches. Is the
           opening predictable or fresh?
        3. VISUAL DENSITY ANALYSIS: Evaluate the balance of text-on-screen, subject motion,
           and background complexity. Identify risk of "Visual Overload."
        4. RETENTION CURVE PREDICTION: Map the "Emotional Velocity" of the script. Identify
           where interest peaks and where it plateaus.
        5. PERSONA-SPECIFIC SALIENCY: For each persona, identify the ONE sensory trigger
           likely to capture their attention.
        6. SCORING:
           - First Impression Score (Saliency).
           - Scroll-Stop Probability (Predictive %).
           - Novelty Score (Uniqueness).

        CONSTRAINTS & RULES:
        - HEURISTIC-ONLY: Base all predictions on proven attention-hacking principles (e.g.,
          The Von Restorff effect, Pattern Interruption), not generic liking.
        - CRITICAL BIAS: Assume the viewer is "Active-Avoiding" ads. Any weak opening must
          result in a Scroll-Stop Probability < 30%.
        - OUTPUT DISCIPLINE: Return results as a structured AttentionEvaluation JSON object.
        """
            report = self.generate(prompt, response_schema=AttentionEvaluation)
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
