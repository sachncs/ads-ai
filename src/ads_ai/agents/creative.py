"""Creative Agent module."""

from __future__ import annotations

import json
import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    AudienceSegments,
    CreativeVariants,
    IterationControlReport,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class CreativeAgent(BaseAgent):
    """Generates multiple ad variant scripts from strategy.

    This agent uses the strategy and audience personas to create
    compelling, platform-specific ad concepts and shot-by-shot scripts.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the CreativeAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def generate_variants(
        self,
        product: str,
        strategy: StrategyBrief,
        personas: AudienceSegments,
        platforms: list[str],
        constraints: str = "",
    ) -> CreativeVariants:
        """Generates multiple ad concepts and scripts.

        Args:
            product: Product description.
            strategy: Strategy document.
            personas: Audience personas simulation results.
            platforms: List of platforms.
            constraints: Optional creative constraints.

        Returns:
            A ``CreativeVariants`` instance containing multiple ad concepts.

        Raises:
            google.api_core.exceptions.InternalServerError: If generation fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "generate_variants started product=%s platforms=%s",
            product[:50],
            platforms,
        )
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Lead Creative Director & Narrative Architect.
        Objective: Produce high-impact, platform-native ad scripts and concepts that
        strictly adhere to the strategy and persona data.

        INPUTS:
        - Product Description: {product}
        - Strategy Brief: {strategy.model_dump_json()}
        - Audience Personas: {personas.model_dump_json()}
        - Target Platform(s): {", ".join(platforms)}
        - Creative Constraints: {constraints}

        EXECUTION STEPS:
        1. CONCEPTUALIZATION: Generate 3-5 distinct creative concepts (e.g., Problem/Solution,
           Direct-to-Camera, Story-driven, Kinetic-Typographic).
        2. NARRATIVE DESIGN: For each variant, design a script with the following
           mandatory elements:
           - Concept Title & Core Hook: Describe the 0:01-0:03 pattern interrupter.
           - Script Scenes: Provide detailed "Visual Cues" (lighting, camera angle, action)
             and matching "Dialogue/VO."
           - Strategic Alignment: Explicitly integrate the "Value Proposition" and
             "Message Pillars."
           - Early Branding: Ensure the brand is identifiable within the first 3 seconds
             and again at the CTA.
           - Call-to-Action (CTA): Use the specific CTA text and urgency type defined in
             the strategy.
        3. VIDEO PROMPT SYNTHESIS: Write a detailed, technical prompt for each variant
           that a video generation model (like Veo) can use to produce the visuals.

        CONSTRAINTS & RULES:
        - NO GENERIC CONTENT: Avoid cliches like "Are you tired of..." or generic business B-roll.
        - PLATFORM NATIVITY: Adjust pacing and "hook" style for the specific platform
          (e.g., TikTok vs. LinkedIn).
        - EARLY BRANDING: Failure to introduce the brand or product name early is a
          critical failure.
        - OUTPUT DISCIPLINE: Return results as a structured CreativeVariants JSON object.
        """
            report = self.generate(prompt, response_schema=CreativeVariants)
            elapsed = time.perf_counter() - start
            logger.info(
                "generate_variants completed product=%s variant_count=%d elapsed=%.3fs",
                product[:50],
                len(report.variants),
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "generate_variants failed product=%s elapsed=%.3fs",
                product[:50],
                elapsed,
            )
            raise

    def refine_variants(
        self,
        variants: list[AdScript],
        iteration_report: IterationControlReport,
    ) -> list[AdScript]:
        """Refines existing ad variants based on iteration directives.

        Args:
            variants: The current list of ad scripts.
            iteration_report: Actionable refinement instructions.

        Returns:
            A list of improved ``AdScript`` instances.

        Raises:
            google.api_core.exceptions.InternalServerError: If refinement fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "refine_variants started variant_count=%d",
            len(variants),
        )
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Elite Creative Editor & Narrative Optimizer.
        Objective: Surgically refine the provided ad scripts based on the specific,
        actionable directives in the 'Iteration Report.'

        INPUTS:
        - Current Ad Scripts: {json.dumps([v.model_dump() for v in variants])}
        - Iteration Control Report: {iteration_report.model_dump_json()}

        EXECUTION STEPS:
        1. DIRECTIVE ADHERENCE: For each variant, apply the specific "Refinement Directives"
           point-by-point.
        2. SCRIPT SURGERY: Modify scenes, hooks, or CTAs as instructed. Ensure the
           character and soul of the creative is preserved while fixing the identified flaws.
        3. BRAND & KPI ALIGNMENT: Ensure the changes improve 'Brand Attribution' or
           'Clarity' as requested in the directives.
        4. COHERENCE CHECK: Verify that the refined script is logically consistent and
           maintains the correct platform pacing.

        CONSTRAINTS & RULES:
        - PRECISION: Do NOT make random changes. Only edit according to directives.
        - COMPLETENESS: Return the full, updated scripts for ALL variants.
        - OUTPUT DISCIPLINE: Return results as a structured CreativeVariants JSON object.
        """
            report: CreativeVariants = self.generate(
                prompt,
                response_schema=CreativeVariants,
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "refine_variants completed variant_count=%d elapsed=%.3fs",
                len(report.variants),
                elapsed,
            )
            return report.variants
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "refine_variants failed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            raise
