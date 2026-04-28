"""Platform Adaptation Agent module."""

from __future__ import annotations

import json
import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    AudienceSegments,
    PlatformAdaptationReport,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class PlatformAdaptationAgent(BaseAgent):
    """Adapts ad variants for specific social media platforms.

    This agent transforms approved ad scripts into platform-native
    formats, ensuring headlines, body copy, and CTAs follow network best
    practices.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the PlatformAdaptationAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def adapt(
        self,
        variant: AdScript,
        strategy: StrategyBrief,
        personas: AudienceSegments,
        platforms: list[str],
    ) -> PlatformAdaptationReport:
        """Adapts a single ad script for multiple platforms.

        Args:
            variant: The approved ad script to adapt.
            strategy: The master strategy brief.
            personas: Target audience segments.
            platforms: List of target social platforms (e.g., Meta, TikTok).

        Returns:
            A ``PlatformAdaptationReport`` with adapted creative copy.

        Raises:
            google.api_core.exceptions.InternalServerError: If adaptation fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "adapt started concept=%s platforms=%s",
            variant.concept_title,
            platforms,
        )
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Senior Performance Marketer & Platform Creative Specialist.
        Objective: Transform an approved ad script into high-performance, platform-native
        creative variants (Meta, TikTok, YouTube, LinkedIn) that respect technical
        constraints and user behavior.

        INPUTS:
        - Approved Ad Script: {variant.model_dump_json()}
        - Strategy Context: {strategy.model_dump_json()}
        - Target Personas: {personas.model_dump_json()}
        - Target Platforms: {json.dumps(platforms)}

        EXECUTION STEPS:
        1. PLATFORM RULESET APPLICATION: Apply specific character limits, aspect ratios
           (9:16, 1:1, 16:9), and duration constraints for each platform.
        2. CREATIVE REWRITING:
           - Headlines: Optimize for each platform's UI (e.g., punchy for TikTok,
             professional for LinkedIn).
           - Body Copy: Adjust the length and tone to match native user behavior.
           - CTA: Use platform-specific action verbs (e.g., "Shop Now", "Learn More",
             "Swipe Up").
        3. VISUAL OVERLAY DESIGN: Specify where text overlays should be placed to avoid
           "UI Dead Zones" (e.g., avoiding the TikTok description area).
        4. NATIVE TRIGGER INTEGRATION: Add platform-specific cues (e.g., "Save for later",
           "Link in Bio").

        CONSTRAINTS & RULES:
        - BRAND CONSISTENCY: Every adaptation must maintain the core "Value Prop" and
          "Tone" from the approved script.
        - TECHNICAL ACCURACY: Do NOT exceed platform character limits for headlines
          or descriptions.
        - OUTPUT DISCIPLINE: Return results as a structured PlatformAdaptationReport
          JSON object.
        """
            report = self.generate(prompt, response_schema=PlatformAdaptationReport)
            elapsed = time.perf_counter() - start
            logger.info(
                "adapt completed concept=%s platform_count=%d elapsed=%.3fs",
                variant.concept_title,
                len(platforms),
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "adapt failed concept=%s elapsed=%.3fs",
                variant.concept_title,
                elapsed,
            )
            raise
