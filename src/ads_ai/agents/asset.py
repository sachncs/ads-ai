"""Asset Production Agent module."""

from __future__ import annotations

import json

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AdScript, AssetProductionReport, PlatformVariant


class AssetProductionAgent(BaseAgent):
    """Converts approved ad scripts into production-ready creative asset plans.

    This agent analyzes scripts and adaptations to define the technical
    specifications, scenes, and visual/auditory ingredients required for
    production.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the AssetProductionAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def plan_production(self,
                        variants: list[AdScript],
                        platform_variants: list[PlatformVariant],
                        constraints: str = "",
                        brand_guidelines: str = "") -> AssetProductionReport:
        """Converts approved ad scripts into production-ready asset plans.

        Args:
            variants: List of final approved ad scripts.
            platform_variants: List of platform-adapted script versions.
            constraints: Textual creative or budget constraints.
            brand_guidelines: Brand assets and visual guidelines.

        Returns:
            An ``AssetProductionReport`` with technical shot designs.

        Raises:
            Exception: If production planning or asset mapping fails.
        """
        prompt = f"""
        Role: Senior Creative Producer & Production Designer.
        Objective: Convert approved ad scripts into a technical "Production Blueprint"
        that specifies every asset, shot, and technical requirement needed to build the
        final creative.

        INPUTS:
        - Final Approved Ad Scripts: {json.dumps(self._to_json_dict(variants))}
        - Platform-Adapted Scripts: {json.dumps(self._to_json_dict(platform_variants))}
        - Creative & Budget Constraints: {constraints}
        - Brand Visual Guidelines: {brand_guidelines}

        EXECUTION STEPS:
        1. ASSET INVENTORY (BOM): Define exactly what is needed specialized by type:
           - Visuals: (e.g., 4k Video clips, PNG Overlays, 3D Renders).
           - Audio: (e.g., High-energy background track, AI Voiceover profile).
           - Copy: (e.g., Dynamic text overlays).
        2. TECHNICAL SHOT DESIGN: For each scene in the script, define the "Technical
           Specifications":
           - Shot Type: (Close-up, Wide, Pan, Zoom).
           - Lighting & Mood: (Cinematic, Bright/High-key, Noir).
           - Motion Graphics: (Callout bubbles, Lower thirds).
        3. PRODUCTION COMPLEXITY AUDIT: Estimate the technical effort (Low/Medium/High)
           and identify potential "Production Bottlenecks."
        4. CONSISTENCY AUDIT: Ensure all assets across all platforms (Meta, TikTok, etc.)
           share the same "Visual DNA."

        CONSTRAINTS & RULES:
        - TECHNICAL PRECISION: All visual specs must include aspect ratios (e.g., 9:16)
          and recommended durations.
        - GROUNDED DESIGN: Every asset must directly support a specific "Message Pillar"
          from the strategy.
        - OUTPUT DISCIPLINE: Return results as a structured AssetProductionReport JSON object.
        """
        return self.generate(prompt, response_schema=AssetProductionReport)
