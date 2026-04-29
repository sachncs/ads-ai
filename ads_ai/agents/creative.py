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
            cot_reasoning = """
        REASONING TRACE:
        1. ANALYZE PERSONA PRIORITIES: Which persona is the PRIMARY conversion target?
           → The one with highest "conversion_intent" in IntentSimulation.
           → All hooks, CTAs, and messaging must speak to THEIR psychology first.
        2. DETERMINE HOOK TYPE: Based on strategy goal and audience:
           → Awareness: Pattern interrupt (visual shock, stat, question)
           → Conversion: Value proposition + proof snippet
        3. STRUCTURAL PATTERN: Select the narrative arc:
           → "Problem/Solution": 0:01-0:03 pain amplification → 0:04-0:08 solution reveal
           → "Direct-to-Camera": 0:01-0:02 authority establishing → 0:03-0:08 proof + CTA
           → "Story-driven": 0:01-0:05 mini-narrative → 0:06-0:08 brand + CTA
           → "Kinetic-Typographic": 0:01-0:03 text animation → 0:04-0:08 visual payoff
        4. PLATFORM CALIBRATION:
           → TikTok: Hook in 0-0.5s, peak moment at 0:02-0:04, CTA by 0:06
           → Meta Feed: Hook at 0:01-0:03, narrative peak at 0:05-0:12, CTA at 0:13-0:15
           → LinkedIn: Hook at 0:01-0:05, credibility beat at 0:06-0:10, CTA at 0:11-0:15
        5. VEO PROMPT SYNTHESIS: The video_prompt field must:
           → Specify lighting mood (e.g., "warm golden hour natural light")
           → Specify camera movement (e.g., "slow push-in, 35mm shallow DOF")
           → Specify visual style (e.g., "clean minimal product photography aesthetic")
           → Include the brand color/asset if available
           → NOT include dialogue - Veo handles VO separately
        """

            few_shot_scenes = """
        EXAMPLE SCRIPT (TikTok - Problem/Solution, 0:08 duration):
        {
          "concept_title": "The [Pain] Interruption",
          "core_idea": "Users discover [product] solves [specific pain] they've accepted as normal",
          "hook": "POV: You're staring at [specific frustration] for the 3rd time this week.",
          "script_scenes": [
            {
              "description": (
                  "Screen recording style - user visibly frustrated at [pain point]"
              ),
              "visual_cues": (
                  "Close-up on [device], notification badges piling up. "
                  "Muted colors, overhead fluorescent lighting."
              ),
              "dialogue_vo": "This is your 47th tab. You've lost count."
            },
            {
              "description": "Cut to clean product reveal - subtle slow motion",
              "visual_cues": (
                  "White background, product enters frame from bottom. "
                  "Key light from left. Brand color accent visible."
              ),
              "dialogue_vo": "Until you try [product]."
            },
            {
              "description": (
                  "Product in use - problem solved, satisfying outcome"
              ),
              "visual_cues": (
                  "Product screen / usage shot. Clean desk, calm user. "
                  "Golden hour light now replaced with warm interior."
              ),
              "dialogue_vo": "[Key benefit]. Without the [former pain]. Starting at [price]."
            }
          ],
          "brand_integration": (
              "Product name and logo visible at 0:06 and 0:08. "
              "Brand color palette used throughout color grading."
          ),
          "cta": "Try free for 14 days. Link in bio.",
          "video_prompt": (
              "Cinematic product reveal, clean white background transition "
              "to warm interior, slow-motion device placement, 35mm lens "
              "shallow depth of field, natural lighting with key light from "
              "left, brand accent colors in every frame, TikTok vertical 9:16"
          )
        }
        """

            prompt = f"""
{cot_reasoning}

        Role: Lead Creative Director & Narrative Architect.
        Objective: Produce high-impact, platform-native ad scripts that strictly adhere
        to the strategy brief and persona psychology. Every creative decision MUST trace
        back to either the Strategy Brief or a Persona Profile.

        INPUTS:
        - Product: {product}
        - Strategy Brief: {strategy.model_dump_json()}
        - Audience Personas: {personas.model_dump_json()}
        - Target Platform(s): {", ".join(platforms) if platforms else "TBD - infer from audience"}
        - Creative Constraints: {constraints or "None specified"}

        VARIANT GENERATION RULES:
        - Generate 3-5 distinct creative concepts (one per "structural pattern" in reasoning)
        - Each variant MUST have a different emotional angle or narrative structure
        - NO two variants should feel interchangeable
        - Each variant MUST be traceable to a specific message pillar in the strategy

        SCRIPT REQUIREMENTS (MANDATORY FOR EACH VARIANT):
        1. CONCEPT TITLE: Descriptive name capturing the creative angle
        2. CORE IDEA: One sentence describing the through-line from hook to CTA
        3. HOOK (0:01-0:03):
           - Must stop the scroll within first 0.5 seconds
           - Must reference a specific pain point OR aspirational state from persona data
           - Must be platform-native in format
        4. SCRIPT SCENES (3-5 scenes):
           - Each scene: description, visual_cues (camera, lighting, action), dialogue/VO
           - Scene progression must follow natural viewing psychology
           - Visual cues must be specific enough for a DP to shoot (not "good lighting")
        5. BRAND INTEGRATION:
           - Brand must appear by scene 2 (0:04-0:06) at the latest
           - Logo/name in frame for minimum 2 seconds total
        6. CTA:
           - Must align with the psychological lever defined in strategy
           - Must be specific: "Try free for 14 days" not "Click here"
           - Must appear at 80%+ mark of video duration
        7. VIDEO PROMPT (for Veo 3.1):
           - Camera movement and lens specification
           - Lighting mood and color grading direction
           - Setting/location description
           - Visual style (documentary, commercial, kinetic, etc.)
           - MUST be different from actual dialogue - prompt describes visual only

        FEW-SHOT EXAMPLE:
        {few_shot_scenes}

        BRAND SAFETY GUARDRAILS:
        - NEVER show before/after transformations for health/appearance products without disclaimer
        - NEVER make comparative claims against named competitors
        - NEVER use urgency tactics that imply limited supply that doesn't exist
        - All claims must be supportable by product documentation

        OUTPUT VALIDATION (INTERNAL CHECK BEFORE RESPONDING):
        □ Hook appears within first 0.03 seconds of viewing
        □ Brand appears by scene 2 (no later than 0:06)
        □ CTA appears at 80%+ mark
        □ Each variant targets different persona OR different message pillar
        □ Video prompt describes VISUALS only (no dialogue/VO text)
        □ No generic phrases ("revolutionary", "game-changing", "cutting-edge")
        □ All claims traceable to product description or strategy brief

        FORMAT: Return ONLY valid JSON as CreativeVariants schema.
        No markdown, no commentary, no preamble. JSON only.
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
