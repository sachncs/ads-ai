"""Brand Linkage Agent module."""

from __future__ import annotations

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AdScript, AudienceSegments, BrandLinkageEvaluation, StrategyBrief


class BrandLinkageAgent(BaseAgent):
    """Evaluates brand recall and attribution strength.

    This agent analyzes ad scripts to ensure the brand identity is
    integrated enough to be correctly attributed by the target audience.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the BrandLinkageAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-flash-lite-preview")

    def evaluate(self,
                 script: AdScript,
                 strategy: StrategyBrief,
                 brand_guidelines: str,
                 personas: AudienceSegments) -> BrandLinkageEvaluation:
        """Evaluates whether an ad is clearly associated with the brand.

        Args:
            script: Ad variant script to evaluate.
            strategy: Strategy document for positioning alignment.
            brand_guidelines: Textual brand asset and voice guidelines.
            personas: Audience personas for recall simulation.

        Returns:
            A ``BrandLinkageEvaluation`` instance with recall metrics.

        Raises:
            Exception: If brand linkage evaluation or simulation fails.
        """
        prompt = f"""
        Role: Senior Brand Strategist & Identity Auditor.
        Objective: Rigorously evaluate the strength of brand linkage and attribution
        within an ad script to ensure the viewer correctly identifies the source and message.

        INPUTS:
        - Ad Variant Script: {script.model_dump_json()}
        - Brand Strategy: {strategy.model_dump_json()}
        - Detailed Brand Guidelines: {brand_guidelines}
        - Target Audience Personas: {personas.model_dump_json()}

        EXECUTION STEPS:
        1. ATTRIBUTION AUDIT: Identify every mention (verbal) or cue (visual) of the brand.
           Record the timestamp/scene of the first occurrence.
        2. DISTINCTIVENESS CHECK: Evaluate the use of "Distinctive Brand Assets" (colors,
           logos, slogans, fonts, audio signatures). Do they match the guidelines?
        3. BRAND-STORY INTEGRATION: Is the brand a "hero" in the story, or an "afterthought"
           at the end?
        4. UNAIDED RECALL SIMULATION: For each persona, simulate their memory 10 seconds
           after viewing. If they are asked "Who was this ad for?", what is their likely
           answer and confidence (0-100)?
        5. SCORING:
           - Attribution Clarity: Is it clear WHO the ad is for within the first 3 seconds?
           - Integration Strength: Is the brand necessary for the ad's logic?
        6. COMPETITIVE CONFUSION: Could this ad be easily mistaken for a competitor's ad
           if the logo were changed?

        CONSTRAINTS & RULES:
        - LATE BRANDING PENALTY: If the brand first appears after 5 seconds, the Linkage
          Score cannot exceed 50.
        - GUIDELINE ADHERENCE: Any deviation from tone or visual cues in the guidelines
          must be flagged as a "Risk."
        - OUTPUT DISCIPLINE: Return results as a structured BrandLinkageEvaluation JSON object.
        """
        return self.generate(prompt, response_schema=BrandLinkageEvaluation)
