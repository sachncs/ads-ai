"""Strategy Agent module."""

from __future__ import annotations

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import BudgetInferenceReport, ExtractedInputs, StrategyBrief


class StrategyAgent(BaseAgent):
    """Generates the master advertising strategy document.

    This agent synthesizes product intelligence and budget constraints
    into a cohesive messaging framework, defining KPIs and audience
    personas.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the StrategyAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def create_brief(self,
                     product: str,
                     goal: str,
                     audience_desc: str,
                     platforms: list[str],
                     budget: str,
                     timeline: str,
                     brand_assets: str,
                     key_differentiators: str,
                     competitors: str,
                     constraints: str,
                     geography_market: str) -> StrategyBrief:
        """Synthesizes a structured StrategyBrief.

        Args:
            product: Product description and value prop.
            goal: Primary business objective.
            audience_desc: Target audience context.
            platforms: Target advertising platforms.
            budget: Budget constraints.
            timeline: Campaign timeline.
            brand_assets: Visual/tonal assets like logos or colors.
            key_differentiators: Competitive advantages.
            competitors: Competing brands/categories.
            constraints: Compliance or creative rules.
            geography_market: Geographic targeting.

        Returns:
            A ``StrategyBrief`` instance.

        Raises:
            Exception: If the strategy synthesis or generation fails.
        """
        prompt = f"""
        Role: Lead Growth Strategist & Brand Architect.
        Objective: Synthesize product intelligence and budget constraints into a
        multi-channel advertising strategy that maximizes conversion and brand equity.

        INPUTS:
        - Product Brand/Name: {product}
        - Primary Campaign Goal: {goal}
        - Target Audience: {audience_desc}
        - Platforms: {', '.join(platforms)}
        - Budget: {budget}
        - Timeline: {timeline}
        - Brand Assets: {brand_assets}
        - Key Differentiators: {key_differentiators}
        - Competitors: {competitors}
        - Constraints: {constraints}
        - Market/Geography: {geography_market}

        EXECUTION STEPS:
        1. CHALLENGE DEFINITION: Define the "Adversarial Market Challenge" (e.g.,
           "High switching cost" or "Low category awareness").
        2. VALUE PROPOSITION REFINEMENT: Distill the product features into a
           "Single Consumer Benefit" (SCB).
        3. MESSAGE PILLAR DESIGN: Design 3 distinct message pillars (e.g.,
           Efficiency, Status, Security).
        4. CROSS-CHANNEL TACTICS: Define how the message shifts between platforms
           (Meta vs. TikTok vs. YouTube).
        5. SUCCESS METRICS (KPIs): Define primary and secondary KPIs with explicit
           readiness targets (e.g., "Clarity score > 85 required for launch").
        6. URGENCY & CTA STRATEGY: Define the psychological lever for the CTA (e.g.,
           Loss Aversion, Scarcity, Exclusive Access).
        7. SYSTEM ALIGNMENT: Define weighted weights (totalling 1.0) for the
           evaluative agents.

        CONSTRAINTS & RULES:
        - STRATEGIC DEPTH: Avoid generic goals like "Increase sales." Use specific
          directives like "Dethrone the incumbent by highlighting [X] flaw."
        - FEASIBILITY: Strategy must be executable within provided budget and
          media plan.
        - NO DRIFT: Respect constraints and funnel types in the narrative.
        - OUTPUT DISCIPLINE: Return results as a structured StrategyBrief JSON object.
        """
        return self.generate(prompt, response_schema=StrategyBrief)
