"""Audience Agent module."""

from __future__ import annotations

import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AudienceSegments, StrategyBrief

logger = logging.getLogger(__name__)


class AudienceAgent(BaseAgent):
    """Models audience personas and simulates ad response.

    This agent transforms high-level strategy into detailed behavioral
    personas and predicts their initial reactions to common ad hooks.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the AudienceAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def model_personas(
        self,
        product: str,
        strategy: StrategyBrief,
        target_audience: str,
        platforms: list[str],
        market_context: str = "",
    ) -> AudienceSegments:
        """Transforms a high-level target audience into structured personas.

        Args:
            product: Product description.
            strategy: Strategy document.
            target_audience: Target audience description.
            platforms: List of platforms.
            market_context: Optional market context.

        Returns:
            An ``AudienceSegments`` instance containing detailed personas.

        Raises:
            google.api_core.exceptions.InternalServerError: If modeling fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "model_personas started target_audience=%s platforms=%s",
            target_audience[:50],
            platforms,
        )
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Senior Behavioral Psychologist & Market Researcher.
        Objective: Synthesize product data into 3 high-fidelity audience personas based
        on deep psychological drivers and behavioral economics.

        INPUTS:
        - Product Intent & URL Data: {product}
        - Core Brand Strategy: {strategy.model_dump_json()}
        - Global Market Trends: {market_context}

        EXECUTION STEPS:
        1. PSYCHOGRAPHIC SEGMENTATION: Identify 3 distinct groups (e.g., The Skeptics,
           The Early Adopters, The Value Seekers).
        2. BARRIER & TRANSFORMATION MAPPING: For each persona, identify:
           - The Current State (Pain point or "Job to be Done").
           - The Desired State (Emotional or functional payoff).
           - The Active Barrier (Why haven't they bought yet?).
        3. BEHAVIORAL TRIGGERS: Define the specific "Internal Trigger" (e.g., Anxiety,
           Social Pressure, Curiosity) that makes this persona receptive to an ad.
        4. CHANNEL AFFINITY: Map each persona to their preferred platform and content
           consumption style (e.g., Long-form educational vs. Short-form kinetic).
        5. DATA GROUNDING: Every persona attribute must be traceable back to the
           "Product Intent" data.

        CONSTRAINTS & RULES:
        - NO CARICATURES: Avoid generic personas like "Tech-savvy Millennial." Use
          behavioral labels like "The Optimization Obsessive."
        - DEPTH: Each persona must include a "Core Question" they are asking when they
          encounter the product.
        - BEHAVIORAL REALISM: Do NOT make all personas "highly likely to buy." At least
          one persona should be a "Hard Sell" with low initial intent.
        - GROUNDED SIMULATION: Completion likelihood must correlate with the defined
          "Pain Point" and "Decision Speed."
        - NO GENERIC TRAITS: Avoid "wants quality" or "is busy." Use specific,
          context-heavy descriptors.
        - OUTPUT DISCIPLINE: Return results as a structured AudienceSegments JSON object.
        """
            report = self.generate(prompt, response_schema=AudienceSegments)
            elapsed = time.perf_counter() - start
            logger.info(
                "model_personas completed target_audience=%s elapsed=%.3fs",
                target_audience[:50],
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "model_personas failed target_audience=%s elapsed=%.3fs",
                target_audience[:50],
                elapsed,
            )
            raise
