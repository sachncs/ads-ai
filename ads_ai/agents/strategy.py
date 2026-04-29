"""Strategy Agent module."""

from __future__ import annotations

import logging

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import StrategyBrief
from ads_ai.utils import agent_timing, build_strategy_prompt

logger = logging.getLogger(__name__)


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

    def create_brief(
        self,
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
        geography_market: str,
    ) -> StrategyBrief:
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
            google.api_core.exceptions.InternalServerError: If synthesis fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "create_brief started product=%s goal=%s platforms=%s",
            product[:50],
            goal,
            platforms,
        )
        with agent_timing(
            logger,
            "create_brief",
            "product",
            product[:50],
            "goal",
            goal,
            "platforms",
            platforms,
        ) as meta:
            prompt = build_strategy_prompt(
                product=product,
                goal=goal,
                audience_desc=audience_desc,
                platforms=platforms,
                budget=budget,
                timeline=timeline,
                brand_assets=brand_assets,
                key_differentiators=key_differentiators,
                competitors=competitors,
                constraints=constraints,
                geography_market=geography_market,
            )
            meta.append(("elapsed", "pending"))
            report = self.generate(prompt, response_schema=StrategyBrief)

        return report
