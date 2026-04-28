"""Deployment Agent module."""

from __future__ import annotations

import json
import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    DeploymentExperimentationReport,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class DeploymentExperimentationAgent(BaseAgent):
    """Plans the launch of validated ad variants and organizes experiments.

    This agent defines tactical rollout plans, including bidding
    strategies, budget allocation, and A/B test setups for the final
    approved creative assets.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the DeploymentExperimentationAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def plan_deployment(
        self,
        variants: list[AdScript],
        brief: StrategyBrief,
        platforms: list[str],
        budget: str = "TBD",
        timeline: str = "TBD",
        geography_market: str = "",
    ) -> DeploymentExperimentationReport:
        """Plans the launch of validated ad variants and defines experiments.

        Args:
            variants: List of final validated ad variants.
            brief: strategy brief for campaign context.
            platforms: List of target social platforms.
            budget: Total campaign budget allocation.
            timeline: Launch and test duration dates.
            geography_market: The primary market (e.g., "India", "USA").

        Returns:
            A ``DeploymentExperimentationReport`` with tactical launch steps.

        Raises:
            google.api_core.exceptions.InternalServerError: If planning fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "plan_deployment started variant_count=%d platforms=%s",
            len(variants),
            platforms,
        )
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Senior Media Buyer & Experimentation Architect.
        Objective: Design a tactical deployment and A/B testing plan for validated ad variants
        to maximize ROAS and collect statistically significant performance data.

        INPUTS:
        - Validated Ad Variants: {json.dumps(self.to_json_dict(variants))}
        - Strategic Strategy Brief: {brief.model_dump_json()}
        - Target Platforms: {json.dumps(self.to_json_dict(platforms))}
        - Total Allocated Budget: {budget}
        - Campaign Timeline: {timeline}
        - Target Marketplace: {geography_market}

        EXECUTION STEPS:
        1. MEDIA BUYING STRATEGY: Define the bidding model (e.g., Lowest Cost, Bid Cap) and
           budget split between "Test Phase" and "Scaling Phase."
        2. EXPERIMENT DESIGN: Design a 48-hour A/B/n test. Specify the "Primary Variable"
           (e.g., Hook, CTA, or Thumbnail).
        3. PIXEL & TRACKING SETUP: Define the custom conversion events and data points
           required to measure the Strategy KPIs.
        4. SCALING PROTOCOL: Define explicit "Performance Triggers"—at what CPA or CTR
           threshold should the budget be increased by 20%?
        5. RISK MITIGATION: Define "Burn Rates" and "Early Stop" rules if performance falls
           below a critical floor.

        CONSTRAINTS & RULES:
        - DATA-DRIVEN: All scaling triggers must be mathematically derived from the
          "Budget Inference" and "Strategy KPIs."
        - PLATFORM-SPECIFIC: Use native terminology for each platform (e.g., "Meta Advantage+"
          vs. "TikTok Spark Ads").
        - OUTPUT DISCIPLINE: Return results as a structured DeploymentExperimentationReport
          JSON object.
        """
            report = self.generate(prompt, response_schema=DeploymentExperimentationReport)
            elapsed = time.perf_counter() - start
            logger.info(
                "plan_deployment completed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "plan_deployment failed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            raise
