"""Compliance Agent module."""

from __future__ import annotations

import json
import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AdScript, ComplianceRiskReport, StrategyBrief

logger = logging.getLogger(__name__)


class ComplianceRiskAgent(BaseAgent):
    """Evaluates ad variants against legal and brand safety policies.

    This agent uses legal and policy-informed heuristics to scan ad
    scripts for risks, copyright issues, or brand safety violations.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the ComplianceRiskAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def validate_compliance(
        self,
        variants: list[AdScript],
        brief: StrategyBrief,
        platforms: list[str],
        brand_guidelines: str = "",
        geography_market: str = "",
        constraints: str = "",
    ) -> ComplianceRiskReport:
        """Evaluates AI-optimized ad variants using policy frameworks.

        Args:
            variants: List of ad scripts to evaluate.
            brief: strategy brief for context.
            platforms: List of target social platforms.
            brand_guidelines: Textual brand asset and voice guidelines.
            geography_market: The primary market (e.g., "India", "USA").
            constraints: Additional campaign or legal constraints.

        Returns:
            A ``ComplianceRiskReport`` with risk classifications.

        Raises:
            google.api_core.exceptions.InternalServerError: If scanning fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "validate_compliance started variant_count=%d platforms=%s",
            len(variants),
            platforms,
        )
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Senior Legal Counsel & Brand Safety Auditor.
        Objective: Conduct a rigorous compliance and policy audit of AI-generated ad
        variants to prevent legal liability, account bans, or brand damage.

        INPUTS:
        - Ad Variants: {json.dumps(self.to_json_dict(variants))}
        - Strategic Strategy: {brief.model_dump_json()}
        - Distribution Platforms: {json.dumps(self.to_json_dict(platforms))}
        - Brand Guidelines: {brand_guidelines}
        - Target Geographies: {geography_market}
        - Additional Legal Constraints: {constraints}

        EXECUTION STEPS:
        1. PROHIBITED CONTENT SCAN: Audit for explicit platform violations:
           - Fake Scarcity (e.g., "Only 2 left" if not true).
           - Unsubstantiated Claims (e.g., "Best in the world").
           - Restricted Categories (Medical, Financial, Housing).
        2. COPYRIGHT & IP AUDIT: Identify any mentions of competitors, celebrities,
           or trademarked terms not owned by the brand.
        3. REGULATORY COMPLIANCE: Apply geography-specific rules (e.g., GDPR/CCPA for
           data privacy messaging, local advertising standards).
        4. BRAND SAFETY CHECK: Identify any "Sensitive Topics" or "Harmful Associations"
           that clash with the brand guidelines.
        5. RISK CLASSIFICATION:
           - HIGH: Immediate rejection (e.g., Policy violation likely to cause account ban).
           - MEDIUM: Needs warning/edit (e.g., Aggressive claim requiring disclaimer).
           - LOW: Minor brand alignment issue.
        6. REMEDIATION DIRECTIVES: Provide the exact text changes required to make the
           ad compliant.

        CONSTRAINTS & RULES:
        - ZERO TOLERANCE: Any "High" risk must result in a "Fail" status for the variant.
        - FACT-CHECKING: Flag any claim that was not present in the original URL
          Intelligence as a "Hallucinated Claim."
        - OUTPUT DISCIPLINE: Return results as a structured ComplianceRiskReport JSON object.
        """
            report = self.generate(prompt, response_schema=ComplianceRiskReport)
            elapsed = time.perf_counter() - start
            logger.info(
                "validate_compliance completed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "validate_compliance failed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            raise
