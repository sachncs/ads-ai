"""External Validation Agent module."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    ExternalValidationPlan,
    IterationControlReport,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class ExternalValidationAgent(BaseAgent):
    """Designs controlled ad experiments and validation plans.

    This agent uses the final iteration results and strategic KPIs to
    design a real-world testing framework for the approved creative
    variants.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the ExternalValidationAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def design_validation(
        self,
        variants: list[AdScript],
        brief: StrategyBrief,
        evaluations: list[Any],
        iteration_report: IterationControlReport,
    ) -> ExternalValidationPlan:
        """Designs the experimental validation plan.

        Args:
            variants: List of ad scripts to validate.
            brief: Master strategy document for KPI mapping.
            evaluations: List of multi-agent evaluation reports (JSON strings).
            iteration_report: The final iteration control report.

        Returns:
            An ``ExternalValidationPlan`` with experimental setup details.

        Raises:
            google.api_core.exceptions.InternalServerError: If planning fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "design_validation started variant_count=%d",
            len(variants),
        )
        start = time.perf_counter()
        try:
            prompt = f"""
        Role: Lead Quality Assurance Engineer & Brand Safety Officer.
        Objective: Conduct an adversarial, final-stage validation of the ad campaign assets
        and strategy to ensure zero defects, zero hallucinations, and maximum strategic
        alignment before deployment.

        INPUTS:
        - Campaign Strategy & Brief: {json.dumps(self.to_json_dict(brief))}
        - Video Prompt & Production Plan: {json.dumps(self.to_json_dict(variants))}
        - Media & Experiment Plan: {json.dumps(self.to_json_dict(iteration_report))}
        - Pipeline Logs: {json.dumps(self.to_json_dict(evaluations))}

        EXECUTION STEPS:
        1. ADVERSARIAL BRAND AUDIT: Identify any visual or narrative element that contradicts
           the brand identity. Search for "Brand Drift."
        2. HALLUCINATION SCAN: Verify every claim in the video script/prompt against the
           original "URL Intelligence" data. Flag any ungrounded features.
        3. STRATEGIC FIDELITY CHECK: Does the final output actually fulfill the "Primary Goal"
           set at the start of the pipeline?
        4. TECHNICAL FEASIBILITY CHECK: Is the video prompt technically executable by Veo?
           Identify any ambiguous or "Impossible" visual requests.
        5. PRODUCTION READINESS RUBRIC: Provide a definitive Pass/Fail/Edit score for:
           - Brand Safety.
           - Message Clarity.
           - Technical Quality.
           - Strategic Alignment.
        6. FINAL BLOCKER LIST: Identify any "Critical Defect" that must be fixed before the
           campaign goes live.

        CONSTRAINTS & RULES:
        - ZERO TOLERANCE: If a "Critical Hallucination" is found (e.g., a fake price),
          the validation MUST result in a "FAIL."
        - GROUNDEDNESS: Base the audit ONLY on provided logs and inputs.
        - OUTPUT DISCIPLINE: Return results as a structured ExternalValidationReport JSON object.
        """
            report = self.generate(prompt,
                                   response_schema=ExternalValidationPlan)
            elapsed = time.perf_counter() - start
            logger.info(
                "design_validation completed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "design_validation failed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            raise
