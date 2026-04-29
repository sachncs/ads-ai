"""Iteration Agent module."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    CompositeReadinessReport,
    IterationControlReport,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class IterationControllerAgent(BaseAgent):
    """Refines ad variants based on evaluation feedback.

    This agent analyzes the multi-agent evaluation reports to generate
    precise, actionable directives for refining ad concepts and
    scripts.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the IterationControllerAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def manage_iteration(
        self,
        variants: list[AdScript],
        strategy: StrategyBrief,
        evaluations: list[dict[str, Any]],
        readiness_report: CompositeReadinessReport,
    ) -> IterationControlReport:
        """Generates specific refinement instructions for each variant.

        Args:
            variants: List of ad scripts currently in the pipeline.
            strategy: The master strategy brief.
            evaluations: List of all evaluation reports.
            readiness_report: The latest readiness scores.

        Returns:
            An ``IterationControlReport`` with refinement directives.

        Raises:
            google.api_core.exceptions.InternalServerError: If planning fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "manage_iteration started variant_count=%d",
            len(variants),
        )
        start = time.perf_counter()
        try:
            cot_reasoning = """
        REASONING TRACE:
        1. FAILURE MODE CLASSIFICATION: For each variant's failing dimensions:
           → BRAND FAILURE: Brand appears too late, confusion about what brand
             → Fix: Move brand intro to scene 1-2, add visual brand anchor
           → CLARITY FAILURE: Message doesn't land, what-is-it confusion
             → Fix: Lead with problem statement, simplify language
           → ATTENTION FAILURE: Hook doesn't interrupt, passive viewing
             → Fix: Stronger pattern interrupt in first 0.5s, add visual surprise
           → INTENT FAILURE: Wrong emotional lever pulled, no desire triggered
             → Fix: Change emotional frame to match persona motivation
           → DIAGNOSTICS FAILURE: Pacing issues, logical gaps, boring middle
             → Fix: Restructure scene order, cut/redesign drag sections

        2. PRIORITY ORDERING: For each failing variant, rank fixes by:
           - Impact: How much will this improve the composite score?
           - Feasibility: Can the creative agent actually implement this?
           - Risk: Will this fix introduce new problems?

        3. SPECIFICITY CHECK: Each directive must answer:
           - WHO: Which agent's score is failing?
           - WHAT: Which specific element (hook, scene 2 dialogue, CTA timing)?
           - HOW: Exact change to make (not "improve", but "replace X with Y")
           - METRIC: Expected score improvement (e.g., "+15 points Brand Linkage")
        """

            few_shot_directives = """
        EXAMPLE ITERATION DIRECTIVES:
        {
          "iteration_directives": [
            {
              "category": "Brand Linkage",
              "severity": "critical",
              "instruction": "MOVE brand introduction from Scene 3 (0:06) to Scene 1 (0:01). "
                           "Add product logo as watermark starting at 0:01.",
              "expected_outcome": "Brand Linkage score increase from 52 to 70+"
            },
            {
              "category": "Attention",
              "severity": "high",
              "instruction": "REPLACE opening scene with a counter-intuitive statistic that "
                           "contradicts the user's assumed reality. Keep duration at 0:02.",
              "expected_outcome": "Scroll-stop probability increase from 35% to 55%+"
            }
          ],
          "prioritized_issues": [
            {
              "issue": "Brand first appears at 0:06 - 3 seconds too late for recall",
              "source_agent": "BrandLinkageAgent",
              "impact": "Primary brand recall will be below target threshold"
            }
          ],
          "suggested_actions": [
            {
              "action": "PREPEND brand watermark to Scene 1",
              "target_scene": "Scene 1",
              "rationale": "Visual brand anchor before dialogue begins"
            },
            {
              "action": "ADD product name to first dialogue line",
              "target_scene": "Scene 1",
              "rationale": "Audio brand introduction within first 0.5 seconds"
            }
          ]
        }
        """

            prompt = f"""
{cot_reasoning}

        Role: Senior Creative Strategist & Iteration Director.
        Objective: Analyze multi-agent evaluation feedback to generate laser-focused,
        actionable refinement directives. You are the bridge between AI evaluation and
        human-level creative improvement — your directives must be precise enough for
        the Creative Agent to execute without ambiguity.

        INPUTS:
        - Current Ad Variants: {json.dumps(self.to_json_dict(variants))}
        - Strategic Brief: {strategy.model_dump_json()}
        - Comprehensive Evaluation Data: {json.dumps(evaluations, default=str)}
        - Readiness Scoring Report: {readiness_report.model_dump_json()}

        EXECUTION STEPS:

        1. DEFECT MAPPING (for each failing variant):
           For each dimension that scored below threshold:
           - Identify the SPECIFIC LINE/SCENE in the script that caused the failure
           - Map it to the source agent (Clarity, Brand, Attention, Diagnostics, Intent)
           - Determine the ROOT CAUSE (not just symptom)

        2. DIRECTIVE GENERATION:
           For each failing dimension, generate 2-3 directives that:
           - Use PRESCRIPTIVE language: "REPLACE", "MOVE", "ADD", "REMOVE", "REWRITE"
           - Reference exact locations: "Scene 2", "hook", "CTA", "line 3"
           - Define success criteria: "Expected outcome: [specific metric] +[X] points"

        3. CONSTRAINT PRESERVATION:
           Identify which parts of the script are WORKING and must be preserved:
           - List specific elements that scored above threshold
           - Directives must NOT modify these without explicit justification

        4. ACTION HIERARCHY:
           For each variant, order actions by:
           1. Critical fixes (score < 40 on any dimension)
           2. High-impact improvements (score 40-60)
           3. Optimization refinements (score 60-70)

        5. FEASIBILITY CHECK:
           For each directive, verify:
           - Creative Agent has the capability to execute (script editing, not production)
           - Does not require new footage/assets not in original brief
           - Does not contradict platform constraints

        FEW-SHOT EXAMPLE:
        {few_shot_directives}

        OUTPUT VALIDATION CHECKS:
        □ Each failing variant has at least 2 directives
        □ Directives use PRESCRIPTIVE language (no "consider", "might", "could")
        □ Each directive has expected_outcome with specific metric
        □ No directive modifies an element that was already passing
        □ Global refinement strategy addresses the ROOT cause pattern across variants

        FORMAT: Return ONLY valid JSON matching IterationControlReport schema.
        No explanation, no preamble. JSON only.
        """
            report = self.generate(prompt,
                                   response_schema=IterationControlReport)
            elapsed = time.perf_counter() - start
            logger.info(
                "manage_iteration completed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "manage_iteration failed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            raise
