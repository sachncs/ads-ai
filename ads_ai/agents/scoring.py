"""Scoring Agent module."""

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
    IntentEvaluation,
    StrategyBrief,
)

logger = logging.getLogger(__name__)


class ScoringAgent(BaseAgent):
    """Aggregates and weights multi-agent evaluation scores.

    This agent synthesizes the outputs from clarity, brand,
    diagnostics, attention, and intent agents into a final readiness
    score for each ad variant.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the ScoringAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def aggregate_and_score(
        self,
        variants: list[AdScript],
        strategy: StrategyBrief,
        evaluations: list[dict[str, Any]],
        intent_evaluation: IntentEvaluation,
    ) -> CompositeReadinessReport:
        """Aggregates scores from all evaluation agents.

        Args:
            variants: List of ad scripts evaluated.
            strategy: The master strategy brief.
            evaluations: List of multi-agent evaluation results.
            intent_evaluation: Behavioral intent simulation results.

        Returns:
            A ``CompositeReadinessReport`` with final scores and decisions.

        Raises:
            google.api_core.exceptions.InternalServerError: If aggregation fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "aggregate_and_score started variant_count=%d",
            len(variants),
        )
        start = time.perf_counter()
        try:
            cot_reasoning = """
        REASONING TRACE:
        1. CRITICAL FAILURE SCAN: First pass - check for ANY "Red Flag" or "Fail" in:
           → BrandLinkageEvaluation.confusion_risk == "High"
           → DiagnosticsEvaluation.overall_diagnostic_score < 40
           → ClarityEvaluation.clarity_score < 50
           If any found: variant CANNOT exceed 25 regardless of other scores.

        2. CONTEXTUAL NORMALIZATION: Adjust scoring based on strategy goal type:
           → "Funny/Humorous" goal: Do NOT penalize low Brand Linkage if brand is
             integrated via in-group references or self-deprecating humor.
             Weight: Conversion Intent (35%), Attention (25%), Clarity (20%),
             Brand (10%), Diagnostics (10%)
           → "Awareness" goal: Weight: Attention (35%), Brand (25%), Clarity (20%),
             Diagnostics (15%), Intent (5%)
           → "Conversion" goal: Weight: Clarity (30%), Intent (30%), Brand (20%),
             Attention (15%), Diagnostics (5%)

        3. TRADE-OFF DETECTION: Explicitly identify when a high Attention score
           came at the cost of Brand clarity, or when a high Clarity score
           sacrificed novelty. Document WHY the trade-off was made.

        4. COMPOSITE SCORE CALCULATION:
           composite = (clarity * w_clarity) + (brand * w_brand) +
                       (attention * w_attention) + (intent * w_intent) +
                       (diagnostics * w_diagnostics)
           where w_* weights are from strategy.pre_release_targets.decision_thresholds

        5. DECISION LOGIC:
           GO: final_score >= threshold AND no critical failures
           CONDITIONAL GO: final_score >= threshold - 10 but has non-critical issues
           NO-GO: final_score < threshold OR critical failure detected
        """

            few_shot_decision = """
        EXAMPLE VARIANT DECISION:
        {
          "concept_title": "The Efficiency Play",
          "final_readiness_score": 78.5,
          "is_ready": true,
          "status": "GO",
          "confidence": 0.82,
          "primary_strength": (
              "Highest Intent score (88) driven by pain-point-to-solution clarity"
          ),
          "primary_blocker": null,
          "category_breakdown": [
            {
              "category": "Clarity",
              "raw_score": 82,
              "weight": 0.25,
              "weighted_contribution": 20.5
            },
            {
              "category": "Brand Linkage",
              "raw_score": 75,
              "weight": 0.20,
              "weighted_contribution": 15.0
            },
            {
              "category": "Attention",
              "raw_score": 79,
              "weight": 0.20,
              "weighted_contribution": 15.8
            },
            {
              "category": "Intent",
              "raw_score": 88,
              "weight": 0.25,
              "weighted_contribution": 22.0
            },
            {
              "category": "Diagnostics",
              "raw_score": 71,
              "weight": 0.10,
              "weighted_contribution": 7.1
            }
          ]
        }

        EXAMPLE TRADE-OFF:
        {
          "dimension_a": "Attention (82)",
          "dimension_b": "Brand Linkage (65)",
          "winning_dimension": "Attention",
          "justification": (
              "High attention score driven by pattern-interrupt hook. "
              "Low brand linkage is acceptable trade-off for awareness goal "
              "because brand will be reinforced in follow-on retargeting ads."
          )
        }
        """

            prompt = f"""
{cot_reasoning}

        Role: Lead Data Scientist & Strategic Readiness Officer.
        Objective: Synthesize multi-agent evaluation data into a single, high-fidelity
        Readiness Report that dictates the final GO/NO-GO decision for each ad variant.
        Your output directly controls what enters production — be precise and defensible.

        INPUTS:
        - Ad Variants: {json.dumps(self.to_json_dict(variants))}
        - Strategy Brief: {strategy.model_dump_json()}
        - Multi-Agent Evaluative Data: {json.dumps(evaluations, default=str)}
        - Behavioral Intent: {intent_evaluation.model_dump_json()}

        EXECUTION STEPS:

        1. CRITICAL FAILURE CHECK (FIRST):
           Scan ALL evaluation data for:
           - brand_linkage_score < 40 (unrecoverable brand confusion)
           - clarity_score < 45 (message incomprehensible)
           - confusion_risk == "High" AND alignment_score < 50
           - Any agent explicitly returning "Fail" or "Red Flag"
           If found: set final_readiness_score = max(score, 25), set primary_blocker,
           set is_ready = false.

        2. CONTEXTUAL WEIGHTING:
           Read strategy.ad_intent to determine goal type.
           Apply goal-specific weight overrides (see reasoning trace above).
           Verify that sum of weights = 1.0. Adjust proportionally if not.

        3. WEIGHTED AGGREGATION:
           Calculate Composite Readiness Score (0-100) for each variant using:
           composite = Σ(category_score × category_weight)
           Round to 1 decimal place.

        4. TRADE-OFF ANALYSIS:
           For each variant, identify the highest-scoring dimension and lowest.
           If gap > 20 points, document as a TradeOff entry.
           If the trade-off aligns with strategy goal, explain WHY it is acceptable.
           If it misaligns, flag as risk.

        5. BEST-IN-CLASS SELECTION:
           Rank variants by final_readiness_score descending.
           Ties within 2 points: break by aggregated_intent_score (highest wins).
           Identify best_overall_variant.

        6. SYSTEM READINESS FLAG:
           Set system_readiness_flag = true ONLY if:
           - At least 1 variant has is_ready = true
           - No critical risks affect the best variant

        7. CRITICAL RISKS:
           Aggregate all "recommended_fixes" across all evaluation agents.
           Deduplicate. If any fix appears in >50% of agents, flag as critical risk.

        FEW-SHOT EXAMPLES:
        {few_shot_decision}

        OUTPUT VALIDATION CHECKS:
        □ All variant_decisions have unique concept_title
        □ Sum of all weights in category_breakdown = 1.0 (tolerance ±0.01)
        □ final_readiness_score matches sum of weighted_contributions
        □ best_overall_variant exists in variant_decisions
        □ Any variant with critical failure has status = "NO-GO"

        FORMAT: Return ONLY valid JSON matching CompositeReadinessReport schema.
        No explanation, no preamble. JSON only.
        """
            report = self.generate(prompt,
                                   response_schema=CompositeReadinessReport)
            elapsed = time.perf_counter() - start
            logger.info(
                "aggregate_and_score completed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "aggregate_and_score failed variant_count=%d elapsed=%.3fs",
                len(variants),
                elapsed,
            )
            raise
