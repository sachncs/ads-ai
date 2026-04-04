"""Scoring Agent module."""

from __future__ import annotations

import json

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AdScript, CompositeReadinessReport


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

    def aggregate_and_score(self,
                            variants: list[AdScript],
                            strategy: StrategyBrief,
                            evaluations: list[dict[str, Any]],
                            intent_evaluation: IntentEvaluation) -> CompositeReadinessReport:
        """Aggregates scores from all evaluation agents.

        Args:
            variants: List of ad scripts evaluated.
            strategy: The master strategy brief.
            evaluations: List of multi-agent evaluation results.
            intent_evaluation: Behavioral intent simulation results.

        Returns:
            A ``CompositeReadinessReport`` with final scores and decisions.

        Raises:
            Exception: If score aggregation or readiness weighting fails.
        """
        prompt = f"""
        Role: Lead Data Scientist & Strategic Readiness Officer.
        Objective: Synthesize multi-agent evaluation data into a single, high-fidelity
        Readiness Report that dictates the final GO/NO-GO decision for each ad variant.

        INPUTS:
        - Ad Variants: {json.dumps(self._to_json_dict(variants))}
        - Strategy Brief: {strategy.model_dump_json()}
        - Multi-Agent Evaluative Data: {json.dumps(evaluations, default=str)}
        - Behavioral Intent: {intent_evaluation.model_dump_json()}

        CONTEXTUAL ADAPTATION:
        - HUMOR & SELF-AWARENESS: If the strategy goal is "Funny" or "Humorous," do not
          penalize low "Brand Linkage" if the brand is integrated subtly or via in-group
          references (e.g., "12th Man Army"). Self-deprecating humor often scores lower on
          traditional "Clarity" but significantly higher on "Shareability" and "Bonding."
        - EVALUATOR WEIGHTING: Prioritize "Conversion Intent" and "Attention" for humor-led
          campaigns over literal message retention.

        EXECUTION STEPS:
        1. CRITICAL FAILURE CHECK: Scan for any "Red Flags" or "Fails" from Compliance or
           Brand agents. If a variant has a fatal flaw, its Readiness Score cannot exceed 20.
        2. WEIGHTED AGGREGATION: Calculate the Composite Readiness Score (0-100) using the
           weights defined in the Strategy Brief.
        3. READINESS TARGET COMPARISON: Compare each variant's scores against the
           "Pre-Release Targets" set by the Strategy agent.
        4. TRADE-OFF ANALYSIS: Explicitly identify where "Attention" was sacrificed for
           "Brand" or vice-versa. Determine if the trade-off aligns with the campaign goal.
        5. BEST-IN-CLASS SELECTION: Identify the single highest-performing variant. If scores
           are tied within 2 points, select the one with the highest "Conversion Intent."
        6. FINAL RECOMMENDATION: Provide a definitive "Ready for Production" flag (True/False).

        CONSTRAINTS & RULES:
        - NO AVERAGING FATAL FLAWS: A high Attention score cannot "save" a variant that fails
          Brand Linkage.
        - DATA INTEGRITY: Use only provided agent scores. Do not inject external opinions.
        - OUTPUT DISCIPLINE: Return results as a structured CompositeReadinessReport JSON object.
        """
        return self.generate(prompt, response_schema=CompositeReadinessReport)
