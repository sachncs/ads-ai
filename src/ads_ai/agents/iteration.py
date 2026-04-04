"""Iteration Agent module."""

from __future__ import annotations

import json

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AdScript, IterationControlReport


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

    def manage_iteration(self,
                         variants: list[AdScript],
                         strategy: StrategyBrief,
                         evaluations: list[dict[str, Any]],
                         readiness_report: CompositeReadinessReport) -> IterationControlReport:
        """Generates specific refinement instructions for each variant.

        Args:
            variants: List of ad scripts currently in the pipeline.
            strategy: The master strategy brief.
            evaluations: List of all evaluation reports.
            readiness_report: The latest readiness scores.

        Returns:
            An ``IterationControlReport`` with refinement directives.

        Raises:
            Exception: If iteration planning or directive generation fails.
        """
        prompt = f"""
        Role: Senior Creative Strategist & Iteration Director.
        Objective: Analyze multi-agent evaluation feedback to generate laser-focused,
        actionable refinement directives that will maximize variant performance in the next
        generation cycle.

        INPUTS:
        - Current Ad Variants: {json.dumps(self._to_json_dict(variants))}
        - Strategic Brief: {strategy.model_dump_json()}
        - Comprehensive Evaluation Data: {json.dumps(evaluations, default=str)}
        - Readiness Scoring Report: {readiness_report.model_dump_json()}

        EXECUTION STEPS:
        1. DEFECT IDENTIFICATION: Map low scores from evaluative agents (Clarity, Brand,
           Attention) to specific lines, scenes, or cues in the ad scripts.
        2. ROOT CAUSE ANALYSIS: Determine *why* a score is low (e.g., "Brand first appears
           at 0:08, causing high drop-off risk").
        3. PRIORITIZED DIRECTIVES: Generate exactly 3 high-impact refinement directives
           for each variant.
           - Directives must be EXPLICIT: Use "Swap", "Remove", "Move", or "Rewrite [X] to [Y]."
        4. OUTCOME PROJECTION: For each directive, define the "Expected Outcome" (e.g.,
           "Increase Attention score by ~15 points").
        5. SYSTEM REFINEMENT STRATEGY: Provide a one-paragraph global summary of how the
           creative approach must shift (e.g., "Shift from emotional storytelling to
           feature-first kinetic text").

        CONSTRAINTS & RULES:
        - NO VAGUE DIRECTIVES: Avoid "Improve the hook" or "Make it clearer." Use "Rewrite
          hook to include a shocking statistic from the URL data."
        - FEASIBILITY: Ensure every directive is technically executable by the Creative agent.
        - OUTPUT DISCIPLINE: Return results as a structured IterationControlReport JSON object.
        """
        return self.generate(prompt, response_schema=IterationControlReport)
