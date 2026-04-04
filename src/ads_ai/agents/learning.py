"""Knowledge Learning Agent module."""

from __future__ import annotations

import json

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import ExternalValidationPlan, KnowledgeLearningReport


class KnowledgeLearningAgent(BaseAgent):
    """Captures learnings from past campaigns and improves system performance.

    This agent analyzes historical results, validation data, and AI
    outputs to identify high-performing patterns and diagnose agent
    accuracy.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the KnowledgeLearningAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def capture_learnings(self,
                         historical_data: list[str],
                         validation_output: ExternalValidationPlan,
                         ai_outputs: list[str],
                         strategy_docs: list[str]) -> KnowledgeLearningReport:
        """Captures and synthesizes learnings from campaign data.

        Args:
            historical_data: List of historical performance records (as JSON strings).
            validation_output: Final validation plan for the current campaign.
            ai_outputs: List of intermediate AI agent outputs (as JSON strings).
            strategy_docs: List of strategy documents (as JSON strings).

        Returns:
            A ``KnowledgeLearningReport`` with strategic takeaways.

        Raises:
            Exception: If pattern recognition or knowledge synthesis fails.
        """
        prompt = f"""
        Role: Senior Data Strategist & Machine Learning Insights Specialist.
        Objective: Rigorously analyze campaign performance data and agent logs to extract
        high-value, systematic insights that improve the pipeline's future decision-making
        and creative output.

        INPUTS:
        - Campaign Performance Data (Historical): {json.dumps(self._to_json_dict(historical_data))}
        - Pipeline Logs & Evaluations: {json.dumps(self._to_json_dict(ai_outputs))}
        - Validation Plan: {json.dumps(self._to_json_dict(validation_output))}
        - Strategy Documents: {json.dumps(self._to_json_dict(strategy_docs))}

        EXECUTION STEPS:
        1. PATTERN RECOGNITION: Identify statistically significant correlations between
           "Creative Cues" (e.g., Hook type, CTA color, Visual Cues) and "Success Metrics"
           (e.g., CTR, CPA).
        2. FAILURE MODE ANALYSIS: Analyze the lowest-performing variants. Identify the
           "Common Defect" (e.g., "High cognitive load in first 3s").
        3. INSIGHT CATEGORIZATION: Group findings into:
           - Strategic (Targeting/Messaging).
           - Creative (Visual/Narrative).
           - Operational (Budget/Platform).
        4. ACTIONABLE OPTIMIZATION RULES: Generate 3-5 "If-Then" rules (e.g., "If audience
           is 'The Skeptic', then prioritize 'Social Proof' message pillar").
        5. SYSTEMIC RECOMMENDATIONS: Suggest specific architectural or prompt-level changes
           to the pipeline to avoid recurring errors.

        CONSTRAINTS & RULES:
        - DATA GROUNDING: Do NOT generate generic advice like "Be more creative." Every
          insight must be tied to a specific data point from the logs.
        - LONG-TERM FOCUS: Prioritize insights that are transferable across different
          products in the same category.
        - OUTPUT DISCIPLINE: Return results as a structured KnowledgeLearningReport JSON object.
        """
        return self.generate(prompt, response_schema=KnowledgeLearningReport)
