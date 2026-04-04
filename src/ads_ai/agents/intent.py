"""Intent Simulation Agent module."""

from __future__ import annotations

import json

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AdScript, IntentEvaluation


class IntentSimulationAgent(BaseAgent):
    """Simulates post-exposure behavioral intent.

    This agent uses persona psychology and social heuristics to predict
    how different audience segments will likely behave after seeing an
    ad.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the IntentSimulationAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def evaluate(self,
                 variants: list[AdScript],
                 strategy: StrategyBrief,
                 personas: AudienceSegments,
                 evaluations: list[dict[str, Any]]) -> IntentEvaluation:
        """Simulates how different personas will likely behave after exposure.

        Args:
            variants: List of ad scripts to evaluate.
            strategy: Strategy brief object.
            personas: Audience personas object.
            evaluations: Previous agent evaluations for context.

        Returns:
            An ``IntentEvaluation`` instance with intent scores.

        Raises:
            Exception: If intent simulation or behavioral analysis fails.
        """
        prompt = f"""
        Role: Senior Consumer Psychologist & Behavioral Economist.
        Objective: Predict post-exposure behavioral intent by simulating the psychological
        processing of ad variants across diverse audience personas.

        INPUTS:
        - Ad Variants: {json.dumps(self._to_json_dict(variants))}
        - Strategic Brief: {strategy.model_dump_json()}
        - Audience Personas: {personas.model_dump_json()}
        - Agent Evaluations: {json.dumps(evaluations, default=str)}

        EXECUTION STEPS:
        1. COGNITIVE PROCESSING SIMULATION: For each persona/variant, model the "Sequence
           of Thought":
           - Exposure -> Pattern Match (Does this matter to me?) -> Value Recognition ->
             Friction (What's the catch?) -> Intent.
        2. MOTIVATION ALIGNMENT AUDIT: Does the variant mirror the specific "Internal Trigger"
           or "Pain Point" defined in the persona profile?
        3. EVALUATIVE DRIFT: Adjust intent predictions based on the Clarity, Brand, and
           Diagnostic scores provided in the 'Agent Evaluations.'
        4. FRICTION & OBJECTION MAPPING: Identify the "Activation Energy" required for the
           CTA. Is the ask too high for the perceived value?
        5. FUNNEL DROP-OFF PREDICTION: Identify the exact moment in the script where the
           persona is likely to stop "leaning in" and start "leaning out."
        6. BEHAVIORAL SCORING:
           - Engagement Intent (Likelihood to like/comment/save).
           - Action Intent (Likelihood to click/visit).
           - Conversion Intent (Likelihood to fulfill the primary campaign goal).
        7. COMPARATIVE RANKING: Rank variants based on their ability to move the "Hardest
           to Convert" persona.

        CONSTRAINTS & RULES:
        - BEHAVIORAL SKEPTICISM: Assume a "Default No" state for all personas. High intent
          scores ( > 80) must be justified by exceptional motivation-alignment.
        - CTA FRICTION: If the CTA requires significant effort (e.g., "Fill out long form")
          without a massive value-prop, conversion intent must be < 20%.
        - OUTPUT DISCIPLINE: Return results as a structured IntentEvaluation JSON object.
        """
        return self.generate(prompt, response_schema=IntentEvaluation)
