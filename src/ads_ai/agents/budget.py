"""Budget Inference Agent module."""

from __future__ import annotations

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import BudgetInferenceReport, ExtractedInputs


class BudgetInferenceAgent(BaseAgent):
    """Provides financial guardrails and budget estimates for campaigns.

    This agent analyzes product inputs and market context to infer
    realistic budget scenarios, CTRs, and conversion costs.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the BudgetInferenceAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def infer_budget(self,
                     goal: str,
                     product: str,
                     funnel_type: str,
                     audience_size: str,
                     platforms: list[str],
                     market_context: str = "") -> BudgetInferenceReport:
        """Generates a budget inference report based on campaign goals.

        Args:
            goal: Primary business objective.
            product: Product description and value prop.
            funnel_type: Type of sales funnel (e.g. TOFU, BOFU).
            audience_size: Estimated audience Reach.
            platforms: Target advertising platforms.
            market_context: Optional additional market context.

        Returns:
            A ``BudgetInferenceReport`` instance.

        Raises:
            Exception: If the inference calculation or generation fails.
        """
        prompt = f"""
        Role: Mathematical Media Planner & Platform Economist.
        Objective: Infer a realistic budget distribution and platform selection based on
        ad goals, market category, and competitive density.

        INPUTS:
        - Campaign Goal: {goal}
        - Product/Brand Intelligence: {product}
        - Funnel Type: {funnel_type}
        - Target Platforms: {', '.join(platforms)}
        - Market Context: {market_context if market_context else "Standard benchmarks."}

        EXECUTION STEPS:
        1. SECTOR DENSITY ANALYSIS: Estimate the cost-per-mille (CPM) and cost-per-click (CPC)
           for the product's category.
        2. PLATFORM OPTIMIZATION: Rank and select 2-3 platforms (e.g., TikTok, LinkedIn, Meta)
           based on where the CPM/Intent ratio is most favorable.
        3. BUDGET ALLOCATION: Distribute the budget across the selected platforms using a
           70/20/10 split (Proven/Growth/Experimental).
        4. UNIT ECONOMICS PROJECTION: Predict the "Break-even CPA" (Cost Per Acquisition)
           target for the campaign.
        5. SCALING TRIGGERS: Define the budget thresholds for "Testing" vs. "Scaling" phases.

        CONSTRAINTS & RULES:
        - DATA-DRIVEN: Do NOT pick platforms at random. Use the "URL Intelligence" to match
           the product's complexity to the platform's user behavior.
        - REALISM: If the budget is low (<$500), recommend focusing on a single "High-Intent"
          platform rather than spreading thin.
        - OUTPUT DISCIPLINE: Return results as a structured BudgetInferenceReport JSON object.
        """
        return self.generate(prompt, response_schema=BudgetInferenceReport)
