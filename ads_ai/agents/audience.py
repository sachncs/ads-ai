"""Audience Agent module."""

from __future__ import annotations

import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AudienceSegments, StrategyBrief

logger = logging.getLogger(__name__)


class AudienceAgent(BaseAgent):
    """Models audience personas and simulates ad response.

    This agent transforms high-level strategy into detailed behavioral
    personas and predicts their initial reactions to common ad hooks.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the AudienceAgent with a GenAI client.

        Args:
            client: The Google GenAI client.
        """
        super().__init__(client, model_name="gemini-3.1-pro-preview")

    def model_personas(
        self,
        product: str,
        strategy: StrategyBrief,
        target_audience: str,
        platforms: list[str],
        market_context: str = "",
    ) -> AudienceSegments:
        """Transforms a high-level target audience into structured personas.

        Args:
            product: Product description.
            strategy: Strategy document.
            target_audience: Target audience description.
            platforms: List of platforms.
            market_context: Optional market context.

        Returns:
            An ``AudienceSegments`` instance containing detailed personas.

        Raises:
            google.api_core.exceptions.InternalServerError: If modeling fails.
            ValueError: If response parsing fails.
        """
        logger.info(
            "model_personas started target_audience=%s platforms=%s",
            target_audience[:50],
            platforms,
        )
        start = time.perf_counter()
        try:
            cot_reasoning = """
        REASONING TRACE:
        1. PERSONA TYPOLOGY SELECTION: Choose 3 distinct behavioral types based on
           audience description and strategy goal:
           → The Advocate: High social proof sensitivity, early majority adopter
           → The Skeptic: Low initial trust, requires proof and risk reversal
           → The Efficiency Seeker: Task-focused, low patience, values speed/demonstration
           → The Status Seeker: Prestige-motivated, responds to exclusivity/authority
           → The Anxious Achiever: Fear-of-loss driven, responds to urgency/proof

        2. BARRIER-TRANSFORMATION MAPPING:
           For each persona, map:
           - CURRENT STATE: The "as-is" condition they believe is normal or unfixable
           - DESIRED STATE: The specific outcome they want but can't articulate
           - ACTIVE BARRIER: Why they haven't solved it yet (cost, trust, complexity, inertia)
           - IMPLICIT QUESTION: What they're silently asking when they see this product

        3. TRIGGER IDENTIFICATION:
           Internal triggers (emotional): Anxiety, Pride, Envy, Guilt, Joy
           External triggers (environmental): Social proof, Authority, Scarcity, Reciprocity
           Match trigger type to persona's dominant psychological driver.

        4. CHANNEL AFFINITY INFERENCE:
           Based on persona demographics and psychographics:
           → Content style: Educational vs. Entertaining vs. Inspirational
           → Platform: LinkedIn vs. TikTok vs. Instagram vs. YouTube
           → Format tolerance: 15s skippable vs. 3min deep dive
        """

            few_shot_persona = """
        EXAMPLE PERSONA PROFILE:
        {
          "id": "persona_001",
          "name": "The Efficiency Obsessive",
          "snapshot": "Senior PM at a 50-200 person startup, constantly context-switching, "
                      "measures everything, skeptical of marketing claims",
          "awareness_level": "Solution-aware (knows category exists, evaluating options)",
          "motivation": "Recover 5+ hours/week of manual work to invest in strategic initiatives",
          "pain_point": "Current workflow requires 8+ tools and 3+ hours daily of manual reporting",
          "desired_outcome": (
              "One integrated dashboard with auto-populated metrics "
              "and zero manual entry"
          ),
          "buying_trigger": "Seeing exact time-savings quantified with a trial period",
          "key_objection": "Implementation will take longer than the time it saves",
          "decision_speed": "moderate (2-4 weeks evaluation)",
          "response_simulation": {
            "initial_reaction": "Skeptical - scrolls past generic B2B ad instantly",
            "attention_grabber": "Specific metric or data point that validates their pain",
            "confusion_points": "Unclear pricing tiers or vague feature descriptions",
            "objection_triggered": "Implementation complexity mentioned before proof",
            "completion_likelihood": 65
          },
          "intent_simulation": {
            "engagement_likelihood": 72,
            "click_likelihood": 45,
            "purchase_intent": 38
          },
          "fit_analysis": {
            "fit_score": 78,
            "explanation": "High efficiency motivation + moderate purchase intent = high potential "
                          "if pricing and implementation are addressed upfront"
          }
        }

        EXAMPLE CROSS-PERSONA INSIGHTS:
        {
          "common_strengths": ["Clear pain point articulation", "Specific time metrics"],
          "common_failure_points": ["Vague pricing", "Implementation complexity not addressed"],
          "highest_performing_persona": "The Efficiency Obsessive",
          "lowest_performing_persona": "The Skeptic"
        }
        """

            prompt = f"""
{cot_reasoning}

        Role: Senior Behavioral Psychologist & Market Researcher.
        Objective: Synthesize product data into 3 high-fidelity audience personas based
        on deep psychological drivers and behavioral economics. These personas will be
        used as simulation engines for all downstream evaluation agents.

        INPUTS:
        - Product: {product}
        - Strategy Brief: {strategy.model_dump_json()}
        - Target Audience Description: {target_audience}
        - Target Platforms: {", ".join(platforms) if platforms else "TBD"}
        - Market Context: {market_context or "Not specified"}

        PERSONA GENERATION RULES:
        - Create EXACTLY 3 personas with DISTINCT behavioral profiles
        - No two personas should have the same motivation + decision_speed combination
        - At least one persona must be a "hard sell" (purchase_intent < 40)
        - Each persona must have a unique awareness_level from: Unaware, Problem-aware,
          Solution-aware, Product-aware, Most-aware

        REQUIRED PERSONA COMPONENTS:

        1. BEHAVIORAL CORE:
           - name: Specific behavioral label (e.g., "The Efficiency Obsessive", not "Tech User")
           - snapshot: 2-3 sentence "day in the life" describing their work/style
           - awareness_level: Where they are in the purchase funnel

        2. MOTIVATION MAPPING:
           - motivation: The specific outcome they want (quantified where possible)
           - pain_point: What specifically frustrates them (NOT generic "time is money")
           - desired_outcome: The concrete end-state they're seeking
           - buying_trigger: The specific proof/event that tips them to act
           - key_objection: The ONE thing that will prevent purchase if not addressed

        3. DECISION DYNAMICS:
           - decision_speed: fast / moderate / slow (affects urgency in creative)
           - response_simulation: How they mentally process an ad (be specific about what
             makes them scroll past vs. stop)
           - intent_simulation: Numerical likelihood scores (0-100) for engagement,
             click, and purchase intent

        4. FIT ANALYSIS:
           - fit_score: How well this persona aligns with the product's core value prop
           - explanation: WHY the fit score is what it is

        5. PLATFORM CHANNELING:
           For each persona, infer optimal:
           - Primary platform (where they're most receptive)
           - Content format (what type of ad they'd stop for)
           - Message frame (logical/proof-based vs. emotional/identity-based)

        FEW-SHOT EXAMPLE:
        {few_shot_persona}

        OUTPUT VALIDATION CHECKS:
        □ Exactly 3 personas generated
        □ Each persona has unique name and behavioral snapshot
        □ Sum of purchase_intent across personas has variance > 200 (not all same)
        □ Each persona has distinct decision_speed from at least one other
        □ Key objection for each persona would require different creative angle to address

        FORMAT: Return ONLY valid JSON matching AudienceSegments schema.
        No explanation, no preamble. JSON only.
        """
            report = self.generate(prompt, response_schema=AudienceSegments)
            elapsed = time.perf_counter() - start
            logger.info(
                "model_personas completed target_audience=%s elapsed=%.3fs",
                target_audience[:50],
                elapsed,
            )
            return report
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(
                "model_personas failed target_audience=%s elapsed=%.3fs",
                target_audience[:50],
                elapsed,
            )
            raise
