"""Prompt builder utilities for reducing duplication in agent prompts."""

from __future__ import annotations

from typing import Any


# =============================================================================
# Chain-of-Thought Reasoning Templates
# =============================================================================

COT_REASONING_TRACE = """
REASONING TRACE (internal - do not output):
1. ANALYZE GOAL HIERARCHY: Is this a awareness, consideration, or conversion goal?
   → Awareness: Lead with "why it matters now"
   → Conversion: Lead with "specific outcome + proof"
2. IDENTIFY ADVERSARIAL CONTEXT: What is the incumbent's weakness? What cognitive
   barrier prevents switch? What status quo is user attached to?
3. DETERMINE SCB (Single Consumer Benefit): The one benefit that, if delivered,
   justifies purchase. NOT a feature list.
4. MAP EMOTIONAL LEVER: Which of the 6 influence levers (Reciprocity, Commitment,
   Social Proof, Authority, Liking, Scarcity, Unity) best aligns with the audience
   psychology and goal?
5. WEIGHT EVALUATION DIMENSIONS: Based on the goal and audience, assign weights.
"""

COT_URL_ANALYSIS = """
REASONING TRACE:
1. CONTENT ANALYSIS: Identify the PRIMARY value proposition FIRST.
   - What is the ONE benefit the page leads with?
   - What evidence/proof does it provide (stats, testimonials, social proof)?
2. INFERRED VS EXPLICIT: Mark each extracted field as:
   - EXPLICIT: Directly stated in page content
   - INFERRED: Reasonably deduced from context
   - UNKNOWN: Cannot be determined from content
3. COMPETITIVE POSITIONING: What "only-ness" claim does the page make?
4. OBJECTION ANTICIPATION: What would make a skeptic not buy?
"""

COT_CREATIVE = """
REASONING TRACE:
1. MESSAGE HIERARCHY: What is the PRIMARY message? SECONDARY?
   - Primary: The one benefit that justifies the purchase
   - Secondary: Supporting proof or emotional reinforcement
2. AUDIENCE SEGMENT: Which psychological segment is this variant targeting?
   - Early adopters vs. mainstream
   - Need-driven vs. aspiration-driven
3. EMOTIONAL LEVER: Which of Cialdani's 6 levers applies?
4. PLATFORM CALIBRATION: How does this variant need to adapt for the platform?
5. HOOK STRATEGY: Pattern interrupt or gradual build?
"""

COT_EVALUATION = """
REASONING TRACE:
1. DIMENSION SCORING: Rate each dimension 0-100.
   - What specific evidence supports each score?
   - What specific issues drag the score down?
2. CRITICAL DEFECTS: Are there any issues that make this undeployable?
   - Legal/ethical violations
   - Brand misalignment
   - Factually incorrect claims
3. PRIORITY FIXES: Which issues to address first?
   - Impact on conversion
   - Ease of fix
"""


# =============================================================================
# Few-Shot Examples
# =============================================================================

KPI_EXAMPLE = """
EXAMPLE KPIs (for a SaaS conversion goal):
{
  "kpis": [
    {
      "name": "Click-Through Rate",
      "target_value": ">2.5%",
      "measurement_method": "Platform analytics"
    },
    {
      "name": "Conversion Rate",
      "target_value": ">4%",
      "measurement_method": "Pixel fire / landing page"
    },
    {
      "name": "Cost Per Acquisition",
      "target_value": "<$45",
      "measurement_method": "Ad platform attribution"
    }
  ],
  "pre_release_targets": {
    "clarity_score_target": 82,
    "brand_linkage_score_target": 78,
    "hook_strength_threshold": "strong (8+/10)",
    "message_retention_likelihood": ">65%",
    "simulated_intent_score": ">72%"
  }
}
"""

MESSAGE_PILLARS_EXAMPLE = """
EXAMPLE MESSAGE PILLARS:
[
  "1. Efficiency Gains: Save X hours/week (specific, quantifiable)",
  "2. Risk Elimination: Zero-downtime guarantee with SLA backing",
  "3. Peer Validation: Join X companies who switched in Q1"
]
"""

URL_EXTRACTION_EXAMPLE = """
EXAMPLE OUTPUT:
{
  "product_description": (
      "AI-powered meeting transcription and analytics platform "
      "that auto-generates action items and provides conversation insights"
  ),
  "value_proposition": (
      "Stop losing decisions in meetings - every word captured, "
      "organized, and actionable in under 60 seconds after your meeting ends"
  ),
  "target_audience": (
      "Remote and hybrid teams at companies with 50-500 employees, "
      "particularly engineering and product managers"
  ),
  "inferred_segments": [
    "Remote Team Coordinators",
    "Engineering Managers",
    "Product Managers",
    "Executive Assistants"
  ],
  "brand_name": "MeetIQ",
  "brand_tone": (
      "Professional but approachable, productivity-focused, trust-building"
  ),
  "visual_identity": (
      "Clean SaaS aesthetic, blue-purple gradient accents, minimal design, "
      "data visualization elements"
  ),
  "competitive_category": "AI Meeting Tools / Sales Intelligence",
  "differentiators": (
      "1-click action item extraction, native integrations with 12+ platforms, "
      "95%+ transcription accuracy claim, privacy-first (GDPR compliant)"
  ),
  "funnel_type": (
      "Mid-funnel (Consideration stage - users aware of problem)"
  ),
  "cta_type": (
      "Free trial (14 days, no credit card) with demo booking as secondary CTA"
  ),
  "missing_information": [
    "Exact pricing tiers (only 'starting at $X/user/mo' provided)",
    "Specific customer names in testimonials (only '500+ companies' mentioned)",
    "Technical integration requirements not detailed"
  ]
}
"""

CREATIVE_VARIANTS_EXAMPLE = """
EXAMPLE AdScript:
{
  "script_id": "variant_001",
  "scenes": [
    {
      "scene_number": 1,
      "duration_seconds": 5,
      "visual_cues": "Pattern interrupt - close-up face showing shock at price",
      "audio_notes": "Abrupt cut to silence, then voiceover begins",
      "dialogue": "What if you could...",
      "on_screen_text": None,
      "hook_or_payoff": "hook"
    },
    {
      "scene_number": 2,
      "duration_seconds": 15,
      "visual_cues": "Split screen: old method vs. new solution",
      "audio_notes": "Upbeat music builds tension",
      "dialogue": "...stop losing decisions in every meeting?",
      "on_screen_text": "Lost decisions: $4,000/productivity hour",
      "hook_or_payoff": "body"
    }
  ],
  "voiceover": (
      "Every meeting ends with decisions that never get made. "
      "MeetIQ captures every word, organizes it instantly, and "
      "makes sure nothing falls through the cracks."
  ),
  "hook": "Stop losing decisions in every meeting",
  "cta_text": "Try Free for 14 Days",
  "platform": "youtube",
  "duration_seconds": 30
}
"""


# =============================================================================
# PromptBuilder Class
# =============================================================================

class PromptBuilder:
    """Reusable prompt builder for agent prompts.

    Provides structured methods for building prompts with consistent
    formatting, few-shot examples, and constraint sections.
    """

    @staticmethod
    def build(
        *,
        role: str,
        objective: str,
        inputs: dict[str, Any],
        execution_steps: list[str] | None = None,
        few_shot: str | None = None,
        constraints: list[str] | None = None,
        output_schema: str | None = None,
        chain_of_thought: str | None = None,
    ) -> str:
        """Builds a structured agent prompt.

        Args:
            role: The role this agent should adopt.
            objective: The goal of the prompt.
            inputs: Key-value pairs to include as inputs.
            execution_steps: Ordered list of execution steps.
            few_shot: Few-shot example text.
            constraints: List of constraint strings.
            output_schema: Description of expected output schema.
            chain_of_thought: Chain-of-thought reasoning section.

        Returns:
            A formatted prompt string.
        """
        parts: list[str] = []

        if chain_of_thought:
            parts.append(chain_of_thought.strip())

        parts.append(f"Role: {role}.")
        parts.append(f"Objective: {objective}")

        if inputs:
            inputs_section = "\n\nINPUTS:"
            for key, value in inputs.items():
                inputs_section += f"\n- {key}: {value}"
            parts.append(inputs_section)

        if execution_steps:
            steps_section = "\n\nEXECUTION STEPS:"
            for i, step in enumerate(execution_steps, 1):
                steps_section += f"\n{i}. {step}"
            parts.append(steps_section)

        if few_shot:
            parts.append(f"\n\nFEW-SHOT EXAMPLES:\n{few_shot}")

        if constraints:
            constraints_section = "\n\nCONSTRAINTS & RULES:"
            for constraint in constraints:
                constraints_section += f"\n- {constraint}"
            parts.append(constraints_section)

        if output_schema:
            parts.append(f"\n\nFORMAT: Return ONLY valid JSON matching {output_schema} schema.")
            parts.append("No explanation, no preamble. JSON only.")

        return "\n".join(parts)


def build_strategy_prompt(
    *,
    product: str,
    goal: str,
    audience_desc: str,
    platforms: list[str],
    budget: str,
    timeline: str,
    brand_assets: str,
    key_differentiators: str,
    competitors: str,
    constraints: str,
    geography_market: str,
) -> str:
    """Builds a strategy agent prompt."""
    return PromptBuilder.build(
        role="Lead Growth Strategist & Brand Architect.",
        objective=(
            "Synthesize product intelligence and budget constraints into a "
            "multi-channel advertising strategy that maximizes conversion and brand equity."
        ),
        inputs={
            "Product Brand/Name": product,
            "Primary Campaign Goal": goal,
            "Target Audience": audience_desc,
            "Platforms": ", ".join(platforms) if platforms else "TBD (infer from audience)",
            "Budget": budget,
            "Timeline": timeline,
            "Brand Assets": brand_assets or "Not provided - use brand tone defaults",
            "Key Differentiators": (
                f"\n    {key_differentiators}" if key_differentiators
                else "Not explicitly provided - infer from product"
            ),
            "Competitors": competitors or "Not explicitly provided",
            "Constraints": constraints or "None specified",
            "Market/Geography": geography_market or "Global / Not specified",
        },
        execution_steps=[
            "ADVERSARIAL CHALLENGE: Define the 'Adversarial Market Challenge'",
            "SINGLE CONSUMER BENEFIT (SCB): Distill ALL product features into ONE benefit",
            "MESSAGE PILLAR DESIGN: Create exactly 3 pillars targeting different psychological levers",
            "CROSS-CHANNEL TACTICS: For each platform, define primary message, hook, and CTA",
            "SUCCESS METRICS (KPIs): Define 2-4 KPIs with numeric targets",
            "PRE-RELEASE TARGETS: Define exact score thresholds for AI evaluation",
            "WEIGHTED EVALUATION: Assign weights (totaling 1.0) across 5 dimensions",
            "CTA STRATEGY: Define psychological lever, CTA text format, and urgency mechanism",
        ],
        few_shot=f"{KPI_EXAMPLE}\n\n{MESSAGE_PILLARS_EXAMPLE}",
        constraints=[
            "NO GENERIC GOALS: 'Increase sales' is prohibited. Use 'Dethrone [incumbent]'",
            "SCB TEST: If you cannot explain the benefit in one sentence, it is not specific enough",
            "FEASIBILITY: Strategy must be executable within provided budget and media plan",
            "PLATFORM FIT: If platforms not specified, infer 2-3 optimal platforms from audience",
            "OUTPUT VALIDATION: Response MUST include all fields in StrategyBrief schema",
            "JSON DISCIPLINE: Return ONLY valid JSON matching StrategyBrief schema",
        ],
        chain_of_thought=COT_REASONING_TRACE,
    )


def build_url_intelligence_prompt(
    *,
    url: str,
) -> str:
    """Builds a URL intelligence agent prompt."""
    return PromptBuilder.build(
        role="Senior Business Intelligence Analyst & Competitive Researcher.",
        objective=(
            "Conduct a thorough extraction of product value propositions, competitive "
            "advantages, and market positioning from the provided URL."
        ),
        inputs={
            "Product URL": url,
        },
        execution_steps=[
            "PRODUCT DESCRIPTION: Start with H1 or hero statement, distill to 1-2 sentences",
            "VALUE PROPOSITION: What EMOTIONAL outcome? What problem does it solve?",
            "TARGET AUDIENCE: Primary buyer persona, company size, role/job title",
            "INFERRED SEGMENTS: 3-5 behavioral/psychographic segments",
            "BRAND IDENTITY: brand_name, brand_tone, visual_identity",
            "COMPETITIVE CATEGORY: How does page position itself vs. alternatives?",
            "DIFFERENTIATORS: 3-5 specific claims, quantify where possible",
            "FUNNEL STAGE: TOFU/MOFU/BOFU based on content depth and CTAs",
            "CTA TYPE: Primary/secondary CTA, urgency mechanism",
            "MISSING INFORMATION: What does a creative strategist need to know?",
        ],
        few_shot=URL_EXTRACTION_EXAMPLE,
        constraints=[
            "ZERO HALLUCINATION: Only use EXPLICIT content from the URL",
            "INFERRED fields must be marked with [INFERRED] prefix",
            "Do NOT invent statistics or claims not present on the page",
            "Capture exact quotes where compelling language is used",
        ],
        chain_of_thought=COT_URL_ANALYSIS,
    )
