"""Pydantic models for agent inputs and outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# === Step 0: URL Intelligence ===


class ExtractedInputs(BaseModel):
    """Structured data extracted from a product URL.

    This model captures the core brand and product information required
    to initialize the advertising pipeline.
    """

    product_description: str
    value_proposition: str
    target_audience: str
    inferred_segments: list[str]
    brand_name: str
    brand_tone: str
    visual_identity: str
    competitive_category: str
    differentiators: str
    funnel_type: str
    cta_type: str
    missing_information: list[str]


# === Step 0.5: Budget Inference ===


class BudgetAssumptions(BaseModel):
    """Baseline metric assumptions for budget estimation.

    Defines the expected performance ranges for CTR, CVR, and CPA.
    """

    expected_ctr_range: str
    expected_cvr_range: str
    expected_cpa_range: str


class TargetVolumeScenario(BaseModel):
    """A single budget scenario at a given scale.

    Maps a target volume (e.g., clicks or conversions) to the required
    spend.
    """

    scenario_type: str = Field(description="Low / Medium / High or Specific")
    target_volume: str
    required_budget: str


class PlatformAllocation(BaseModel):
    """Budget split for a single advertising platform.

    Defines how much of the total budget should be allocated to a
    specific platform like Meta or LinkedIn.
    """

    platform: str
    allocation_percentage: str


class BudgetInferenceReport(BaseModel):
    """Complete budget inference output with scenarios and allocation.

    Used by the Budget Agent to provide financial guardrails for the
    campaign strategy.
    """

    funnel_type: str
    assumptions: BudgetAssumptions
    target_volume_scenarios: list[TargetVolumeScenario]
    recommended_budget: str
    min_budget: str
    max_budget: str
    platform_allocation: list[PlatformAllocation]
    confidence_level: str
    sensitivity_drivers: list[str]
    user_validation_flag: str = Field(
        description="Must be confirmed or adjusted by user",)


# === Step 1: Strategy ===


class KPI(BaseModel):
    """A single key performance indicator definition.

    Defines what to measure, the target value, and the verification
    method.
    """

    name: str
    target_value: str
    measurement_method: str


class Persona(BaseModel):
    """Demographics and behavioral profile for an audience segment.

    Used in the initial strategy brief to define target audience
    characteristics.
    """

    demographics: str
    motivation: str
    pain_point: str
    buying_trigger: str
    likely_objection: str


class MessageStrategy(BaseModel):
    """Core messaging framework including tone and triggers.

    Outlines the 'how' and 'what' of the brand communication.
    """

    value_proposition: str
    message_pillars: list[str]
    tone: str
    emotional_trigger: str


class PreReleaseTargets(BaseModel):
    """Pre-release AI evaluation score targets.

    Defines the success criteria that the AI agents will use to validate
    ad variants before deployment.
    """

    clarity_score_target: int
    brand_linkage_score_target: int
    hook_strength_threshold: str
    message_retention_likelihood: str
    simulated_intent_score: str


class StrategyBrief(BaseModel):
    """Complete advertising strategy document.

    The foundational document that guides all downstream creative and
    evaluative agents.
    """

    product_name: str = ""  # Added to ensure compatibility
    ad_intent: str
    intent_explanation: str
    kpis: list[KPI]
    pre_release_targets: PreReleaseTargets
    audience_personas: list[Persona]
    message_strategy: MessageStrategy
    creative_constraints: dict[str, Any]
    decision_thresholds: dict[str, float]


# === Step 2: Audience ===


class ResponseSimulation(BaseModel):
    """Simulated viewer response to an ad.

    Predicts the cognitive and emotional reaction of a persona to the
    creative content.
    """

    initial_reaction: str
    attention_grabber: str
    confusion_points: str
    objection_triggered: str
    completion_likelihood: int = Field(ge=0, le=100)


class IntentSimulation(BaseModel):
    """Simulated engagement and purchase intent scores.

    Numerical estimation of how likely a persona is to take action.
    """

    engagement_likelihood: int = Field(ge=0, le=100)
    click_likelihood: int = Field(ge=0, le=100)
    purchase_intent: int = Field(ge=0, le=100)


class FitAnalysis(BaseModel):
    """Message-to-persona fit assessment.

    Evaluates the relevance of the ad message to the specific persona's
    needs.
    """

    fit_score: int = Field(ge=0, le=100)
    explanation: str


class PersonaProfile(BaseModel):
    """Full persona profile with simulated response data.

    Expanded version of the base Persona used during the simulation
    phase.
    """

    id: str = ""
    name: str
    snapshot: str
    awareness_level: str = Field(
        description="Unaware / Problem-aware / Solution-aware / Product-aware")
    motivation: str
    pain_point: str
    desired_outcome: str
    buying_trigger: str
    key_objection: str
    decision_speed: str = Field(description="fast / moderate / slow")
    response_simulation: ResponseSimulation
    intent_simulation: IntentSimulation
    fit_analysis: FitAnalysis


class CrossPersonaInsights(BaseModel):
    """Aggregate insights across all personas.

    Identifies patterns, strengths, and weaknesses across the entire
    audience model.
    """

    common_strengths: list[str]
    common_failure_points: list[str]
    highest_performing_persona: str
    lowest_performing_persona: str


class AudienceSegments(BaseModel):
    """Complete audience modeling output.

    Contains all persona profiles and aggregated strategic insights.
    """

    personas: list[PersonaProfile]
    cross_persona_insights: CrossPersonaInsights
    optimization_recommendations: list[str]


# === Step 3: Creative ===


class Scene(BaseModel):
    """A single scene within an ad script.

    Defines the visual, auditory, and narrative elements of a discrete
    moment in time.
    """

    description: str
    visual_cues: str
    dialogue_vo: str


class AdScript(BaseModel):
    """A complete ad variant script.

    The primary creative output of the pipeline, including hooks,
    scenes, and CTAs.
    """

    concept_title: str
    core_idea: str
    hook: str
    script_scenes: list[Scene]
    brand_integration: str
    cta: str
    video_prompt: str = Field(
        description="Detailed prompt for video generation models "
        "like Veo based on the visual cues and core idea")
    variant_name: str


class CreativeVariants(BaseModel):
    """Collection of generated ad variants.

    Allows the pipeline to process multiple creative concepts
    simultaneously.
    """

    variants: list[AdScript]


# === Step 4.1: Clarity ===


class ClaritySubScores(BaseModel):
    """Granular clarity scoring dimensions.

    Breaks down clarity into comprehensibility, cognitive load,
    explicitness, and noise.
    """

    comprehensibility: int = Field(ge=0, le=100)
    cognitive_load: int = Field(ge=0, le=100)
    explicitness: int = Field(ge=0, le=100)
    signal_vs_noise: int = Field(ge=0, le=100)


class PersonaComprehension(BaseModel):
    """Simulated comprehension for a single persona.

    Documents what a specific persona understood (and misunderstood).
    """

    persona_name: str
    what_they_think_it_is_about: str
    potential_misunderstandings: str


class ClarityEvaluation(BaseModel):
    """Complete message clarity evaluation output.

    Diagnostic report on how well the ad communicates its core message.
    """

    core_message: str
    message_consistency: str
    clarity_score: int = Field(ge=0, le=100)
    sub_scores: ClaritySubScores
    persona_comprehensions: list[PersonaComprehension]
    retention_likelihood: int = Field(ge=0, le=100)
    retention_justification: str
    time_to_clarity: str
    issues_identified: list[str]
    recommended_fixes: list[str]


# === Step 4.2: Brand Linkage ===


class BrandLinkageSubScores(BaseModel):
    """Granular brand linkage scoring dimensions.

    Evaluates attribution clarity, timing, integration, and
    memorability.
    """

    attribution_clarity: int = Field(ge=0, le=100)
    timing: int = Field(ge=0, le=100)
    integration: int = Field(ge=0, le=100)
    memorability: int = Field(ge=0, le=100)


class PersonaRecall(BaseModel):
    """Simulated brand recall for a single persona.

    Estimates the probability that the persona will correctly attribute
    the ad to the brand.
    """

    persona_name: str
    expected_brand_answer: str
    confidence_level: int = Field(ge=0, le=100)
    flags: list[str]


class BrandLinkageEvaluation(BaseModel):
    """Complete brand linkage evaluation output.

    Assesses the strength of the connection between the creative and the
    brand.
    """

    brand_attribution_summary: str
    brand_linkage_score: int = Field(ge=0, le=100)
    sub_scores: BrandLinkageSubScores
    persona_recalls: list[PersonaRecall]
    confusion_risk: str = Field(description="Low / Medium / High")
    confusion_explanation: str
    distinctiveness_score: int = Field(ge=0, le=100)
    alignment_score: int = Field(ge=0, le=100)
    issues_identified: list[str]
    recommended_fixes: list[str]


# === Step 4.3: Diagnostics ===


class HookAnalysis(BaseModel):
    """Diagnostic analysis of the ad hook effectiveness.

    Evaluates the strength and type of the initial few seconds of the
    ad.
    """

    hook_strength_score: int = Field(ge=0, le=100)
    hook_type_classification: str
    weaknesses: list[str]


class RiskSegment(BaseModel):
    """A single drop-off risk point in the ad timeline.

    Identifies specific moments where viewers are likely to stop
    watching.
    """

    segment: str
    risk_severity: str = Field(description="Low / Medium / High")
    cause: str


class DiagnosticsEvaluation(BaseModel):
    """Complete creative diagnostics evaluation.

    Technical teardown of the ad's narrative structure, pacing, and
    flow.
    """

    structure_type: str
    coherence_score: int = Field(ge=0, le=100)
    hook_analysis: HookAnalysis
    pacing_score: int = Field(ge=0, le=100)
    sections_drag_rush: list[str]
    drop_off_risk_timeline: list[RiskSegment]
    redundancy_score: int = Field(ge=0, le=100, description="Lower is better")
    elements_to_remove: list[str]
    transition_quality_score: int = Field(ge=0, le=100)
    platform_fit_score: int = Field(ge=0, le=100)
    mismatches_identified: list[str]
    overall_diagnostic_score: int = Field(ge=0, le=100)
    issues_identified: list[str]
    recommended_fixes: list[str]


# === Step 4.4: Attention ===


class PersonaAttention(BaseModel):
    """Attention simulation for a single persona.

    Models engagement patterns for a specific audience segment.
    """

    persona_name: str
    attention_grabber: str
    disengagement_cause: str
    retention_score: int = Field(ge=0, le=100)


class AttentionEvaluation(BaseModel):
    """Complete attention heuristic evaluation.

    Estimates the ad's ability to stop the scroll and sustain interest.
    """

    first_impression_score: int = Field(ge=0, le=100)
    attention_trigger_type: str
    scroll_stop_probability: int = Field(ge=0, le=100)
    novelty_score: int = Field(ge=0, le=100)
    familiarity_risk: str = Field(description="Low / Medium / High")
    density_score: int = Field(ge=0, le=100)
    overload_risk: str = Field(description="Low / Medium / High")
    retention_curve_description: str
    mid_point_attention_strength: int = Field(ge=0, le=100)
    persona_attention_insights: list[PersonaAttention]
    risk_factors: list[str]
    overall_attention_score: int = Field(ge=0, le=100)
    recommended_fixes: list[str]


# === Step 4.5: Intent ===


class PersonaIntent(BaseModel):
    """Intent simulation for a single persona.

    Estimates the behavioral likelihood at different stages of the
    funnel.
    """

    persona_name: str
    behavioral_path: str
    engagement_intent: int = Field(ge=0, le=100)
    action_intent: int = Field(ge=0, le=100)
    conversion_intent: int = Field(ge=0, le=100)
    justification: str
    objections: list[str]
    objection_severity: str = Field(description="Low / Medium / High")
    motivation_alignment_score: int = Field(ge=0, le=100)
    alignment_gaps: list[str]


class FunnelDropOff(BaseModel):
    """A single funnel drop-off point and reason.

    Specific to the conversion funnel, identifying friction points.
    """

    point: str = Field(
        description="After hook / Mid-content / Before CTA / At CTA")
    reason: str


class VariantIntent(BaseModel):
    """Aggregated intent scores for one ad variant.

    Summarizes the expected behavioral impact of a specific creative.
    """

    concept_title: str
    persona_intents: list[PersonaIntent]
    funnel_drop_offs: list[FunnelDropOff]
    aggregated_intent_score: int = Field(ge=0, le=100)


class VariantRanking(BaseModel):
    """Ranking entry for one ad variant.

    Used to compare variants against each other based on intent.
    """

    rank: int
    concept_title: str
    reasoning: str


class IntentEvaluation(BaseModel):
    """Complete cross-variant intent evaluation.

    The final evaluative report before global scoring and iteration.
    """

    variant_intents: list[VariantIntent]
    variant_ranking: list[VariantRanking]
    persona_variant_fit_matrix: dict[str, dict[str, int]] = Field(
        description="Persona Name -> {Variant Title -> Fit Score}")
    key_risks: list[str]
    optimization_recommendations: list[str]


# === Step 5: Scoring ===


class CategoryScore(BaseModel):
    """Weighted score for a single evaluation category.

    Represents the contribution of a dimension like 'Brand' or 'Intent'
    to the total.
    """

    category: str
    raw_score: float
    weight: float
    weighted_contribution: float


class VariantFinalDecision(BaseModel):
    """Final GO/NO-GO decision metadata for a variant.

    The ultimate pass/fail status for a creative concept.
    """

    concept_title: str
    final_readiness_score: float
    is_ready: bool
    status: str
    confidence: float
    primary_strength: str
    primary_blocker: str | None
    category_breakdown: list[CategoryScore]


class TradeOff(BaseModel):
    """Evaluation of a trade-off between conflicting metrics.

    Documents why one performance dimension was prioritized over
    another.
    """

    dimension_a: str
    dimension_b: str
    winning_dimension: str
    justification: str


class CompositeReadinessReport(BaseModel):
    """Final aggregated readiness assessment across all variants.

    The executive summary of the entire pipeline's intelligence.
    """

    target_kpis: list[str]
    variant_decisions: list[VariantFinalDecision]
    best_overall_variant: str
    system_readiness_flag: bool
    strategic_trade_offs: list[TradeOff]
    critical_risks: list[str]


# === Step 6: Iteration ===


class IterationDirective(BaseModel):
    """Specific instruction for refining an ad variant.

    Actionable guidance for the LLM to improve the creative content.
    """

    category: str
    severity: str
    instruction: str
    expected_outcome: str


class PrioritizedIssue(BaseModel):
    """High-priority issue identified for iteration.

    Filters all agent feedback into the most critical blockers.
    """

    issue: str
    source_agent: str
    impact: str


class IterationAction(BaseModel):
    """Specific action to be taken in the next iteration.

    A concrete edit request targeting a scene or element.
    """

    action: str
    target_scene: str = "All"
    rationale: str


class VariantIterationPlan(BaseModel):
    """Complete iteration plan for a single variant.

    Consolidates directives, issues, and actions for a concept.
    """

    concept_title: str
    iteration_directives: list[IterationDirective]
    prioritized_issues: list[PrioritizedIssue]
    suggested_actions: list[IterationAction]


class IterationControlReport(BaseModel):
    """Aggregate iteration plan across all variants.

    Guides the feedback loop for the next generation cycle.
    """

    variant_plans: list[VariantIterationPlan]
    cycle_count: int
    global_refinement_strategy: str


# === Step 7: Adaptation ===


class PlatformVariant(BaseModel):
    """Platform-specific adaptation of an ad script.

    Modifies copy and visuals for Meta, TikTok, etc.
    """

    platform: str
    adapted_headline: str
    adapted_body_copy: str
    visual_spec_changes: str
    cta_adaptation: str


class PlatformAdaptations(BaseModel):
    """Collection of adaptations for a single variant across platforms.

    Groups multiple platform versions for a single concept.
    """

    concept_title: str
    adaptations: list[PlatformVariant]


class PlatformAdaptationReport(BaseModel):
    """Complete platform adaptation output.

    The final script adjustments for multi-channel deployment.
    """

    variant_adaptations: list[PlatformAdaptations]


# === Step 8: Compliance ===


class RiskIdentified(BaseModel):
    """A single compliance or brand safety risk.

    Flags violations of legal, policy, or brand guidelines.
    """

    risk_type: str
    severity: str
    description: str
    remediation: str


class VariantComplianceReport(BaseModel):
    """Compliance assessment for a single variant.

    A pass/fail report on policy adherence for a concept.
    """

    concept_title: str
    status: str
    risks: list[RiskIdentified]


class ComplianceRiskReport(BaseModel):
    """Aggregate compliance risk report.

    The final safety sign-off for the entire campaign.
    """

    variant_reports: list[VariantComplianceReport]
    overall_status: str


# === Step 9: Asset Production ===


class RequiredAsset(BaseModel):
    """A single required creative asset (image, video, etc.).

    Specifies the ingredients needed to build the final ad.
    """

    asset_type: str
    description: str
    role_in_creative: str


class ShotSceneDesign(BaseModel):
    """Production design for a single shot or scene.

    The technical Blueprint for visual and audio production.
    """

    shot_number: int
    visual_description: str
    audio_description: str
    technical_specs: str


class AssetProductionVariant(BaseModel):
    """Complete production plan for a single variant.

    The full production brief for a concept.
    """

    concept_title: str
    required_assets: list[RequiredAsset]
    production_scenes: list[ShotSceneDesign]
    estimated_production_complexity: str


class AssetProductionReport(BaseModel):
    """Aggregate asset production report.

    Consolidates production needs across all variants.
    """

    production_variants: list[AssetProductionVariant]


# === Step 10: Validation ===


class ExperimentDesign(BaseModel):
    """Design for a controlled ad experiment.

    Defines the statistical framework for pre-launch testing.
    """

    description: str
    hypothesis: str
    variables: list[str]


class MetricMapping(BaseModel):
    """Mapping of business metrics to simulated indicators.

    Connects proxy AI scores to real-world KPI expectations.
    """

    metric: str
    indicator: str


class VariantValidationResult(BaseModel):
    """Validation outcome for a single variant.

    The post-simulation validation status.
    """

    concept_title: str
    validation_status: str
    predicted_confidence: float


class ExternalValidationPlan(BaseModel):
    """Complete external validation design.

    The final plan for A/B or market testing.
    """

    experiment_designs: list[ExperimentDesign]
    metric_mapping: list[MetricMapping]
    variant_validation_results: list[VariantValidationResult]


# === Step 11: Deployment ===


class ExperimentLaunchPlan(BaseModel):
    """Launch plan for a single deployment experiment.

    The tactical execution plan for a specific platform.
    """

    platform: str
    budget_allocation: str
    bidding_strategy: str


class DeploymentExperimentationReport(BaseModel):
    """Complete deployment and experimentation plan.

    The master rollout schedule and test plan.
    """

    launch_plans: list[ExperimentLaunchPlan]
    test_timeline: str
    scaling_triggers: list[str]


# === Step 12: Learning ===


class CampaignRecord(BaseModel):
    """Historical record of campaign performance.

    Used for long-term pattern recognition and system tuning.
    """

    campaign_id: str
    performance_metrics: dict[str, Any]


class PredictionRealityReport(BaseModel):
    """Comparison of predicted vs actual outcomes.

    The primary feedback mechanism for improving AI accuracy.
    """

    predicted_performance: dict[str, Any]
    actual_performance: dict[str, Any]
    variance_explanation: str


class KeyPatterns(BaseModel):
    """Identified high-performing patterns.

    Strategic takeaways for future campaign designs.
    """

    pattern_type: str
    description: str


class AgentPerformance(BaseModel):
    """Diagnostic performance of an individual agent.

    Tracks how well specific agents predict real outcomes.
    """

    agent_name: str
    accuracy_score: float


class KnowledgeLearningReport(BaseModel):
    """Complete cross-campaign learning report.

    The system-wide recursive improvement output.
    """

    campaign_records: list[CampaignRecord]
    prediction_reality_reports: list[PredictionRealityReport]
    key_patterns: list[KeyPatterns]
    agent_performance_diagnostics: list[AgentPerformance]


# === Step 13: Video Generation ===


class VideoGenerationResult(BaseModel):
    """Result of a video generation task.

    Contains the file path and metadata for a Veo-generated ad.
    """

    video_file_path: str
    generation_metadata: dict[str, Any] | None = None
