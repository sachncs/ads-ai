"""Orchestrator pipeline for the multi-agent advertising system.

This module coordinates 17 specialized agents through a strict 13-step
workflow to generate, evaluate, iterate, and finalize production-ready
advertising creatives and video assets.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from pydantic import BaseModel

from ads_ai.agents import (
    AdScript,
    AssetProductionAgent,
    AssetProductionReport,
    AttentionHeuristicAgent,
    AudienceAgent,
    AudienceSegments,
    BrandLinkageAgent,
    BudgetInferenceAgent,
    BudgetInferenceReport,
    ComplianceRiskAgent,
    ComplianceRiskReport,
    CompositeReadinessReport,
    CreativeAgent,
    DeploymentExperimentationAgent,
    DeploymentExperimentationReport,
    DiagnosticsAgent,
    ExternalValidationAgent,
    ExternalValidationPlan,
    ExtractedInputs,
    IntentSimulationAgent,
    IterationControllerAgent,
    IterationControlReport,
    KnowledgeLearningAgent,
    KnowledgeLearningReport,
    MessageClarityAgent,
    PlatformAdaptationAgent,
    PlatformVariant,
    ScoringAgent,
    StrategyAgent,
    StrategyBrief,
    URLIntelligenceAgent,
    VideoGenerationAgent,
    VideoGenerationResult,
)
from ads_ai.utils.file_ops import ensure_output_dir, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BUDGET_PLACEHOLDER = "TBD"
"""Sentinel value indicating that budget has not been specified."""

DEFAULT_CONVERGENCE_THRESHOLD = 70
"""Minimum composite score required for a variant to receive a GO decision."""

DEFAULT_MAX_ITERATIONS = 3
"""Maximum number of evaluate-iterate cycles before forcing a decision."""

MAX_PARALLEL_EVALUATIONS = 4
"""Upper bound on concurrent variant evaluation threads."""

PASSING_DECISIONS = frozenset({"GO"})
"""Set of variant decision labels that qualify for post-approval stages."""

FAILING_DECISIONS = frozenset({"NO-GO", "CONDITIONAL GO"})
"""Set of variant decision labels that trigger refinement during iteration."""

# ---------------------------------------------------------------------------
# Pipeline Result Schema
# ---------------------------------------------------------------------------


class PipelineResult(BaseModel):
    """Structured output returned by the orchestrator pipeline.

    Attributes:
        strategy: The strategy brief produced in step 1.
        personas: Audience persona profiles produced in step 2.
        approved_variants: Ad scripts that passed the GO gate.
        readiness_report: Final composite readiness scores and decisions.
        iteration_history: Reports from each iteration cycle.
        platform_adaptations: Platform-specific variant adaptations.
        compliance_report: Output of the compliance and risk agent.
        production_report: Asset production plan.
        validation_plan: Human / external validation design.
        deployment_report: Deployment and experimentation plan.
        learning_report: Knowledge and learning capture output.
        budget_inference_report: Optional budget inference if auto-derived.
        video_results: List of video generation results.
        video_paths: List of output video file paths.
    """

    strategy: StrategyBrief
    personas: AudienceSegments
    approved_variants: list[AdScript]
    readiness_report: CompositeReadinessReport
    iteration_history: list[IterationControlReport]
    platform_adaptations: list[PlatformVariant]
    compliance_report: ComplianceRiskReport
    production_report: AssetProductionReport
    validation_plan: ExternalValidationPlan
    deployment_report: DeploymentExperimentationReport
    learning_report: KnowledgeLearningReport
    budget_inference_report: BudgetInferenceReport | None = None
    video_results: list[VideoGenerationResult] = []
    video_paths: list[str] = []


# ---------------------------------------------------------------------------
# Orchestrator Pipeline
# ---------------------------------------------------------------------------


class OrchestratorPipeline:
    """Coordinates the 12-step advertising pipeline across 17 agents.

    The orchestrator does not perform agent logic itself. It assigns,
    routes, validates, and iterates until convergence or exhaustion of
    the maximum iteration budget.

    Attributes:
        client: The Google GenAI client instance.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes all sub-agents with the shared GenAI client.

        Args:
            client: An authenticated ``genai.Client`` instance.
        """
        self.client = client
        self.url_agent = URLIntelligenceAgent(client)
        self.budget_agent = BudgetInferenceAgent(client)
        self.strategy_agent = StrategyAgent(client)
        self.audience_agent = AudienceAgent(client)
        self.creative_agent = CreativeAgent(client)
        self.clarity_agent = MessageClarityAgent(client)
        self.brand_agent = BrandLinkageAgent(client)
        self.diagnostics_agent = DiagnosticsAgent(client)
        self.attention_agent = AttentionHeuristicAgent(client)
        self.intent_agent = IntentSimulationAgent(client)
        self.scoring_agent = ScoringAgent(client)
        self.adaptation_agent = PlatformAdaptationAgent(client)
        self.iteration_agent = IterationControllerAgent(client)
        self.validation_agent = ExternalValidationAgent(client)
        self.learning_agent = KnowledgeLearningAgent(client)
        self.compliance_agent = ComplianceRiskAgent(client)
        self.production_agent = AssetProductionAgent(client)
        self.deployment_agent = DeploymentExperimentationAgent(client)
        self.video_agent = VideoGenerationAgent(client)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_from_url(
        self,
        url: str,
        goal: str,
        platforms: list[str] | None = None,
        budget: str = DEFAULT_BUDGET_PLACEHOLDER,
        timeline: str = DEFAULT_BUDGET_PLACEHOLDER,
        threshold: int = DEFAULT_CONVERGENCE_THRESHOLD,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        output_dir: str | None = None,
    ) -> PipelineResult | None:
        """Runs the full pipeline starting from a product URL.

        Extracts structured inputs via the URL Intelligence Agent, optionally
        infers a budget, then delegates to ``run``.

        Args:
            url: The product or landing page URL to analyze.
            goal: Primary business objective in plain language.
            platforms: Target advertising platforms.
            budget: Spending budget or ``"TBD"`` for auto-inference.
            timeline: Campaign timeline or ``"TBD"``.
            threshold: Minimum composite score for GO decisions.
            max_iterations: Maximum evaluate-iterate cycles.
            output_dir: Root directory for run outputs.

        Returns:
            A ``PipelineResult`` on success, or ``None`` if no variants pass.
        """
        pipeline_start = time.perf_counter()
        logger.info(
            "run_from_url started url=%s goal=%s",
            url,
            goal,
        )
        step_start = time.perf_counter()
        logger.info("STEP 0 -> URL INTELLIGENCE (Extracting from URL)")
        extracted = self.url_agent.parse_url(url)
        logger.info(
            "STEP 0 completed url=%s elapsed=%.3fs",
            url,
            time.perf_counter() - step_start,
        )

        product = (f"{extracted.brand_name}: {extracted.product_description}\n"
                   f"Value Prop: {extracted.value_proposition}")
        audience_desc = (f"Primary: {extracted.target_audience}\n"
                         f"Segments: {', '.join(extracted.inferred_segments)}")

        inferred_budget_report = self.maybe_infer_budget(
            budget, goal, product, extracted, platforms)
        if inferred_budget_report is not None:
            budget = (
                f"Recommended: {inferred_budget_report.recommended_budget} "
                f"(Min: {inferred_budget_report.min_budget}, "
                f"(Max: {inferred_budget_report.max_budget})")

        constraints = (f"Tone: {extracted.brand_tone}. "
                       f"Funnel: {extracted.funnel_type}. "
                       f"CTA: {extracted.cta_type}")

        result = self.run(
            product=product,
            goal=goal,
            audience_desc=audience_desc,
            platforms=platforms,
            budget=budget,
            timeline=timeline,
            brand_assets=extracted.visual_identity,
            key_differentiators=extracted.differentiators,
            competitors=extracted.competitive_category,
            constraints=constraints,
            threshold=threshold,
            max_iterations=max_iterations,
            output_dir=output_dir,
        )

        if result is not None and inferred_budget_report is not None:
            result.budget_inference_report = inferred_budget_report

        logger.info(
            "run_from_url completed url=%s total_elapsed=%.3fs",
            url,
            time.perf_counter() - pipeline_start,
        )
        return result

    def run(
        self,
        product: str,
        goal: str,
        audience_desc: str,
        platforms: list[str] | None = None,
        budget: str = DEFAULT_BUDGET_PLACEHOLDER,
        timeline: str = DEFAULT_BUDGET_PLACEHOLDER,
        brand_assets: str = "",
        key_differentiators: str = "",
        competitors: str = "",
        constraints: str = "",
        geography_market: str = "",
        threshold: int = DEFAULT_CONVERGENCE_THRESHOLD,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        historical_data: list[dict[str, Any]] | None = None,
        past_strategies: list[StrategyBrief] | None = None,
        output_dir: str | None = None,
    ) -> PipelineResult | None:
        """Executes the full 12-step orchestration pipeline.

        Args:
            product: Description of the product or service.
            goal: Primary business objective.
            audience_desc: Human-readable audience description.
            platforms: Target advertising platforms.
            budget: Spending budget or placeholder.
            timeline: Campaign timeline or placeholder.
            brand_assets: Brand asset descriptions (logo, colors, etc.).
            key_differentiators: Competitive differentiators.
            competitors: Competitive category or named competitors.
            constraints: Additional creative or compliance constraints.
            geography_market: Geographic or market targeting context.
            threshold: Minimum composite score for GO decisions.
            max_iterations: Maximum evaluate-iterate cycles.
            historical_data: Prior campaign performance records.
            past_strategies: Previously generated strategy briefs.
            output_dir: Root directory for run outputs.

        Returns:
            A ``PipelineResult`` on success, or ``None`` if no variants pass.
        """
        pipeline_start = time.perf_counter()
        if platforms is None:
            platforms = []

        run_output_path = ensure_output_dir(output_dir or "outputs")
        logger.info(
            "run started product=%s goal=%s output=%s",
            product[:50],
            goal,
            run_output_path,
        )

        # STEP 1 -> STRATEGY
        step_start = time.perf_counter()
        logger.info("STEP 1 -> STRATEGY DEFINITION")
        brief = self.strategy_agent.create_brief(
            product,
            goal,
            audience_desc,
            platforms,
            budget,
            timeline,
            brand_assets,
            key_differentiators,
            competitors,
            constraints,
            geography_market,
        )
        save_json(brief, run_output_path / "step_1_strategy.json")
        logger.info(
            "STEP 1 completed elapsed=%.3fs",
            time.perf_counter() - step_start,
        )

        # STEP 2 -> AUDIENCE
        step_start = time.perf_counter()
        logger.info("STEP 2 -> AUDIENCE MODELING")
        personas = self.audience_agent.model_personas(
            product,
            brief,
            audience_desc,
            platforms,
        )
        save_json(personas, run_output_path / "step_2_audience.json")
        logger.info(
            "STEP 2 completed persona_count=%d elapsed=%.3fs",
            len(personas.personas),
            time.perf_counter() - step_start,
        )

        # STEP 3 -> CREATIVE GENERATION
        step_start = time.perf_counter()
        logger.info("STEP 3 -> CREATIVE GENERATION")
        creative_output = self.creative_agent.generate_variants(
            product,
            brief,
            personas,
            platforms,
            constraints,
        )
        save_json(creative_output,
                  run_output_path / "step_3_creative_base.json")
        logger.info(
            "STEP 3 completed variant_count=%d elapsed=%.3fs",
            len(creative_output.variants),
            time.perf_counter() - step_start,
        )

        current_variants = creative_output.variants
        iteration_reports: list[IterationControlReport] = []
        all_evaluations: list[dict[str, Any]] = []

        # STEPS 4-6 -> EVALUATE, AGGREGATE, ITERATE
        readiness_report = self.run_evaluation_loop(
            current_variants=current_variants,
            brief=brief,
            personas=personas,
            brand_assets=brand_assets,
            constraints=constraints,
            max_iterations=max_iterations,
            iteration_reports=iteration_reports,
            all_evaluations_out=all_evaluations,
            output_dir=run_output_path,
        )

        # Filter to approved variants
        approved_variants = [
            current_variants[i]
            for i, decision in enumerate(readiness_report.variant_decisions)
            if decision.status in PASSING_DECISIONS
        ]

        if not approved_variants:
            logger.warning(
                "No variants met the 'GO' threshold. "
                "Falling back to the highest scoring variant for production.")
            best_decision = max(
                readiness_report.variant_decisions,
                key=lambda x: x.final_readiness_score,
            )
            for v in current_variants:
                if v.concept_title == best_decision.concept_title:
                    approved_variants = [v]
                    break

            if not approved_variants:
                logger.error(
                    "Critical failure: Could not find best variant object. total_elapsed=%.3fs",
                    time.perf_counter() - pipeline_start,
                )
                return None

        # STEPS 7-12 -> POST-APPROVAL LINEAR STAGES
        result = self.run_post_approval_stages(
            approved_variants=approved_variants,
            brief=brief,
            personas=personas,
            platforms=platforms,
            brand_assets=brand_assets,
            constraints=constraints,
            geography_market=geography_market,
            budget=budget,
            timeline=timeline,
            readiness_report=readiness_report,
            iteration_reports=iteration_reports,
            all_evaluations=all_evaluations,
            historical_data=historical_data or [],
            past_strategies=past_strategies or [],
            output_dir=run_output_path,
        )

        logger.info(
            "run completed product=%s total_elapsed=%.3fs",
            product[:50],
            time.perf_counter() - pipeline_start,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def maybe_infer_budget(
        self,
        budget: str,
        goal: str,
        product: str,
        extracted: ExtractedInputs,
        platforms: list[str] | None,
    ) -> BudgetInferenceReport | None:
        """Infers a budget via the Budget Agent when budget is TBD.

        Args:
            budget: The current budget string.
            goal: Business objective.
            product: Product description string.
            extracted: Structured data from URL Intelligence.
            platforms: Target platforms.

        Returns:
            A ``BudgetInferenceReport`` if inference was performed,
            otherwise ``None``.
        """
        if budget != DEFAULT_BUDGET_PLACEHOLDER:
            return None

        step_start = time.perf_counter()
        logger.info("STEP 0.5 -> BUDGET INFERENCE")
        report = self.budget_agent.infer_budget(
            goal=goal,
            product=product,
            funnel_type=extracted.funnel_type,
            audience_size="Unknown",
            platforms=platforms or [],
            market_context=extracted.competitive_category,
        )
        logger.info(
            "STEP 0.5 completed elapsed=%.3fs",
            time.perf_counter() - step_start,
        )
        return report

    def evaluate_single_variant(
        self,
        index: int,
        variant: AdScript,
        brief: StrategyBrief,
        personas: AudienceSegments,
        brand_assets: str,
        constraints: str,
    ) -> dict[str, Any]:
        """Runs all four evaluation agents against a single variant.

        Args:
            index: Positional index of the variant.
            variant: The ad script to evaluate.
            brief: The current strategy brief.
            personas: Audience persona profiles.
            brand_assets: Brand asset descriptions.
            constraints: Creative or compliance constraints.

        Returns:
            A dictionary containing the variant index and each agent's
            evaluation result.
        """
        label = variant.concept_title or f"V{index + 1}"
        logger.info("  Evaluating variant: %s", label)

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_clarity = executor.submit(self.clarity_agent.evaluate,
                                             variant, brief, personas)
            future_brand = executor.submit(self.brand_agent.evaluate, variant,
                                           brief, brand_assets, personas)
            future_diagnostics = executor.submit(
                self.diagnostics_agent.evaluate, variant, brief, personas,
                constraints)
            future_attention = executor.submit(self.attention_agent.evaluate,
                                               variant, None, personas)

            clarity = future_clarity.result()
            brand = future_brand.result()
            diagnostics = future_diagnostics.result()
            attention = future_attention.result()

        return {
            "variant_index": index,
            "clarity": clarity,
            "brand": brand,
            "diagnostics": diagnostics,
            "attention": attention,
        }

    def run_evaluation_loop(
        self,
        current_variants: list[AdScript],
        brief: StrategyBrief,
        personas: AudienceSegments,
        brand_assets: str,
        constraints: str,
        max_iterations: int,
        iteration_reports: list[IterationControlReport],
        all_evaluations_out: list[dict[str, Any]],
        output_dir: Path,
    ) -> CompositeReadinessReport:
        """Runs the evaluate-aggregate-iterate loop (steps 4-6).

        Mutates ``current_variants``, ``iteration_reports``, and
        ``all_evaluations_out`` in place. Returns the final readiness
        report once convergence is reached or iterations are exhausted.

        Args:
            current_variants: Mutable list of current ad variants.
            brief: The strategy brief.
            personas: Audience persona profiles.
            brand_assets: Brand asset descriptions.
            constraints: Creative constraints.
            max_iterations: Maximum iteration cycles.
            iteration_reports: Accumulator for iteration reports.
            all_evaluations_out: Accumulator for the latest evaluations.
            output_dir: Run artifact directory.

        Returns:
            The final ``CompositeReadinessReport``.
        """
        readiness_report: CompositeReadinessReport | None = None
        evaluations: list[dict[str, Any]] = []

        for cycle in range(max_iterations + 1):
            logger.info("--- Iteration Cycle %d ---", cycle)
            save_json(
                {
                    "cycle": cycle,
                    "variants": [v.model_dump() for v in current_variants],
                },
                output_dir / f"iteration_{cycle}_variants.json",
            )

            # STEP 4 -> PARALLEL EVALUATION
            step_start = time.perf_counter()
            logger.info("STEP 4 -> PARALLEL EVALUATION")
            worker_count = min(MAX_PARALLEL_EVALUATIONS, len(current_variants))

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        self.evaluate_single_variant,
                        i,
                        variant,
                        brief,
                        personas,
                        brand_assets,
                        constraints,
                    ) for i, variant in enumerate(current_variants)
                ]
                for future in as_completed(futures):
                    evaluations.append(future.result())

            evaluations.sort(key=lambda e: e["variant_index"])
            logger.info(
                "STEP 4 completed variant_count=%d elapsed=%.3fs",
                len(current_variants),
                time.perf_counter() - step_start,
            )

            # Intent evaluation
            step_start = time.perf_counter()
            logger.info("STEP 4b -> INTENT SIMULATION")
            intent_evaluation = self.intent_agent.evaluate(
                current_variants,
                brief,
                personas,
                evaluations,
            )
            logger.info(
                "STEP 4b completed elapsed=%.3fs",
                time.perf_counter() - step_start,
            )

            # STEP 5 -> AGGREGATION & DECISION
            step_start = time.perf_counter()
            logger.info("STEP 5 -> AGGREGATION & DECISION")
            readiness_report = self.scoring_agent.aggregate_and_score(
                current_variants,
                brief,
                evaluations,
                intent_evaluation,
            )
            logger.info(
                "STEP 5 completed elapsed=%.3fs",
                time.perf_counter() - step_start,
            )

            all_converged = all(
                d.status == "GO" for d in readiness_report.variant_decisions)

            if all_converged:
                logger.info("All variants met thresholds (GO). Exiting loop.")
                break

            if cycle == max_iterations:
                logger.info("Max iterations reached.")
                break

            # STEP 6 -> ITERATION
            step_start = time.perf_counter()
            logger.info("STEP 6 -> ITERATION LOOP")
            iteration_report = self.iteration_agent.manage_iteration(
                current_variants,
                brief,
                evaluations,
                readiness_report,
            )
            iteration_reports.append(iteration_report)
            logger.info(
                "STEP 6 completed elapsed=%.3fs",
                time.perf_counter() - step_start,
            )

            current_variants = self.apply_iteration_feedback(
                current_variants,
                readiness_report,
                iteration_report,
                brief,
            )

        # Expose the final evaluations upward.
        all_evaluations_out.clear()
        all_evaluations_out.extend(evaluations)

        if readiness_report is None:
            raise RuntimeError(
                "readiness_report is None after evaluation loop; "
                "this should never happen.")
        return readiness_report

    def apply_iteration_feedback(
        self,
        variants: list[AdScript],
        readiness_report: CompositeReadinessReport,
        iteration_report: IterationControlReport,
        brief: StrategyBrief,
    ) -> list[AdScript]:
        """Refines failing variants using iteration feedback.

        Args:
            variants: The current list of ad variants.
            readiness_report: The latest readiness scoring output.
            iteration_report: The iteration controller's plan.
            brief: The strategy brief for context injection.

        Returns:
            A new list of variants with failing ones refined.
        """
        improved: list[AdScript] = []

        for i, variant in enumerate(variants):
            decision = readiness_report.variant_decisions[i]

            if decision.status not in FAILING_DECISIONS:
                improved.append(variant)
                continue

            plan = next(
                (p for p in iteration_report.variant_plans
                 if p.concept_title == variant.concept_title),
                None,
            )

            if plan is None:
                improved.append(variant)
                continue

            issue_descriptions = [
                f"{issue.issue} ({issue.impact})"
                for issue in plan.prioritized_issues
            ]
            feedback = f"Issues: {issue_descriptions}"
            logger.info("  Refining variant: %s", variant.concept_title)

            prompt = (
                "Role: Surgical Creative Editor / Iteration Controller\n"
                "Objective: Refine the ad variant while PRESERVING "
                "high-performing elements.\n"
                f"Original Variant: {variant.model_dump_json()}\n"
                f"Strategy Brief: {brief.model_dump_json()}\n"
                f"Evaluation Feedback: {feedback}\n\n"
                "INSTRUCTIONS:\n"
                "1. IDENTIFY SUCCESSFUL ELEMENTS: Do not change parts of the "
                "script that\n"
                "already align with the strategy and have zero feedback issues.\n"
                "2. TARGETED MODIFICATION: Only modify scenes or copy explicitly "
                "mentioned in the feedback.\n"
                "3. SCHEMA COMPLIANCE: Return the improved script in the exact "
                "same AdScript schema.\n")
            refined = self.creative_agent.generate(
                prompt,
                response_schema=AdScript,
            )
            improved.append(refined)

        return improved

    # ------------------------------------------------------------------
    # Pipeline step methods (7-12)
    # ------------------------------------------------------------------

    def _run_step_7_adaptation(
        self,
        approved_variants: list[AdScript],
        brief: StrategyBrief,
        personas: AudienceSegments,
        platforms: list[str],
        output_dir: Path,
    ) -> list[PlatformVariant]:
        """STEP 7: Platform Adaptation."""
        step_start = time.perf_counter()
        logger.info("STEP 7 -> PLATFORM ADAPTATION")
        all_adaptations: list[PlatformVariant] = []
        for variant in approved_variants:
            report = self.adaptation_agent.adapt(
                variant, brief, personas, platforms)
            for va in report.variant_adaptations:
                all_adaptations.extend(va.adaptations)
        save_json(all_adaptations, output_dir / "step_7_adaptations.json")
        logger.info(
            "STEP 7 completed adaptation_count=%d elapsed=%.3fs",
            len(all_adaptations),
            time.perf_counter() - step_start,
        )
        return all_adaptations

    def _run_step_8_compliance(
        self,
        approved_variants: list[AdScript],
        brief: StrategyBrief,
        platforms: list[str],
        brand_assets: str,
        geography_market: str,
        constraints: str,
        output_dir: Path,
    ) -> ComplianceRiskReport:
        """STEP 8: Compliance Check."""
        step_start = time.perf_counter()
        logger.info("STEP 8 -> COMPLIANCE CHECK")
        report = self.compliance_agent.validate_compliance(
            approved_variants, brief, platforms, brand_assets,
            geography_market, constraints)
        save_json(report, output_dir / "step_8_compliance.json")
        logger.info(
            "STEP 8 completed status=%s elapsed=%.3fs",
            report.overall_status,
            time.perf_counter() - step_start,
        )
        return report

    def _run_step_9_production(
        self,
        approved_variants: list[AdScript],
        all_adaptations: list[PlatformVariant],
        constraints: str,
        brand_assets: str,
        output_dir: Path,
    ) -> AssetProductionReport:
        """STEP 9: Asset Production Planning."""
        step_start = time.perf_counter()
        logger.info("STEP 9 -> ASSET PRODUCTION")
        report = self.production_agent.plan_production(
            approved_variants, all_adaptations, constraints, brand_assets)
        save_json(report, output_dir / "step_9_production.json")
        logger.info(
            "STEP 9 completed variant_count=%d elapsed=%.3fs",
            len(report.production_variants),
            time.perf_counter() - step_start,
        )
        return report

    def _run_step_10_validation(
        self,
        approved_variants: list[AdScript],
        brief: StrategyBrief,
        all_evaluations: list[dict[str, Any]],
        iteration_reports: list[IterationControlReport],
        output_dir: Path,
    ) -> ExternalValidationPlan:
        """STEP 10: Human / External Validation Design."""
        step_start = time.perf_counter()
        logger.info("STEP 10 -> HUMAN / EXTERNAL VALIDATION")
        latest = iteration_reports[-1] if iteration_reports else None
        if latest is None:
            latest = IterationControlReport(
                variant_plans=[],
                cycle_count=0,
                global_refinement_strategy="Passed on first pass.",
            )
        plan = self.validation_agent.design_validation(
            approved_variants, brief, all_evaluations, latest)
        save_json(plan, output_dir / "step_10_validation.json")
        logger.info(
            "STEP 10 completed elapsed=%.3fs",
            time.perf_counter() - step_start,
        )
        return plan

    def _run_step_11_deployment(
        self,
        approved_variants: list[AdScript],
        brief: StrategyBrief,
        platforms: list[str],
        budget: str,
        timeline: str,
        geography_market: str,
        output_dir: Path,
    ) -> DeploymentExperimentationReport:
        """STEP 11: Deployment & Experimentation Planning."""
        step_start = time.perf_counter()
        logger.info("STEP 11 -> DEPLOYMENT & EXPERIMENTATION")
        report = self.deployment_agent.plan_deployment(
            approved_variants, brief, platforms, budget, timeline,
            geography_market)
        save_json(report, output_dir / "step_11_deployment.json")
        logger.info(
            "STEP 11 completed elapsed=%.3fs",
            time.perf_counter() - step_start,
        )
        return report

    def _run_step_12_learning(
        self,
        historical_data: list[dict[str, Any]],
        validation_plan: ExternalValidationPlan,
        all_evaluations: list[dict[str, Any]],
        past_strategies: list[StrategyBrief],
        brief: StrategyBrief,
        output_dir: Path,
    ) -> KnowledgeLearningReport:
        """STEP 12: Learning & System Improvement."""
        step_start = time.perf_counter()
        logger.info("STEP 12 -> LEARNING & SYSTEM IMPROVEMENT")
        report = self.learning_agent.capture_learnings(
            historical_data, validation_plan, all_evaluations,
            past_strategies + [brief])
        save_json(report, output_dir / "step_12_learning.json")
        logger.info(
            "STEP 12 completed elapsed=%.3fs",
            time.perf_counter() - step_start,
        )
        return report

    def _run_step_13_video(
        self,
        approved_variants: list[AdScript],
        brief: StrategyBrief,
        production_report: AssetProductionReport,
        output_dir: Path,
    ) -> tuple[list[VideoGenerationResult], list[str]]:
        """STEP 13: Multi-Variant Video Generation."""
        step_start = time.perf_counter()
        logger.info("STEP 13 -> MULTI-VARIANT VIDEO GENERATION (Veo 3.1)")
        video_results: list[VideoGenerationResult] = []
        video_paths: list[str] = []

        for i, variant in enumerate(approved_variants[:3]):
            safe_product = re.sub(r"[^a-zA-Z0-9_\-]", "_",
                                  brief.product_name)[:20]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"ad_{safe_product}_{i}_{timestamp}.mp4"
            video_output_path = output_dir / video_filename

            logger.info(
                "  Generating video for variant %d: %s",
                i + 1,
                video_output_path,
            )

            prod_plan = (
                production_report.production_variants[i]
                if i < len(production_report.production_variants)
                else production_report.production_variants[0]
            )

            result = self.video_agent.generate_video(
                script=variant,
                production_plan=prod_plan,
                output_path=str(video_output_path),
            )

            if result:
                video_results.append(result)
                video_paths.append(str(video_output_path.absolute()))
                logger.info("  Video saved to %s", video_output_path)

        logger.info(
            "STEP 13 completed video_count=%d elapsed=%.3fs",
            len(video_results),
            time.perf_counter() - step_start,
        )
        return video_results, video_paths

    def run_post_approval_stages(
        self,
        approved_variants: list[AdScript],
        brief: StrategyBrief,
        personas: AudienceSegments,
        platforms: list[str],
        brand_assets: str,
        constraints: str,
        geography_market: str,
        budget: str,
        timeline: str,
        readiness_report: CompositeReadinessReport,
        iteration_reports: list[IterationControlReport],
        all_evaluations: list[dict[str, Any]],
        historical_data: list[dict[str, Any]],
        past_strategies: list[StrategyBrief],
        output_dir: Path,
    ) -> PipelineResult:
        """Executes linear post-approval stages 7 through 13.

        Args:
            approved_variants: Variants that passed the GO gate.
            brief: Strategy brief.
            personas: Audience personas.
            platforms: Target platforms.
            brand_assets: Brand asset descriptions.
            constraints: Creative constraints.
            geography_market: Geographic targeting context.
            budget: Budget string.
            timeline: Timeline string.
            readiness_report: Final readiness report from iteration loop.
            iteration_reports: All iteration reports.
            all_evaluations: Latest evaluation results.
            historical_data: Prior campaign performance data.
            past_strategies: Previously generated strategy briefs.
            output_dir: Run artifact directory.

        Returns:
            A fully populated ``PipelineResult``.
        """
        all_adaptations = self._run_step_7_adaptation(
            approved_variants, brief, personas, platforms, output_dir)
        compliance_report = self._run_step_8_compliance(
            approved_variants, brief, platforms, brand_assets,
            geography_market, constraints, output_dir)
        production_report = self._run_step_9_production(
            approved_variants, all_adaptations, constraints, brand_assets,
            output_dir)
        validation_plan = self._run_step_10_validation(
            approved_variants, brief, all_evaluations, iteration_reports,
            output_dir)
        deployment_report = self._run_step_11_deployment(
            approved_variants, brief, platforms, budget, timeline,
            geography_market, output_dir)
        learning_report = self._run_step_12_learning(
            historical_data, validation_plan, all_evaluations, past_strategies,
            brief, output_dir)
        video_results, video_paths = self._run_step_13_video(
            approved_variants, brief, production_report, output_dir)

        result = PipelineResult(
            strategy=brief,
            personas=personas,
            approved_variants=approved_variants,
            readiness_report=readiness_report,
            iteration_history=iteration_reports,
            platform_adaptations=all_adaptations,
            compliance_report=compliance_report,
            production_report=production_report,
            validation_plan=validation_plan,
            deployment_report=deployment_report,
            learning_report=learning_report,
            video_results=video_results,
            video_paths=video_paths,
        )
        save_json(result, output_dir / "pipeline_final_report.json")
        return result
