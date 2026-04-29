# mypy: disable-error-code="attr-defined"
"""Integration tests for the OrchestratorPipeline step-by-step logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from google import genai

from ads_ai.agents.models import (
    AdScript,
    AssetProductionReport,
    AssetProductionVariant,
    AudienceSegments,
    ComplianceRiskReport,
    CompositeReadinessReport,
    CreativeVariants,
    DeploymentExperimentationReport,
    ExternalValidationPlan,
    IterationControlReport,
    KnowledgeLearningReport,
    PlatformAdaptationReport,
    Scene,
    ShotSceneDesign,
    StrategyBrief,
    VideoGenerationResult,
)
from ads_ai.pipeline import OrchestratorPipeline


class TestOrchestratorPipeline:
    """Tests for the OrchestratorPipeline execution flow."""

    @pytest.fixture
    def mock_orchestrator(self) -> OrchestratorPipeline:
        """Provides an orchestrator with all sub-agents mocked."""
        mock_client = MagicMock(spec=genai.Client)
        with patch("ads_ai.pipeline.StrategyAgent"), \
             patch("ads_ai.pipeline.AudienceAgent"), \
             patch("ads_ai.pipeline.CreativeAgent"), \
             patch("ads_ai.pipeline.MessageClarityAgent"), \
             patch("ads_ai.pipeline.BrandLinkageAgent"), \
             patch("ads_ai.pipeline.DiagnosticsAgent"), \
             patch("ads_ai.pipeline.AttentionHeuristicAgent"), \
             patch("ads_ai.pipeline.IntentSimulationAgent"), \
             patch("ads_ai.pipeline.ScoringAgent"), \
             patch("ads_ai.pipeline.PlatformAdaptationAgent"), \
             patch("ads_ai.pipeline.IterationControllerAgent"), \
             patch("ads_ai.pipeline.ExternalValidationAgent"), \
             patch("ads_ai.pipeline.KnowledgeLearningAgent"), \
             patch("ads_ai.pipeline.ComplianceRiskAgent"), \
             patch("ads_ai.pipeline.AssetProductionAgent"), \
             patch("ads_ai.pipeline.DeploymentExperimentationAgent"), \
             patch("ads_ai.pipeline.VideoGenerationAgent"), \
             patch("ads_ai.pipeline.URLIntelligenceAgent"), \
             patch("ads_ai.pipeline.BudgetInferenceAgent"):
            return OrchestratorPipeline(mock_client)

    def test_pipeline_run_success_first_pass(
            self, mock_orchestrator: OrchestratorPipeline) -> None:
        """Should complete the pipeline if variants pass on the first iteration."""
        from ads_ai.agents.models import KPI, MessageStrategy, Persona, PreReleaseTargets

        # Use real model instances for strategy/audience (accessed in post-approval stages)
        mock_brief = StrategyBrief(
            product_name="TestProduct",
            ad_intent="Brand awareness",
            intent_explanation="Test",
            kpis=[
                KPI(name="CTR",
                    target_value="1%",
                    measurement_method="Impressions")
            ],
            pre_release_targets=PreReleaseTargets(
                clarity_score_target=80,
                brand_linkage_score_target=75,
                hook_strength_threshold="High",
                message_retention_likelihood="80%",
                simulated_intent_score="80",
            ),
            audience_personas=[
                Persona(demographics="All",
                        motivation="Value",
                        pain_point="None",
                        buying_trigger="Deal",
                        likely_objection="Price")
            ],
            message_strategy=MessageStrategy(value_proposition="Best",
                                             message_pillars=["Quality"],
                                             tone="Professional",
                                             emotional_trigger="Trust"),
            creative_constraints={},
            decision_thresholds={},
        )
        mock_orchestrator.strategy_agent.create_brief.return_value = mock_brief
        from ads_ai.agents.models import (
            KPI,
            AudienceSegments,
            CrossPersonaInsights,
            MessageStrategy,
            Persona,
            PreReleaseTargets,
        )
        cross_insights = CrossPersonaInsights(
            common_strengths=[],
            common_failure_points=[],
            highest_performing_persona="",
            lowest_performing_persona="",
        )
        mock_orchestrator.audience_agent.model_personas.return_value = AudienceSegments(
            personas=[],
            cross_persona_insights=cross_insights,
            optimization_recommendations=[])
        # 3. Creative
        mock_orchestrator.creative_agent.generate_variants.return_value = CreativeVariants(
            variants=[
                AdScript(
                    concept_title="Test",
                    core_idea="Idea",
                    hook="H",
                    script_scenes=[
                        Scene(description="D", visual_cues="V", dialogue_vo="D")
                    ],
                    brand_integration="Logo.",
                    cta="Buy.",
                    video_prompt="4k.",
                    variant_name="V1",
                )
            ])

        # 5. Scoring & Decision - GO decision
        mock_report = MagicMock(spec=CompositeReadinessReport)
        mock_report.variant_decisions = [
            MagicMock(
                status="GO",
                final_readiness_score=95.0,
                concept_title="Test",
            )
        ]
        mock_orchestrator.scoring_agent.aggregate_and_score.return_value = mock_report

        # 7-12 Post-approval
        mock_orchestrator.adaptation_agent.adapt.return_value = PlatformAdaptationReport(
            variant_adaptations=[])
        mock_orchestrator.compliance_agent.validate_compliance.return_value = ComplianceRiskReport(
            variant_reports=[], overall_status="PASS")
        mock_orchestrator.production_agent.plan_production.return_value = AssetProductionReport(
            production_variants=[
                AssetProductionVariant(
                    concept_title="Test",
                    required_assets=[],
                    production_scenes=[
                        ShotSceneDesign(shot_number=1,
                                        visual_description="",
                                        audio_description="",
                                        technical_specs="")
                    ],
                    estimated_production_complexity="Low",
                )
            ])
        mock_orchestrator.validation_agent.design_validation.return_value = (
            ExternalValidationPlan(
                experiment_designs=[],
                metric_mapping=[],
                variant_validation_results=[])
        )
        mock_orchestrator.deployment_agent.plan_deployment.return_value = (
            DeploymentExperimentationReport(
                launch_plans=[], test_timeline="", scaling_triggers=[])
        )
        mock_orchestrator.learning_agent.capture_learnings.return_value = (
            KnowledgeLearningReport(
                campaign_records=[],
                prediction_reality_reports=[],
                key_patterns=[],
                agent_performance_diagnostics=[])
        )
        mock_orchestrator.video_agent.generate_video.return_value = (
            VideoGenerationResult(video_file_path="test.mp4")
        )

        result = mock_orchestrator.run(
            product="Product",
            goal="Goal",
            audience_desc="Audience",
        )

        assert result is not None
        assert mock_orchestrator.strategy_agent.create_brief.called
        assert mock_orchestrator.iteration_agent.manage_iteration.call_count == 0

    def test_pipeline_iteration_limit(
            self, mock_orchestrator: OrchestratorPipeline) -> None:
        """Should terminate after max iterations even if no variants pass."""
        from ads_ai.agents.models import (
            KPI,
            CrossPersonaInsights,
            MessageStrategy,
            Persona,
            PreReleaseTargets,
        )
        cross_insights = CrossPersonaInsights(
            common_strengths=[],
            common_failure_points=[],
            highest_performing_persona="",
            lowest_performing_persona="",
        )
        mock_orchestrator.strategy_agent.create_brief.return_value = StrategyBrief(
            product_name="TestProduct",
            ad_intent="Brand awareness",
            intent_explanation="Test",
            kpis=[
                KPI(name="CTR",
                    target_value="1%",
                    measurement_method="Impressions")
            ],
            pre_release_targets=PreReleaseTargets(
                clarity_score_target=80,
                brand_linkage_score_target=75,
                hook_strength_threshold="High",
                message_retention_likelihood="80%",
                simulated_intent_score="80",
            ),
            audience_personas=[
                Persona(demographics="All",
                        motivation="Value",
                        pain_point="None",
                        buying_trigger="Deal",
                        likely_objection="Price")
            ],
            message_strategy=MessageStrategy(value_proposition="Best",
                                             message_pillars=["Quality"],
                                             tone="Professional",
                                             emotional_trigger="Trust"),
            creative_constraints={},
            decision_thresholds={},
        )
        mock_orchestrator.audience_agent.model_personas.return_value = AudienceSegments(
            personas=[],
            cross_persona_insights=cross_insights,
            optimization_recommendations=[])
        mock_orchestrator.creative_agent.generate_variants.return_value = CreativeVariants(
            variants=[
                AdScript(
                    concept_title="Test",
                    core_idea="Idea",
                    hook="H",
                    script_scenes=[
                        Scene(description="D", visual_cues="V", dialogue_vo="D")
                    ],
                    brand_integration="Logo.",
                    cta="Buy.",
                    video_prompt="4k.",
                    variant_name="V1",
                )
            ])

        mock_report = MagicMock(spec=CompositeReadinessReport)
        mock_report.variant_decisions = [
            MagicMock(
                status="NO-GO",
                final_readiness_score=30.0,
                concept_title="Test",
            )
        ]
        mock_orchestrator.scoring_agent.aggregate_and_score.return_value = mock_report

        mock_orchestrator.iteration_agent.manage_iteration.return_value = IterationControlReport(
            variant_plans=[],
            cycle_count=1,
            global_refinement_strategy="Improve hook.",
        )
        mock_orchestrator.creative_agent.refine_variants.return_value = [
            AdScript(
                concept_title="Test",
                core_idea="Idea",
                hook="H",
                script_scenes=[
                    Scene(description="D", visual_cues="V", dialogue_vo="D")
                ],
                brand_integration="Logo.",
                cta="Buy.",
                video_prompt="4k.",
                variant_name="V1",
            )
        ]

        # Post-approval stages (called after fallback selection)
        mock_orchestrator.adaptation_agent.adapt.return_value = PlatformAdaptationReport(
            variant_adaptations=[])
        mock_orchestrator.compliance_agent.validate_compliance.return_value = ComplianceRiskReport(
            variant_reports=[], overall_status="PASS")
        mock_orchestrator.production_agent.plan_production.return_value = AssetProductionReport(
            production_variants=[
                AssetProductionVariant(
                    concept_title="Test",
                    required_assets=[],
                    production_scenes=[
                        ShotSceneDesign(shot_number=1,
                                        visual_description="",
                                        audio_description="",
                                        technical_specs="")
                    ],
                    estimated_production_complexity="Low",
                )
            ])
        mock_orchestrator.validation_agent.design_validation.return_value = (
            ExternalValidationPlan(
                experiment_designs=[],
                metric_mapping=[],
                variant_validation_results=[])
        )
        mock_orchestrator.deployment_agent.plan_deployment.return_value = (
            DeploymentExperimentationReport(
                launch_plans=[], test_timeline="", scaling_triggers=[])
        )
        mock_orchestrator.learning_agent.capture_learnings.return_value = (
            KnowledgeLearningReport(
                campaign_records=[],
                prediction_reality_reports=[],
                key_patterns=[],
                agent_performance_diagnostics=[])
        )
        mock_orchestrator.video_agent.generate_video.return_value = (
            VideoGenerationResult(video_file_path="test.mp4")
        )

        result = mock_orchestrator.run(product="Product",
                                       goal="Goal",
                                       audience_desc="Audience",
                                       max_iterations=2)

        assert result is not None
        # Should evaluate 3 times (Initial + 2 iterations)
        assert mock_orchestrator.scoring_agent.aggregate_and_score.call_count == 3
        # Should iterate 2 times
        assert mock_orchestrator.iteration_agent.manage_iteration.call_count == 2
