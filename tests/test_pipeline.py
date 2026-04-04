"""Integration tests for the OrchestratorPipeline step-by-step logic."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

from google import genai
import pytest

from ads_ai.pipeline import OrchestratorPipeline
from ads_ai.agents.models import (
    StrategyBrief,
    AudienceSegments,
    CreativeVariants,
    AdScript,
    CompositeReadinessReport,
    IterationControlReport,
    PlatformAdaptationReport,
    ComplianceRiskReport,
    AssetProductionReport,
    ExternalValidationPlan,
    DeploymentExperimentationReport,
    KnowledgeLearningReport,
)


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

    def test_pipeline_run_success_first_pass(self, mock_orchestrator: OrchestratorPipeline) -> None:
        """Should complete the pipeline if variants pass on the first iteration."""
        # 1. Strategy
        mock_orchestrator.strategy_agent.create_brief.return_value = MagicMock(spec=StrategyBrief)
        # 2. Audience
        mock_orchestrator.audience_agent.model_personas.return_value = MagicMock(spec=AudienceSegments)
        # 3. Creative
        mock_variant = AdScript(concept_title="Test", hook="H", body_copy="B", call_to_action="C", visual_cues="V")
        mock_orchestrator.creative_agent.generate_variants.return_value = CreativeVariants(variants=[mock_variant])

        # 4. Evaluation (Step 4 is parallel, but handled via internals)
        # 5. Scoring & Decision
        mock_decision = MagicMock()
        mock_decision.final_decision = "GO"
        mock_report = MagicMock(spec=CompositeReadinessReport)
        mock_report.variant_decisions = [mock_decision]
        mock_orchestrator.scoring_agent.aggregate_and_score.return_value = mock_report

        # 7-12 Post-approval
        mock_orchestrator.adaptation_agent.adapt.return_value = MagicMock(spec=PlatformAdaptationReport)
        mock_orchestrator.compliance_agent.validate_compliance.return_value = MagicMock(spec=ComplianceRiskReport)
        mock_orchestrator.production_agent.plan_production.return_value = MagicMock(spec=AssetProductionReport)
        mock_orchestrator.validation_agent.design_validation.return_value = MagicMock(spec=ExternalValidationPlan)
        mock_orchestrator.deployment_agent.plan_deployment.return_value = MagicMock(spec=DeploymentExperimentationReport)
        mock_orchestrator.learning_agent.capture_learnings.return_value = MagicMock(spec=KnowledgeLearningReport)

        result = mock_orchestrator.run(
            product="Product",
            goal="Goal",
            audience_desc="Audience",
        )

        assert result is not None
        assert mock_orchestrator.strategy_agent.create_brief.called
        assert mock_orchestrator.iteration_agent.manage_iteration.call_count == 0

    def test_pipeline_iteration_limit(self, mock_orchestrator: OrchestratorPipeline) -> None:
        """Should terminate after max iterations even if no variants pass."""
        # Setup mocks to always return NO-GO
        mock_orchestrator.strategy_agent.create_brief.return_value = MagicMock(spec=StrategyBrief)
        mock_orchestrator.audience_agent.model_personas.return_value = MagicMock(spec=AudienceSegments)
        mock_variant = AdScript(concept_title="Test", hook="H", body_copy="B", call_to_action="C", visual_cues="V")
        mock_orchestrator.creative_agent.generate_variants.return_value = CreativeVariants(variants=[mock_variant])

        mock_decision = MagicMock()
        mock_decision.final_decision = "NO-GO"
        mock_report = MagicMock(spec=CompositeReadinessReport)
        mock_report.variant_decisions = [mock_decision]
        mock_orchestrator.scoring_agent.aggregate_and_score.return_value = mock_report

        mock_orchestrator.iteration_agent.manage_iteration.return_value = MagicMock(spec=IterationControlReport)
        mock_orchestrator.creative_agent.generate.return_value = mock_variant

        result = mock_orchestrator.run(
            product="Product",
            goal="Goal",
            audience_desc="Audience",
            max_iterations=2
        )

        assert result is None
        # Should evaluate 3 times (Initial + 2 iterations)
        assert mock_orchestrator.scoring_agent.aggregate_and_score.call_count == 3
        # Should iterate 2 times
        assert mock_orchestrator.iteration_agent.manage_iteration.call_count == 2
