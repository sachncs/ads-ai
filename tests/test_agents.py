"""Unit tests for the BaseAgent and shared agent infrastructure."""

from __future__ import annotations

import time
from unittest.mock import MagicMock
from unittest.mock import patch

from google import genai
import pytest
from pydantic import BaseModel

from ads_ai.agents.base import BaseAgent


class MockSchema(BaseModel):
    """Simple Pydantic model for testing JSON validation."""
    field_a: str
    field_b: int


class TestBaseAgent:
    """Tests for the BaseAgent shared logic."""

    def test_init_defaults(self) -> None:
        """Should use default_text_model from settings if not provided."""
        mock_client = MagicMock(spec=genai.Client)
        agent = BaseAgent(mock_client)
        assert agent.client == mock_client
        assert agent.model_name == "gemini-3.1-pro-preview"

    def test_init_override(self) -> None:
        """Should use provided model name."""
        mock_client = MagicMock(spec=genai.Client)
        agent = BaseAgent(mock_client, model_name="test-model")
        assert agent.model_name == "test-model"

    def test_to_json_dict_recursive(self) -> None:
        """Should recursively convert objects to JSON-compatible dicts."""
        mock_client = MagicMock(spec=genai.Client)
        agent = BaseAgent(mock_client)

        data = {
            "pydantic": MockSchema(field_a="hello", field_b=123),
            "list": [MockSchema(field_a="world", field_b=456)],
            "nested": {
                "val": MockSchema(field_a="deep", field_b=789)
            }
        }

        result = agent.to_json_dict(data)
        assert result["pydantic"]["field_a"] == "hello"
        assert result["pydantic"]["field_b"] == 123
        assert result["list"][0]["field_a"] == "world"
        assert result["nested"]["val"]["field_b"] == 789

    def test_generate_text_success(self) -> None:
        """Should return raw text when no schema is provided."""
        mock_client = MagicMock(spec=genai.Client)
        mock_response = MagicMock()
        mock_response.text = "Raw response text"
        mock_client.models.generate_content.return_value = mock_response

        agent = BaseAgent(mock_client)
        result = agent.generate("Test prompt")

        assert result == "Raw response text"
        mock_client.models.generate_content.assert_called_once()

    def test_generate_json_success(self) -> None:
        """Should return validated Pydantic model when schema is provided."""
        mock_client = MagicMock(spec=genai.Client)
        mock_response = MagicMock()
        mock_response.text = '{"field_a": "validated", "field_b": 99}'
        mock_client.models.generate_content.return_value = mock_response

        agent = BaseAgent(mock_client)
        result = agent.generate("Test prompt", response_schema=MockSchema)

        assert isinstance(result, MockSchema)
        assert result.field_a == "validated"
        assert result.field_b == 99

    @patch("time.sleep", return_value=None)
    def test_generate_retry_on_503(self, mock_sleep: MagicMock) -> None:
        """Should retry on 503 errors and eventually succeed or fail."""
        mock_client = MagicMock(spec=genai.Client)

        # Simulate two 503 errors followed by success
        err_msg = "503 Service Unavailable"
        mock_client.models.generate_content.side_effect = [
            Exception(err_msg),
            Exception(err_msg),
            MagicMock(text='{"field_a": "after_retry", "field_b": 1}')
        ]

        agent = BaseAgent(mock_client)
        # Use low retries for testing if needed, though default is 3
        result = agent.generate("Test prompt", response_schema=MockSchema)

        assert result.field_a == "after_retry"
        assert mock_client.models.generate_content.call_count == 3
        assert mock_sleep.call_count == 2

    def test_generate_fail_non_503(self) -> None:
        """Should not retry on non-503 errors."""
        mock_client = MagicMock(spec=genai.Client)
        mock_client.models.generate_content.side_effect = Exception("400 Bad Request")

        agent = BaseAgent(mock_client)
        with pytest.raises(Exception, match="400 Bad Request"):
            agent.generate("Test prompt")

        assert mock_client.models.generate_content.call_count == 1

    def test_generate_max_retries_exceeded(self) -> None:
        """Should raise the last exception if all retries fail."""
        mock_client = MagicMock(spec=genai.Client)
        mock_client.models.generate_content.side_effect = Exception("503 Still failing")

        with patch("time.sleep", return_value=None):
            agent = BaseAgent(mock_client)
            with pytest.raises(Exception, match="503 Still failing"):
                agent.generate("Test prompt")

        # Default max_retries is 3
        assert mock_client.models.generate_content.call_count == 3


class TestSpecializedAgents:
    """Correctness tests for higher-level agent prompt synthesis."""

    def test_strategy_agent_create_brief(self) -> None:
        """StrategyAgent should synthesize inputs into a StrategyBrief."""
        from ads_ai.agents.strategy import StrategyAgent
        from ads_ai.agents.models import (
            StrategyBrief,
            KPI,
            PreReleaseTargets,
            Persona,
            MessageStrategy,
        )

        mock_client = MagicMock(spec=genai.Client)
        mock_brief = StrategyBrief(
            product_name="Test Product",
            ad_intent="Test Intent",
            intent_explanation="Test explanation",
            kpis=[KPI(name="CTR", target_value="1%", measurement_method="Impressions")],
            pre_release_targets=PreReleaseTargets(
                clarity_score_target=80,
                brand_linkage_score_target=75,
                hook_strength_threshold="High",
                message_retention_likelihood="80%",
                simulated_intent_score="80",
            ),
            audience_personas=[
                Persona(
                    demographics="Adults 25-54",
                    motivation="Value",
                    pain_point="Price",
                    buying_trigger="Discount",
                    likely_objection="Too expensive",
                )
            ],
            message_strategy=MessageStrategy(
                value_proposition="Best value",
                message_pillars=["Price", "Quality"],
                tone="Professional",
                emotional_trigger="Trust",
            ),
            creative_constraints={},
            decision_thresholds={"clarity": 0.7},
        )

        mock_response = MagicMock()
        mock_response.text = mock_brief.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        agent = StrategyAgent(mock_client)
        result = agent.create_brief(
            product="Product",
            goal="Goal",
            audience_desc="Audience",
            platforms=["Meta"],
            budget="1000",
            timeline="1 month",
            brand_assets="Logo",
            key_differentiators="Quality",
            competitors="Others",
            constraints="None",
            geography_market="US",
        )

        assert isinstance(result, StrategyBrief)
        assert result.product_name == "Test Product"
        assert result.ad_intent == "Test Intent"

    def test_creative_agent_generate_variants(self) -> None:
        """CreativeAgent should generate multiple ad script variants."""
        from ads_ai.agents.creative import CreativeAgent
        from ads_ai.agents.models import CreativeVariants, AdScript, StrategyBrief, AudienceSegments, Scene

        mock_client = MagicMock(spec=genai.Client)
        mock_variants = CreativeVariants(
            variants=[
                AdScript(
                    concept_title="V1",
                    core_idea="Idea 1",
                    hook="Hook 1",
                    script_scenes=[Scene(description="S1", visual_cues="V1", dialogue_vo="D1")],
                    brand_integration="Logo at end.",
                    cta="Buy now.",
                    video_prompt="Cinematic 4k shot.",
                    variant_name="Variant 1",
                ),
                AdScript(
                    concept_title="V2",
                    core_idea="Idea 2",
                    hook="Hook 2",
                    script_scenes=[Scene(description="S2", visual_cues="V2", dialogue_vo="D2")],
                    brand_integration="Logo at start.",
                    cta="Shop now.",
                    video_prompt="Cinematic 4k render.",
                    variant_name="Variant 2",
                ),
            ]
        )

        mock_response = MagicMock()
        mock_response.text = mock_variants.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        agent = CreativeAgent(mock_client)
        brief = MagicMock(spec=StrategyBrief)
        brief.model_dump_json.return_value = "{}"
        personas = MagicMock(spec=AudienceSegments)
        personas.model_dump_json.return_value = "{}"

        result = agent.generate_variants(
            product="Product",
            strategy=brief,
            personas=personas,
            platforms=["Meta"],
        )

        assert len(result.variants) == 2
        assert result.variants[0].concept_title == "V1"

    def test_scoring_agent_aggregation(self) -> None:
        """ScoringAgent should aggregate multiple evaluations into a CompositeReadinessReport."""
        from ads_ai.agents.scoring import ScoringAgent
        from ads_ai.agents.models import CompositeReadinessReport, AdScript, CategoryScore

        mock_client = MagicMock(spec=genai.Client)
        mock_report = CompositeReadinessReport(
            target_kpis=["ROI"],
            variant_decisions=[{
                "concept_title": "V1",
                "final_readiness_score": 85.0,
                "is_ready": True,
                "status": "GO",
                "confidence": 0.9,
                "primary_strength": "Hook",
                "primary_blocker": None,
                "category_breakdown": [
                    CategoryScore(category="clarity", raw_score=85.0, weight=0.3, weighted_contribution=25.5),
                    CategoryScore(category="brand", raw_score=85.0, weight=0.3, weighted_contribution=25.5),
                ],
            }],
            best_overall_variant="V1",
            system_readiness_flag=True,
            strategic_trade_offs=[],
            critical_risks=[],
        )

        mock_response = MagicMock()
        mock_response.text = mock_report.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        agent = ScoringAgent(mock_client)
        from ads_ai.agents.models import Scene
        result = agent.aggregate_and_score(
            variants=[AdScript(
                concept_title="V1", core_idea="I", hook="H",
                script_scenes=[Scene(description="D", visual_cues="V", dialogue_vo="D")],
                brand_integration="Logo.", cta="Buy.", video_prompt="4k.", variant_name="V1",
            )],
            strategy=MagicMock(),
            evaluations=[{"clarity": MagicMock()}],
            intent_evaluation=MagicMock(),
        )

        assert result.system_readiness_flag is True

    def test_iteration_controller_management(self) -> None:
        """IterationControllerAgent should produce an IterationControlReport."""
        from ads_ai.agents.iteration import IterationControllerAgent
        from ads_ai.agents.models import (
            IterationControlReport,
            IterationDirective,
            PrioritizedIssue,
            IterationAction,
            VariantIterationPlan,
        )

        mock_client = MagicMock(spec=genai.Client)
        mock_report = IterationControlReport(
            variant_plans=[
                VariantIterationPlan(
                    concept_title="V1",
                    iteration_directives=[
                        IterationDirective(
                            category="Clarity",
                            severity="High",
                            instruction="Rewrite hook",
                            expected_outcome="+15 clarity",
                        )
                    ],
                    prioritized_issues=[
                        PrioritizedIssue(issue="Weak hook", source_agent="Clarity", impact="High")
                    ],
                    suggested_actions=[
                        IterationAction(action="Rewrite hook line", target_scene="Hook", rationale="Clarity")
                    ],
                )
            ],
            cycle_count=1,
            global_refinement_strategy="Focus on hook clarity.",
        )

        mock_response = MagicMock()
        mock_response.text = mock_report.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        agent = IterationControllerAgent(mock_client)
        result = agent.manage_iteration(
            variants=[],
            strategy=MagicMock(),
            evaluations=[],
            readiness_report=MagicMock(),
        )

        assert result.cycle_count == 1
        assert result.global_refinement_strategy == "Focus on hook clarity."
