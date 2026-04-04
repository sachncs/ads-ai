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

        result = agent._to_json_dict(data)
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
        from ads_ai.agents.models import StrategyBrief

        mock_client = MagicMock(spec=genai.Client)
        mock_brief = StrategyBrief(
            product_name="Test Product",
            value_proposition="Test Prop",
            target_audience="Test Audience",
            strategic_pillars=["Pillar 1"],
            creative_direction="Test Direction",
            key_messages=["Msg 1"],
            cta_directives="CTA 1",
            tone_and_manner="Tone 1",
            platform_strategies={"Meta": "Strategy 1"},
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
        )

        assert isinstance(result, StrategyBrief)
        assert result.product_name == "Test Product"
        assert "Meta" in result.platform_strategies

    def test_creative_agent_generate_variants(self) -> None:
        """CreativeAgent should generate multiple ad script variants."""
        from ads_ai.agents.creative import CreativeAgent
        from ads_ai.agents.models import CreativeVariants, AdScript, StrategyBrief, AudienceSegments

        mock_client = MagicMock(spec=genai.Client)
        mock_variants = CreativeVariants(
            variants=[
                AdScript(concept_title="V1", hook="H1", body_copy="B1", call_to_action="C1", visual_cues="V1"),
                AdScript(concept_title="V2", hook="H2", body_copy="B2", call_to_action="C2", visual_cues="V2"),
            ]
        )

        mock_response = MagicMock()
        mock_response.text = mock_variants.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        agent = CreativeAgent(mock_client)
        # Mock inputs
        brief = MagicMock(spec=StrategyBrief)
        brief.model_dump_json.return_value = "{}"
        personas = MagicMock(spec=AudienceSegments)
        personas.model_dump_json.return_value = "{}"

        result = agent.generate_variants(
            product="Product",
            brief=brief,
            personas=personas,
            platforms=["Meta"],
        )

        assert len(result.variants) == 2
        assert result.variants[0].concept_title == "V1"

    def test_scoring_agent_aggregation(self) -> None:
        """ScoringAgent should aggregate multiple evaluations into a CompositeReadinessReport."""
        from ads_ai.agents.scoring import ScoringAgent
        from ads_ai.agents.models import CompositeReadinessReport, AdScript

        mock_client = MagicMock(spec=genai.Client)
        mock_report = CompositeReadinessReport(
            variant_decisions=[], # Simplified for test
            composite_score=85.0,
            readiness_label="READY",
            critical_defects=[],
            optimization_priorities=[],
        )

        mock_response = MagicMock()
        mock_response.text = mock_report.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        agent = ScoringAgent(mock_client)
        result = agent.aggregate_and_score(
            variants=[AdScript(concept_title="V1", hook="H1", body_copy="B1", call_to_action="C1", visual_cues="V1")],
            brief=MagicMock(),
            evaluations=[{"clarity": MagicMock()}],
            intent_evaluation=MagicMock(),
        )

        assert result.composite_score == 85.0

    def test_iteration_controller_management(self) -> None:
        """IterationControllerAgent should produce an IterationControlReport."""
        from ads_ai.agents.iteration import IterationControllerAgent
        from ads_ai.agents.models import IterationControlReport

        mock_client = MagicMock(spec=genai.Client)
        mock_report = IterationControlReport(
            variant_plans=[],
            iteration_control_parameters={},
            progress_tracking_summary="Summary",
            convergence_status="Status",
            cross_variant_insights=[],
        )

        mock_response = MagicMock()
        mock_response.text = mock_report.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        agent = IterationControllerAgent(mock_client)
        result = agent.manage_iteration(
            variants=[],
            brief=MagicMock(),
            evaluations=[],
            readiness_report=MagicMock(),
        )

        assert result.progress_tracking_summary == "Summary"
