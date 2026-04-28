"""Quantified quality tests for each agent's output integrity."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from google import genai

from ads_ai.agents.creative import CreativeAgent
from ads_ai.agents.video import VideoGenerationAgent
from ads_ai.agents.scoring import ScoringAgent
from ads_ai.agents.models import (
    AdScript,
    AssetProductionVariant,
    CreativeVariants,
    StrategyBrief,
    AudienceSegments,
    CompositeReadinessReport,
    ShotSceneDesign,
)


@pytest.fixture
def mock_client() -> MagicMock:
    """Provides a mocked GenAI client."""
    client = MagicMock(spec=genai.Client)
    client.models = MagicMock()
    return client


class TestAgentQuality:
    """Tests evaluating agent outputs against quantified standards."""

    def test_creative_agent_variant_quality(self, mock_client: MagicMock) -> None:
        """CreativeAgent must produce variants with hooks and multiple scenes."""
        # Setup mock inputs
        brief = MagicMock(spec=StrategyBrief)
        brief.model_dump_json.return_value = '{"product_name": "Test"}'
        personas = MagicMock(spec=AudienceSegments)
        personas.model_dump_json.return_value = '{"personas": []}'

        # Mock variant logic
        agent = CreativeAgent(mock_client)
        mock_variant = AdScript(
            concept_title="Test title",
            core_idea="Test idea",
            hook="Short and punchy hook.",  # < 15 words
            script_scenes=[
                {"description": "S1", "visual_cues": "C1", "dialogue_vo": "D1"},
                {"description": "S2", "visual_cues": "C2", "dialogue_vo": "D2"},
                {"description": "S3", "visual_cues": "C3", "dialogue_vo": "D3"},
            ],
            brand_integration="Logo on screen.",
            cta="Buy now.",
            video_prompt="Cinematic 4k high-fidelity render...",
            variant_name="V1"
        )

        mock_response = MagicMock()
        mock_response.text = CreativeVariants(variants=[mock_variant]).model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        result = agent.generate_variants("Prod", brief, personas, ["Meta"])

        # QUANTIFIED ASSERTIONS
        assert len(result.variants[0].script_scenes) >= 3, "Variant must have >= 3 scenes."
        assert len(result.variants[0].hook.split()) < 15, "Hook must be concise (< 15 words)."
        assert "cinematic" in result.variants[0].video_prompt.lower(), "Video prompt must contain cinematic markers."

    def test_video_generation_prompt_synthesis_quality(self, mock_client: MagicMock) -> None:
        """VideoGenerationAgent must synthesize detailed prompts with cinematic markers."""
        agent = VideoGenerationAgent(mock_client)

        script = MagicMock(spec=AdScript)
        script.model_dump_json.return_value = '{"concept_title": "Test"}'
        plan = MagicMock(spec=AssetProductionVariant)
        plan.model_dump_json.return_value = '{"concept_title": "Test"}'

        # Mock detailed cinematic prompt
        cinematic_prompt = (
            "A high-fidelity 4k cinematic render of a product with stunning detail. "
            "Atmospheric lighting with soft lens flares macro-photography and vivid color grading. "
            "The camera glides smoothly in a slow 360-degree orbit around the object, "
            "capturing every photorealistic texture in seamless detail with dynamic depth of field."
        )
        mock_response = MagicMock()
        mock_response.text = cinematic_prompt
        mock_client.models.generate_content.return_value = mock_response

        result_prompt = agent.synthesize_video_prompt(script, plan)

        # QUANTIFIED ASSERTIONS
        assert len(result_prompt.split()) > 40, "Prompt should be descriptively long."
        markers = ["4k", "cinematic", "lighting", "shot", "camera"]
        found_markers = [m for m in markers if m in result_prompt.lower()]
        assert len(found_markers) >= 3, f"Prompt should contain cinematic markers. Found: {found_markers}"

    def test_scoring_agent_ready_criteria(self, mock_client: MagicMock) -> None:
        """ScoringAgent should only flag 'Ready' if no critical defects exist."""
        agent = ScoringAgent(mock_client)

        # Mock high-score report with NO critical defects
        mock_report = CompositeReadinessReport(
            target_kpis=["ROI"],
            variant_decisions=[
                {
                    "concept_title": "V1",
                    "final_readiness_score": 92.5,
                    "is_ready": True,
                    "status": "GO",
                    "confidence": 0.9,
                    "primary_strength": "Hook",
                    "primary_blocker": None,
                    "category_breakdown": []
                }
            ],
            best_overall_variant="V1",
            system_readiness_flag=True,
            strategic_trade_offs=[],
            critical_risks=[]
        )

        mock_response = MagicMock()
        mock_response.text = mock_report.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        result = agent.aggregate_and_score([], MagicMock(), [], MagicMock())

        # QUANTIFIED ASSERTIONS
        assert result.system_readiness_flag is True
        assert result.variant_decisions[0].primary_blocker or "" == ""
        assert result.variant_decisions[0].final_readiness_score > 80.0

    def test_brand_linkage_quality(self, mock_client: MagicMock) -> None:
        """BrandLinkageAgent must identify brand integration points."""
        from ads_ai.agents.brand import BrandLinkageAgent
        from ads_ai.agents.models import BrandLinkageEvaluation

        agent = BrandLinkageAgent(mock_client)
        mock_report = BrandLinkageEvaluation(
            brand_attribution_summary="Logo is prominent in scene 1 and 3.",
            brand_linkage_score=90,
            sub_scores={"attribution_clarity": 95, "timing": 85, "integration": 90, "memorability": 90},
            persona_recalls=[],
            confusion_risk="Low",
            confusion_explanation="Clear logo placement.",
            distinctiveness_score=85,
            alignment_score=90,
            issues_identified=[],
            recommended_fixes=[]
        )
        mock_response = MagicMock()
        mock_response.text = mock_report.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        result = agent.evaluate(MagicMock(), MagicMock(), "Logo", MagicMock())

        # QUANTIFIED ASSERTIONS
        assert result.brand_linkage_score >= 80, "High quality linkage required."
        assert result.confusion_risk == "Low"

    def test_clarity_agent_comprehension(self, mock_client: MagicMock) -> None:
        """MessageClarityAgent must provide specific recommended fixes for low clarity."""
        from ads_ai.agents.clarity import MessageClarityAgent
        from ads_ai.agents.models import ClarityEvaluation

        agent = MessageClarityAgent(mock_client)
        mock_report = ClarityEvaluation(
            core_message="Buy our product.",
            message_consistency="High",
            clarity_score=65, # Below target
            sub_scores={"comprehensibility": 70, "cognitive_load": 60, "explicitness": 65, "signal_vs_noise": 65},
            persona_comprehensions=[],
            retention_likelihood=60,
            retention_justification="Vague CTA.",
            time_to_clarity="5s",
            issues_identified=["Vague call to action."],
            recommended_fixes=["Change CTA to 'Shop Now' for clarity."]
        )
        mock_response = MagicMock()
        mock_response.text = mock_report.model_dump_json()
        mock_client.models.generate_content.return_value = mock_response

        result = agent.evaluate(MagicMock(), MagicMock(), MagicMock())

        # QUANTIFIED ASSERTIONS
        if result.clarity_score < 70:
            assert len(result.recommended_fixes) > 0, "Fixes must be provided for sub-par scores."
            assert "cta" in result.recommended_fixes[0].lower()
