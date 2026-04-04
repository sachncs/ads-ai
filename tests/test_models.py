"""Exhaustive validation tests for all agent Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ads_ai.agents.models import (
    AdScript,
    AssetProductionReport,
    AssetProductionVariant,
    AttentionEvaluation,
    AudienceSegments,
    BrandLinkageEvaluation,
    BudgetInferenceReport,
    ClarityEvaluation,
    ComplianceRiskReport,
    CompositeReadinessReport,
    CreativeVariants,
    DeploymentExperimentationReport,
    DiagnosticsEvaluation,
    ExternalValidationPlan,
    ExtractedInputs,
    IntentEvaluation,
    IterationControlReport,
    KnowledgeLearningReport,
    PlatformAdaptationReport,
    Scene,
    ShotSceneDesign,
    StrategyBrief,
    VideoGenerationResult,
)


class TestStep0Models:
    """Tests for Step 0 (Intelligence & Budget) models."""

    def test_extracted_inputs_valid(self) -> None:
        """Should validate a complete ExtractedInputs model."""
        data = {
            "product_description": "A high-end ergonomic chair.",
            "value_proposition": "Ultimate comfort for remote workers.",
            "target_audience": "Remote professionals aged 25-45.",
            "inferred_segments": ["Tech", "Design", "Management"],
            "brand_name": "ErgoRest",
            "brand_tone": "Professional yet approachable.",
            "visual_identity": "Minimalist, Earth tones.",
            "competitive_category": "Office Furniture",
            "differentiators": "Patented lumbar support.",
            "funnel_type": "Direct Response",
            "cta_type": "Shop Now",
            "missing_information": [],
        }
        model = ExtractedInputs(**data)
        assert model.brand_name == "ErgoRest"

    def test_extracted_inputs_invalid(self) -> None:
        """Should fail if required fields are missing."""
        with pytest.raises(ValidationError):
            ExtractedInputs(brand_name="Incomplete")


class TestStep1Models:
    """Tests for Step 1 (Strategy) models."""

    def test_strategy_brief_valid(self) -> None:
        """Should validate a complete StrategyBrief model."""
        data = {
            "product_name": "ErgoRest Chair",
            "ad_intent": "Drive Sales",
            "intent_explanation": "Direct conversion goal.",
            "kpis": [
                {"name": "CPA", "target_value": "<$50", "measurement_method": "FB Pixel"}
            ],
            "pre_release_targets": {
                "clarity_score_target": 80,
                "brand_linkage_score_target": 75,
                "hook_strength_threshold": "High",
                "message_retention_likelihood": "Medium",
                "simulated_intent_score": "7/10",
            },
            "audience_personas": [
                {
                    "demographics": "Remote workers",
                    "motivation": "Back pain relief",
                    "pain_point": "Uncomfortable chairs",
                    "buying_trigger": "New home office setup",
                    "likely_objection": "Price",
                }
            ],
            "message_strategy": {
                "value_proposition": "The last chair you'll ever need.",
                "message_pillars": ["Comfort", "Durability", "Style"],
                "tone": "Authoritative",
                "emotional_trigger": "Relief",
            },
            "creative_constraints": {"max_duration": 30},
            "decision_thresholds": {"min_score": 0.7},
        }
        model = StrategyBrief(**data)
        assert model.product_name == "ErgoRest Chair"
        assert len(model.kpis) == 1


class TestStep3Models:
    """Tests for Step 3 (Creative) models."""

    def test_ad_script_valid(self) -> None:
        """Should validate a complete AdScript model."""
        data = {
            "concept_title": "The Infinite Hook",
            "core_idea": "A chair that disappears as you sit.",
            "hook": "Stop settling for back pain.",
            "script_scenes": [
                {
                    "description": "Person sitting down.",
                    "visual_cues": "Close up on spine.",
                    "dialogue_vo": "Your back deserves better.",
                }
            ],
            "brand_integration": "Logo appears on the headrest.",
            "cta": "Shop ErgoRest.",
            "video_prompt": "Cinematic 4k shot of an office.",
            "variant_name": "Variant A",
        }
        # Fixed the typo in the dict key "brand_integration:"
        data["brand_integration"] = data.pop("brand_integration: ")
        model = AdScript(**data)
        assert model.concept_title == "The Infinite Hook"
        assert len(model.script_scenes) == 1


class TestStep4Models:
    """Tests for Step 4 (Evaluations) models."""

    @pytest.mark.parametrize("score", [-1, 101])
    def test_clarity_score_bounds(self, score: int) -> None:
        """Score fields should enforce 0-100 range."""
        from ads_ai.agents.models import ClaritySubScores
        with pytest.raises(ValidationError):
            ClaritySubScores(
                comprehensibility=score,
                cognitive_load=50,
                explicitness=50,
                signal_vs_noise=50,
            )

    def test_diagnostics_evaluation_valid(self) -> None:
        """Should validate a complete DiagnosticsEvaluation model."""
        data = {
            "structure_type": "Linear",
            "coherence_score": 85,
            "hook_analysis": {
                "hook_strength_score": 90,
                "hook_type_classification": "Visual",
                "weaknesses": [],
            },
            "pacing_score": 80,
            "sections_drag_rush": [],
            "drop_off_risk_timeline": [
                {"segment": "0:05", "risk_severity": "Low", "cause": "Transition"}
            ],
            "redundancy_score": 10,
            "elements_to_remove": [],
            "transition_quality_score": 90,
            "platform_fit_score": 95,
            "mismatches_identified": [],
            "overall_diagnostic_score": 88,
            "issues_identified": [],
            "recommended_fixes": [],
        }
        model = DiagnosticsEvaluation(**data)
        assert model.coherence_score == 85
        assert model.hook_analysis.hook_strength_score == 90
