"""Unit tests for core pipeline components and configuration."""

from __future__ import annotations

from unittest.mock import MagicMock

import google.genai as genai
import pytest

from ads_ai.agents import ExtractedInputs, URLIntelligenceAgent
from ads_ai.agents.intelligence import URLValidationError
from ads_ai.config import get_settings
from ads_ai.pipeline import OrchestratorPipeline


class TestSettings:
    """Validates that application settings load correctly."""

    def test_default_model(self) -> None:
        """Default text model should be gemini-3.1-pro-preview."""
        result = get_settings()
        assert result.default_text_model == "gemini-3.1-pro-preview"

    def test_default_log_level(self) -> None:
        """Default log level should be INFO."""
        result = get_settings()
        assert result.log_level == "INFO"


class TestURLIntelligenceAgent:
    """Validates URL Intelligence Agent parsing with mocked LLM."""

    def test_parse_url_returns_extracted_inputs(self) -> None:
        """Agent should parse mock LLM response into ExtractedInputs."""
        mock_client = MagicMock(spec=genai.Client)

        valid_json = ExtractedInputs(
            product_description="Test Product",
            value_proposition="Test Prop",
            target_audience="Test Audience",
            inferred_segments=["A", "B"],
            brand_name="Test Brand",
            brand_tone="Test Tone",
            visual_identity="Test Identity",
            competitive_category="Test Category",
            differentiators="Test Diff",
            funnel_type="Test Funnel",
            cta_type="Buy Now",
            missing_information=[],
        ).model_dump_json()

        mock_response = MagicMock()
        mock_response.text = valid_json
        mock_client.models = MagicMock()
        mock_client.models.generate_content = MagicMock(
            return_value=mock_response)

        agent = URLIntelligenceAgent(mock_client)
        result = agent.parse_url("https://example.com")

        assert result.brand_name == "Test Brand"
        assert "A" in result.inferred_segments

    def test_validate_url_rejects_file_scheme(self) -> None:
        """Should raise URLValidationError for non-HTTP(S) schemes."""
        with pytest.raises(URLValidationError, match="scheme must be http or https"):
            URLIntelligenceAgent._validate_url("file:///etc/passwd")

    def test_validate_url_rejects_private_ip(self) -> None:
        """Should raise URLValidationError for private IP addresses."""
        with pytest.raises(URLValidationError, match="restricted IP"):
            URLIntelligenceAgent._validate_url("http://192.168.1.1")

    def test_validate_url_rejects_loopback(self) -> None:
        """Should raise URLValidationError for loopback addresses."""
        with pytest.raises(URLValidationError, match="restricted IP"):
            URLIntelligenceAgent._validate_url("http://127.0.0.1")

    def test_validate_url_rejects_too_long(self) -> None:
        """Should raise URLValidationError for excessively long URLs."""
        with pytest.raises(URLValidationError, match="exceeds maximum length"):
            URLIntelligenceAgent._validate_url("https://example.com/" + "x" * 3000)

    def test_validate_url_accepts_valid_https(self) -> None:
        """Should not raise for a well-formed HTTPS URL."""
        URLIntelligenceAgent._validate_url("https://example.com/product")


class TestOrchestratorInitialization:
    """Validates orchestrator agent wiring."""

    def test_agents_are_initialized(self) -> None:
        """All sub-agents should be instantiated during __init__."""
        mock_client = MagicMock(spec=genai.Client)
        orchestrator = OrchestratorPipeline(client=mock_client)

        assert orchestrator.strategy_agent is not None
        assert orchestrator.client is mock_client
