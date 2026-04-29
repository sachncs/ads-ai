"""Tests for the CLI entry point in main.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ads_ai.main import (
    EXIT_CODE_MISSING_ARGS,
    EXIT_CODE_MISSING_KEY,
    EXIT_CODE_PIPELINE_FAILURE,
    build_argument_parser,
    main,
    validate_environment,
)


class TestBuildArgumentParser:
    """Tests for CLI argument construction."""

    def test_parser_has_required_url_flag(self) -> None:
        """Should accept --url argument."""
        parser = build_argument_parser()
        args = parser.parse_args(["--url", "https://example.com", "--goal", "Sales"])
        assert args.url == "https://example.com"
        assert args.goal == "Sales"

    def test_parser_has_required_product_audience_flags(self) -> None:
        """Should accept --product and --audience arguments."""
        parser = build_argument_parser()
        args = parser.parse_args([
            "--product", "Chair", "--audience", "Workers", "--goal", "Sales"
        ])
        assert args.product == "Chair"
        assert args.audience == "Workers"

    def test_parser_default_platforms_empty(self) -> None:
        """Should default platforms to empty list."""
        parser = build_argument_parser()
        args = parser.parse_args(["--goal", "Sales"])
        assert args.platforms == []

    def test_parser_platforms_multiple(self) -> None:
        """Should accept multiple platforms."""
        parser = build_argument_parser()
        args = parser.parse_args([
            "--goal", "Sales", "--platforms", "TikTok", "Meta"
        ])
        assert args.platforms == ["TikTok", "Meta"]

    def test_parser_goal_required(self) -> None:
        """Should fail when --goal is missing."""
        parser = build_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestValidateEnvironment:
    """Tests for API key validation."""

    def test_missing_api_key_exits(self) -> None:
        """Should exit with EXIT_CODE_MISSING_KEY when API key is missing."""
        with patch("ads_ai.main.settings") as mock_settings:
            mock_settings.gemini_api_key = ""
            with pytest.raises(SystemExit) as exc_info:
                validate_environment()
            assert exc_info.value.code == EXIT_CODE_MISSING_KEY

    def test_api_key_present_passes(self) -> None:
        """Should not exit when API key is set."""
        with patch("ads_ai.main.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            # Should not raise
            validate_environment()


class TestMain:
    """Tests for the main entry point."""

    @patch("ads_ai.main.validate_environment")
    @patch("ads_ai.main.OrchestratorPipeline")
    @patch("ads_ai.main.genai.Client")
    @patch("ads_ai.main.settings")
    def test_url_mode_success(
        self,
        mock_settings: MagicMock,
        mock_client_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Should dispatch to run_from_url when --url is provided."""
        mock_settings.gemini_api_key = "test-key"
        mock_orchestrator = mock_pipeline_cls.return_value

        with patch("sys.argv", [
            "ads-ai", "--url", "https://example.com", "--goal", "Sales"
        ]):
            main()

        mock_orchestrator.run_from_url.assert_called_once()
        call_kwargs = mock_orchestrator.run_from_url.call_args.kwargs
        assert call_kwargs["url"] == "https://example.com"
        assert call_kwargs["goal"] == "Sales"

    @patch("ads_ai.main.validate_environment")
    @patch("ads_ai.main.OrchestratorPipeline")
    @patch("ads_ai.main.genai.Client")
    @patch("ads_ai.main.settings")
    def test_explicit_mode_success(
        self,
        mock_settings: MagicMock,
        mock_client_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Should dispatch to run when --product and --audience are provided."""
        mock_settings.gemini_api_key = "test-key"
        mock_orchestrator = mock_pipeline_cls.return_value

        with patch("sys.argv", [
            "ads-ai",
            "--product", "Chair",
            "--audience", "Workers",
            "--goal", "Sales",
        ]):
            main()

        mock_orchestrator.run.assert_called_once()
        call_kwargs = mock_orchestrator.run.call_args.kwargs
        assert call_kwargs["product"] == "Chair"
        assert call_kwargs["audience_desc"] == "Workers"
        assert call_kwargs["goal"] == "Sales"

    @patch("ads_ai.main.validate_environment")
    @patch("ads_ai.main.settings")
    def test_explicit_mode_missing_args(
        self,
        mock_settings: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Should exit with EXIT_CODE_MISSING_ARGS when required args missing."""
        mock_settings.gemini_api_key = "test-key"

        with patch("sys.argv", [
            "ads-ai", "--product", "Chair", "--goal", "Sales"
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == EXIT_CODE_MISSING_ARGS

    @patch("ads_ai.main.validate_environment")
    @patch("ads_ai.main.OrchestratorPipeline")
    @patch("ads_ai.main.genai.Client")
    @patch("ads_ai.main.settings")
    def test_keyboard_interrupt(
        self,
        mock_settings: MagicMock,
        mock_client_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Should exit with 130 on KeyboardInterrupt."""
        mock_settings.gemini_api_key = "test-key"
        mock_orchestrator = mock_pipeline_cls.return_value
        mock_orchestrator.run_from_url.side_effect = KeyboardInterrupt()

        with patch("sys.argv", [
            "ads-ai", "--url", "https://example.com", "--goal", "Sales"
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 130

    @patch("ads_ai.main.validate_environment")
    @patch("ads_ai.main.OrchestratorPipeline")
    @patch("ads_ai.main.genai.Client")
    @patch("ads_ai.main.settings")
    def test_unexpected_exception(
        self,
        mock_settings: MagicMock,
        mock_client_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Should exit with EXIT_CODE_PIPELINE_FAILURE on generic errors."""
        mock_settings.gemini_api_key = "test-key"
        mock_orchestrator = mock_pipeline_cls.return_value
        mock_orchestrator.run_from_url.side_effect = RuntimeError("boom")

        with patch("sys.argv", [
            "ads-ai", "--url", "https://example.com", "--goal", "Sales"
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == EXIT_CODE_PIPELINE_FAILURE
