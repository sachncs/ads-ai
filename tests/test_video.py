"""Tests for VideoGenerationAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ads_ai.agents.models import (
    AdScript,
    AssetProductionVariant,
    Scene,
    VideoGenerationResult,
)
from ads_ai.agents.video import VideoGenerationAgent, VideoGenerationError


class TestVideoGenerationAgent:
    """Tests for video generation workflow."""

    def test_synthesize_video_prompt_success(self) -> None:
        """Should return synthesized prompt text."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Cinematic 4k shot of product"
        mock_client.models.generate_content.return_value = mock_response

        agent = VideoGenerationAgent(mock_client)
        script = AdScript(
            concept_title="V1",
            core_idea="Idea",
            hook="Hook",
            script_scenes=[Scene(description="S", visual_cues="V", dialogue_vo="D")],
            brand_integration="Logo",
            cta="Buy",
            video_prompt="4k",
            variant_name="V1",
        )
        plan = AssetProductionVariant(
            concept_title="V1",
            required_assets=[],
            production_scenes=[],
            estimated_production_complexity="Low",
        )

        result = agent.synthesize_video_prompt(script, plan)
        assert result == "Cinematic 4k shot of product"

    def test_synthesize_video_prompt_failure(self) -> None:
        """Should raise VideoGenerationError on API failure."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API down")

        agent = VideoGenerationAgent(mock_client)
        script = AdScript(
            concept_title="V1",
            core_idea="Idea",
            hook="Hook",
            script_scenes=[Scene(description="S", visual_cues="V", dialogue_vo="D")],
            brand_integration="Logo",
            cta="Buy",
            video_prompt="4k",
            variant_name="V1",
        )
        plan = AssetProductionVariant(
            concept_title="V1",
            required_assets=[],
            production_scenes=[],
            estimated_production_complexity="Low",
        )

        with pytest.raises(VideoGenerationError, match="Prompt synthesis failed"):
            agent.synthesize_video_prompt(script, plan)

    def test_generate_video_success(self) -> None:
        """Should return VideoGenerationResult with output path."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "video prompt text"
        mock_client.models.generate_content.return_value = mock_response

        mock_video_op = MagicMock()
        mock_video_op.done = True
        mock_video_op.response.generated_videos = [MagicMock()]
        mock_client.models.generate_videos.return_value = mock_video_op

        agent = VideoGenerationAgent(mock_client)
        script = AdScript(
            concept_title="V1",
            core_idea="Idea",
            hook="Hook",
            script_scenes=[Scene(description="S", visual_cues="V", dialogue_vo="D")],
            brand_integration="Logo",
            cta="Buy",
            video_prompt="4k",
            variant_name="V1",
        )
        plan = AssetProductionVariant(
            concept_title="V1",
            required_assets=[],
            production_scenes=[],
            estimated_production_complexity="Low",
        )

        result = agent.generate_video(script, plan, output_path="out.mp4")
        assert isinstance(result, VideoGenerationResult)
        assert result.video_file_path == "out.mp4"

    def test_generate_video_timeout(self) -> None:
        """Should raise VideoGenerationError when polling times out."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "video prompt text"
        mock_client.models.generate_content.return_value = mock_response

        mock_video_op = MagicMock()
        mock_video_op.done = False
        mock_client.models.generate_videos.return_value = mock_video_op

        agent = VideoGenerationAgent(mock_client)
        script = AdScript(
            concept_title="V1",
            core_idea="Idea",
            hook="Hook",
            script_scenes=[Scene(description="S", visual_cues="V", dialogue_vo="D")],
            brand_integration="Logo",
            cta="Buy",
            video_prompt="4k",
            variant_name="V1",
        )
        plan = AssetProductionVariant(
            concept_title="V1",
            required_assets=[],
            production_scenes=[],
            estimated_production_complexity="Low",
        )

        with patch("time.sleep", return_value=None):
            with patch("time.perf_counter") as mock_perf:
                mock_perf.side_effect = [
                    0, 0, 0, 0, 301, 301, 301, 301, 301, 301,
                ]
                with pytest.raises(VideoGenerationError, match="timed out"):
                    agent.generate_video(script, plan)

    def test_generate_video_no_videos(self) -> None:
        """Should raise VideoGenerationError when no videos are generated."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "video prompt text"
        mock_client.models.generate_content.return_value = mock_response

        mock_video_op = MagicMock()
        mock_video_op.done = True
        mock_video_op.response.generated_videos = []
        mock_client.models.generate_videos.return_value = mock_video_op

        agent = VideoGenerationAgent(mock_client)
        script = AdScript(
            concept_title="V1",
            core_idea="Idea",
            hook="Hook",
            script_scenes=[Scene(description="S", visual_cues="V", dialogue_vo="D")],
            brand_integration="Logo",
            cta="Buy",
            video_prompt="4k",
            variant_name="V1",
        )
        plan = AssetProductionVariant(
            concept_title="V1",
            required_assets=[],
            production_scenes=[],
            estimated_production_complexity="Low",
        )

        with pytest.raises(VideoGenerationError, match="No videos generated"):
            agent.generate_video(script, plan)

    def test_generate_video_unexpected_error(self) -> None:
        """Should wrap unexpected errors in VideoGenerationError."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("boom")

        agent = VideoGenerationAgent(mock_client)
        script = AdScript(
            concept_title="V1",
            core_idea="Idea",
            hook="Hook",
            script_scenes=[Scene(description="S", visual_cues="V", dialogue_vo="D")],
            brand_integration="Logo",
            cta="Buy",
            video_prompt="4k",
            variant_name="V1",
        )
        plan = AssetProductionVariant(
            concept_title="V1",
            required_assets=[],
            production_scenes=[],
            estimated_production_complexity="Low",
        )

        with pytest.raises(VideoGenerationError, match="Prompt synthesis failed"):
            agent.generate_video(script, plan)
