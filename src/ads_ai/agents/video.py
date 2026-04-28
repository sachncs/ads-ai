"""Video Generation Agent module."""

from __future__ import annotations

import logging
import time
from typing import cast

from google import genai  # type: ignore[attr-defined]

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import (
    AdScript,
    AssetProductionVariant,
    VideoGenerationResult,
)
from ads_ai.config import settings

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_TIMEOUT_SECONDS = 300
"""Maximum seconds to wait for video generation to complete."""


class VideoGenerationError(Exception):
    """Raised when video generation fails unrecoverably."""


class VideoGenerationAgent(BaseAgent):
    """Generates cinematic video ads using Veo 3.1.

    This agent uses a highly descriptive prompt synthesis step before
    triggering the Veo generative video model to produce the final
    creative asset.
    """

    def __init__(self, client: genai.Client) -> None:
        """Initializes the VideoGenerationAgent with a GenAI client.

        Args:
            client: The Google GenAI client.

        Attributes:
            video_model: The Veo model identifier used for video generation
                (defaults to ``settings.default_video_model``).
        """
        super().__init__(client, model_name=settings.default_text_model)
        self.video_model = settings.default_video_model

    def synthesize_video_prompt(
        self,
        script: AdScript,
        production_plan: AssetProductionVariant,
    ) -> str:
        """Synthesizes a descriptive Veo prompt from an ad script.

        Args:
            script: The final approved ad script.
            production_plan: Technical production design for visual flow.

        Returns:
            A single, continuous text prompt string for the video model.

        Raises:
            VideoGenerationError: If prompt synthesis fails.
        """
        variant_name = getattr(script, "concept_title", "?")
        logger.info(
            "synthesize_video_prompt started concept=%s",
            variant_name,
        )
        call_start = time.perf_counter()
        try:
            video_prompt = f"""
        Role: Senior Video Director & AI Prompt Architect for Veo.
        Objective: Synthesize a high-fidelity, technically precise visual prompt for a
        text-to-video AI model (Veo) based on an advertising script and production design.

        INPUTS:
        - Ad Script Narrative: {script.model_dump_json()}
        - Technical Production Plan: {production_plan.model_dump_json()}

        EXECUTION STEPS:
        1. VISUAL DNA DEFINITION: Establish the cinematic style (e.g., High-contrast,
           Soft-lighting, Macro-photography, Handheld-kinetic).
        2. SUBJECT & ACTION MAPPING: Describe the primary subjects and their EXACT motions
           as specified in the script.
        3. CAMERA ARCHITECTURE: Define the camera's path (e.g., A slow 360-degree orbit around
           the product, followed by a close-up zoom on the logo).
        4. TEMPORAL CONSISTENCY: Ensure the scene transitions are described as a continuous flow.
           Avoid "And then..."—use "The camera glides into..."
        5. LIGHTING & ENVIRONMENT: Define the 3D space, atmospheric effects (e.g., dust motes,
           lens flares), and time of day.
        6. TECHNICAL QUALITY MARKERS: Append high-fidelity descriptors (e.g., 4k, photorealistic,
           seamless textures, high-quality rendering).

        CONSTRAINTS & RULES:
        - SINGLE PARAGRAPH: Output ONLY the final prompt as a single, continuous paragraph
          of descriptive text.
        - NO META-TALK: Do not include "Here is the prompt" or any preamble.
        - NARRATIVE GROUNDING: Every visual element must directly serve the "Brand Integration"
          or "CTA" defined in the script.
        - OUTPUT DISCIPLINE: Return ONLY the raw prompt text.
        """
            api_response = self.client.models.generate_content(
                model=self.model_name,
                contents=video_prompt,
            )
            elapsed = time.perf_counter() - call_start
            logger.info(
                "synthesize_video_prompt completed concept=%s elapsed=%.3fs",
                variant_name,
                elapsed,
            )
            return cast(str, api_response.text).strip()
        except Exception as e:
            elapsed = time.perf_counter() - call_start
            logger.exception(
                "synthesize_video_prompt failed concept=%s elapsed=%.3fs",
                variant_name,
                elapsed,
            )
            raise VideoGenerationError(
                f"Prompt synthesis failed for {variant_name}: {e}") from e

    def generate_video(
        self,
        script: AdScript,
        production_plan: AssetProductionVariant,
        output_path: str = "final_ad.mp4",
    ) -> VideoGenerationResult:
        """Generates the final video ad using Veo 3.1.

        Args:
            script: The final approved ad script.
            production_plan: Technical production design for visual flow.
            output_path: Target filesystem path for the .mp4 file.

        Returns:
            A ``VideoGenerationResult`` with the path to the saved video file.

        Raises:
            VideoGenerationError: If video generation fails unrecoverably.
        """
        variant_name = getattr(script, "concept_title", "?")
        logger.info(
            "generate_video started concept=%s output=%s",
            variant_name,
            output_path,
        )
        call_start = time.perf_counter()
        try:
            video_prompt = self.synthesize_video_prompt(script, production_plan)
            logger.info(
                "generate_video video_prompt_length=%d",
                len(video_prompt),
            )

            logger.info(
                "generate_video triggering model=%s",
                self.video_model,
            )
            video_operation = self.client.models.generate_videos(
                model=self.video_model,
                prompt=video_prompt,
            )

            # Poll the operation status until the video is ready.
            video_timeout_seconds = DEFAULT_VIDEO_TIMEOUT_SECONDS
            polling_start = time.perf_counter()
            while not video_operation.done:
                elapsed = time.perf_counter() - polling_start
                if elapsed >= video_timeout_seconds:
                    logger.error(
                        "Video generation timed out after %ds concept=%s",
                        video_timeout_seconds,
                        variant_name,
                    )
                    raise VideoGenerationError(
                        f"Video generation timed out after "
                        f"{video_timeout_seconds}s for {variant_name}")
                logger.info(
                    "Waiting for video generation to complete (polling every 10s)...",
                )
                time.sleep(10)
                video_operation = self.client.operations.get(video_operation)

            if not video_operation.response or not video_operation.response.generated_videos:
                logger.error(
                    "Video generation failed or returned no videos concept=%s",
                    variant_name,
                )
                raise VideoGenerationError(
                    f"No videos generated for {variant_name}")

            video_result = video_operation.response.generated_videos[0]

            logger.info(
                "generate_video downloading concept=%s output=%s",
                variant_name,
                output_path,
            )
            self.client.files.download(file=video_result.video)
            video_result.video.save(output_path)

            elapsed = time.perf_counter() - call_start
            logger.info(
                "generate_video completed concept=%s output=%s elapsed=%.3fs",
                variant_name,
                output_path,
                elapsed,
            )
            return VideoGenerationResult(video_file_path=output_path)

        except VideoGenerationError:
            elapsed = time.perf_counter() - call_start
            logger.exception(
                "generate_video failed concept=%s elapsed=%.3fs",
                variant_name,
                elapsed,
            )
            raise
        except Exception as e:
            elapsed = time.perf_counter() - call_start
            logger.exception(
                "generate_video unexpected error concept=%s elapsed=%.3fs error=%s",
                variant_name,
                elapsed,
                str(e)[:200],
            )
            raise VideoGenerationError(
                f"Unexpected error generating video for {variant_name}: {e}"
            ) from e
