"""Video Generation Agent module."""

from __future__ import annotations

import logging
import time

from google import genai

from ads_ai.agents.base import BaseAgent
from ads_ai.agents.models import AdScript, AssetProductionVariant, VideoGenerationResult
from ads_ai.config import settings

logger = logging.getLogger(__name__)


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
        """
        super().__init__(client, model_name=settings.default_text_model)
        self.video_model = settings.default_video_model

    def _synthesize_video_prompt(
        self, script: AdScript, production_plan: AssetProductionVariant
    ) -> str:
        """Uses the LLM to synthesize a single highly descriptive Veo prompt.

        Args:
            script: The final approved ad script.
            production_plan: technical production design for visual flow.

        Returns:
            A single, continuous text prompt string for the video model.
        """
        prompt = f"""
        Role: Senior Video Director & AI Prompt Architect for Veo.
        Objective: Synthezise a high-fidelity, technically precise visual prompt for a
        text-to-video AI model (Veo) based on an advertising script and production design.

        INPUTS:
        - Ad Script Narrative: {script.model_dump_json()}
        - Technical Production Plan: {production_plan.model_dump_json()}

        EXECUTION STEPS:
        1. VISUAL DNA DEFINTION: Establish the cinematic style (e.g., High-contrast,
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
        # Generate the prompt string without using a JSON schema
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()

    def generate_video(
        self,
        script: AdScript,
        production_plan: AssetProductionVariant,
        output_path: str = "final_ad.mp4",
    ) -> VideoGenerationResult | None:
        """Generates the final video ad using Veo 3.1.

        Args:
            script: The final approved ad script.
            production_plan: Technical production design for visual flow.
            output_path: Target filesystem path for the .mp4 file.

        Returns:
            A ``VideoGenerationResult`` if successful, otherwise ``None``.

        Raises:
            Exception: If prompt synthesis or video generation fails.
        """
        logger.info(
            "Synthesizing Veo prompt for variant: %s...", script.variant_name)
        try:
            veo_prompt = self._synthesize_video_prompt(script, production_plan)
            logger.info("Veo Prompt: %s", veo_prompt)

            logger.info(
                "Triggering video generation (veo-3.1-generate-preview)...")
            operation = self.client.models.generate_videos(
                model=self.video_model,
                prompt=veo_prompt,
            )

            # Poll the operation status until the video is ready.
            while not operation.done:
                logger.info(
                    "Waiting for video generation to complete (polling every 10s)..."
                )
                time.sleep(10)
                operation = self.client.operations.get(operation)

            # Download the generated video.
            if not operation.response or not operation.response.generated_videos:
                logger.error("Video generation failed or returned no videos.")
                return None

            generated_video = operation.response.generated_videos[0]

            logger.info(
                "Video generated successfully. Downloading to %s...", output_path
            )
            self.client.files.download(file=generated_video.video)
            generated_video.video.save(output_path)
            logger.info("Video saved.")

            return VideoGenerationResult(video_file_path=output_path)

        except Exception as e:
            logger.error("Error during video generation: %s", e)
            return None
