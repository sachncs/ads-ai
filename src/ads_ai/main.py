"""CLI entry point for the ads.ai advertising pipeline.

Provides two execution modes:
1. **URL mode** — extracts product context from a URL automatically.
2. **Explicit mode** — accepts product and audience descriptions directly.

Example usage::

    ads-ai --url "https://example.com" --goal "Maximize Sales"
    ads-ai --product "Ergonomic Chair" --audience "Remote Workers" --goal "Sales"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from google import genai

from ads_ai.config import settings
from ads_ai.pipeline import OrchestratorPipeline

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

EXIT_CODE_MISSING_KEY = 1
"""Exit code when the Gemini API key is not configured."""

EXIT_CODE_MISSING_ARGS = 2
"""Exit code when required CLI arguments are absent."""

EXIT_CODE_PIPELINE_FAILURE = 3
"""Exit code when the pipeline raises an unrecoverable error."""


def build_argument_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser.

    Returns:
        A configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description="Run the ads.ai multi-agent advertising pipeline.",)

    parser.add_argument(
        "--url",
        type=str,
        help="Product URL for automatic context extraction.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="Primary business objective (e.g. 'Maximize Sales').",
    )
    parser.add_argument(
        "--product",
        type=str,
        help="Product description.  Required when --url is omitted.",
    )
    parser.add_argument(
        "--audience",
        type=str,
        help="Target audience description.  Required when --url is omitted.",
    )
    parser.add_argument(
        "--platforms",
        nargs="*",
        default=[],
        help="Target platforms (e.g. TikTok Meta YouTube).",
    )
    parser.add_argument(
        "--budget",
        type=str,
        default="TBD",
        help="Campaign budget or 'TBD' for auto-inference.",
    )
    parser.add_argument(
        "--brand-colors",
        type=str,
        help="Explicit brand color palette (e.g. 'Blue #0000FF, Gold #FFD700').",
    )
    parser.add_argument(
        "--brand-images",
        type=str,
        help="Descriptions of brand imagery or logo assets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom directory for run artifacts (default: outputs/).",
    )

    return parser


def validate_environment() -> None:
    """Exits if the Gemini API key is missing.

    Raises:
        SystemExit: With ``EXIT_CODE_MISSING_KEY`` if the API key is not set.
    """
    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY is not set in the environment.")
        sys.exit(EXIT_CODE_MISSING_KEY)


def main() -> None:
    """Parses arguments and dispatches the appropriate pipeline mode."""
    parser = build_argument_parser()
    args = parser.parse_args()

    validate_environment()

    client = genai.Client(api_key=settings.gemini_api_key)
    orchestrator = OrchestratorPipeline(client)

    run_start = time.perf_counter()
    try:
        if args.url:
            logger.info("Executing URL-driven pipeline mode.")
            orchestrator.run_from_url(
                url=args.url,
                goal=args.goal,
                platforms=args.platforms,
                budget=args.budget,
                output_dir=args.output_dir,
            )
        else:
            if not args.product or not args.audience:
                logger.error(
                    "--product and --audience are required when --url is omitted.",
                )
                sys.exit(EXIT_CODE_MISSING_ARGS)

            logger.info("Executing explicit-input pipeline mode.")

            brand_context = ""
            if args.brand_colors or args.brand_images:
                brand_context = (
                    f"Brand Colors: {args.brand_colors}. Brand Images: {args.brand_images}"
                )

            orchestrator.run(
                product=args.product,
                goal=args.goal,
                audience_desc=args.audience,
                platforms=args.platforms,
                budget=args.budget,
                brand_assets=brand_context,
                output_dir=args.output_dir,
            )
        elapsed = time.perf_counter() - run_start
        logger.info("Pipeline complete. total_elapsed=%.3fs", elapsed)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(128 + 2)  # 130 = 128 + SIGINT(2)

    except Exception:
        elapsed = time.perf_counter() - run_start
        logger.exception(
            "Pipeline failed with an unrecoverable error. total_elapsed=%.3fs",
            elapsed,
        )
        sys.exit(EXIT_CODE_PIPELINE_FAILURE)


if __name__ == "__main__":
    main()
