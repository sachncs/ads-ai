"""ads.ai — AI-Driven Multi-Agent Advertising Pipeline."""

from __future__ import annotations

from ads_ai.pipeline import OrchestratorPipeline
from ads_ai.pipeline_stages import (
    PipelineStageRegistry,
    RetryPolicy,
    StageConfig,
    StageExecutionResult,
    StageStatus,
)

__version__ = "0.2.0"
__all__ = [
    "OrchestratorPipeline",
    "PipelineStageRegistry",
    "RetryPolicy",
    "StageConfig",
    "StageExecutionResult",
    "StageStatus",
]
