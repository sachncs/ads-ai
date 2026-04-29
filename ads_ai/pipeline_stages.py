"""Dynamic pipeline stage registry and execution engine.

This module provides a declarative, configurable stage system that replaces
the hardcoded 13-step pipeline. Stages are defined as first-class objects
with dependencies, conditions, parallel hints, and per-stage retry policies.

Example:
    registry = PipelineStageRegistry()
    registry.register(StageConfig(
        id="strategy",
        agent_factory=lambda c: StrategyAgent(c),
        input_schema=StrategyBrief,
        output_schema=StrategyBrief,
        depends_on=["url_intelligence"],
        condition=lambda ctx: ctx.get("url") is not None,
    ))
    result = registry.execute(context, output_dir=Path("outputs"))
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ads_ai.utils.file_ops import ensure_output_dir, save_json

logger = logging.getLogger(__name__)

TInput = TypeVar("TInput", bound=BaseModel | dict[str, Any])
TOutput = TypeVar("TOutput", bound=BaseModel)


class StageStatus(str, Enum):
    """Possible execution states for a stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CONDITIONAL_SKIP = "conditional_skip"


class RetryPolicy(BaseModel):
    """Retry configuration for a single stage."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    retryable_exceptions: list[str] = field(
        default_factory=lambda: ["InternalServerError", "ResourceExhausted", "503"]
    )


@dataclass
class StageConfig(Generic[TInput, TOutput]):
    """Declarative configuration for a single pipeline stage.

    Attributes:
        id: Unique identifier for this stage (e.g., "strategy", "creative").
        description: Human-readable description of what this stage does.
        agent_factory: Callable that produces an agent instance given a GenAI client.
        input_schema: Type used to validate/serialize inputs to this stage.
        output_schema: Pydantic model class for this stage's output.
        depends_on: List of stage IDs that must complete before this stage runs.
        required_context_keys: Keys that must be present in the execution context.
        condition: Optional callable; if it returns False, stage is skipped.
        retry_policy: Retry configuration; uses sensible defaults if not specified.
        parallel_hints: Hints for parallel execution (max_workers, etc.).
        output_filename: Filename pattern for saving stage output.
        timeout_seconds: Maximum time for this stage before timing out.
        critical: If True, failure marks entire pipeline as failed.
    """

    id: str
    description: str
    agent_factory: Callable[[Any], Any]
    input_schema: type[TInput] | None = None
    output_schema: type[TOutput] | None = None
    depends_on: list[str] = field(default_factory=list)
    required_context_keys: list[str] = field(default_factory=list)
    condition: Callable[[dict[str, Any]], bool] | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    parallel_hints: dict[str, Any] = field(default_factory=dict)
    output_filename: str | None = None
    timeout_seconds: float | None = None
    critical: bool = True

    def validate_context(self, context: dict[str, Any]) -> bool:
        """Checks if required context keys are present.

        Args:
            context: The current pipeline execution context.

        Returns:
            True if all required keys are present, False otherwise.
        """
        for key in self.required_context_keys:
            if key not in context or context[key] is None:
                logger.debug(
                    "Stage %s skipped: required key '%s' not in context",
                    self.id,
                    key,
                )
                return False
        return True

    def should_run(self, context: dict[str, Any]) -> bool:
        """Determines whether this stage should execute.

        Args:
            context: The current pipeline execution context.

        Returns:
            True if the stage should run, False to skip.
        """
        if not self.validate_context(context):
            return False
        if self.condition is not None:
            try:
                if not self.condition(context):
                    logger.info("Stage %s skipped: condition returned False", self.id)
                    return False
            except Exception as e:
                logger.warning(
                    "Stage %s condition evaluation failed: %s. Running stage.",
                    self.id,
                    e,
                )
        return True


@dataclass
class StageExecutionResult:
    """Result of a single stage execution.

    Attributes:
        stage_id: The ID of the executed stage.
        status: Execution status (completed, failed, skipped, etc.).
        output: The stage's output if successful.
        error: Error message if failed.
        elapsed_seconds: Wall-clock time for execution.
        attempt_count: Number of attempts made (including retries).
        metadata: Additional execution metadata.
    """

    stage_id: str
    status: StageStatus
    output: Any = None
    error: str | None = None
    elapsed_seconds: float = 0.0
    attempt_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStageRegistry:
    """Dynamic registry and executor for pipeline stages.

    Manages stage registration, dependency resolution, parallel execution
    hints, and execution with retry policies.

    Example:
        registry = PipelineStageRegistry()

        # Register stages with declarative config
        registry.register(StageConfig(
            id="strategy",
            agent_factory=lambda c: StrategyAgent(c),
            depends_on=["url_intelligence"],
        ))

        # Execute with context and output directory
        results = registry.execute(
            context={"url": "https://example.com"},
            output_dir=Path("outputs"),
        )

    Alternative: Use ``create_legacy`` to get a pre-configured registry
    matching the existing 13-step pipeline.
    """

    def __init__(self) -> None:
        """Initializes an empty stage registry."""
        self._stages: dict[str, StageConfig] = {}
        self._execution_order: list[str] = []
        self._parallel_groups: list[list[str]] = []

    def register(self, config: StageConfig) -> PipelineStageRegistry:
        """Registers a stage configuration.

        Args:
            config: The stage configuration to register.

        Returns:
            Self for chaining.
        """
        self._stages[config.id] = config
        logger.debug(
            "Registered stage id=%s depends_on=%s",
            config.id,
            config.depends_on,
        )
        return self

    def register_many(self, configs: list[StageConfig]) -> PipelineStageRegistry:
        """Registers multiple stage configurations.

        Args:
            configs: List of stage configurations to register.

        Returns:
            Self for chaining.
        """
        for cfg in configs:
            self.register(cfg)
        return self

    def resolve_execution_order(self) -> list[list[str]]:
        """Resolves stage dependencies into an execution plan.

        Performs a topological sort with parallelization hints to produce
        an execution plan as lists of stages that can run concurrently.

        Returns:
            A list of stage ID lists. Each inner list contains stages that
            can be executed in parallel. Outer list is sequential.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        in_degree: dict[str, int] = {sid: 0 for sid in self._stages}
        adjacency: dict[str, list[str]] = {sid: [] for sid in self._stages}

        for stage_id, config in self._stages.items():
            for dep in config.depends_on:
                if dep not in self._stages:
                    raise ValueError(
                        f"Stage '{stage_id}' depends on unknown stage '{dep}'. "
                        f"Available stages: {list(self._stages.keys())}"
                    )
                adjacency[dep].append(stage_id)
                in_degree[stage_id] += 1

        levels: dict[str, int] = {}
        current_level: list[str] = [
            sid for sid, deg in in_degree.items() if deg == 0
        ]
        level_idx = 0

        while current_level:
            levels.update({sid: level_idx for sid in current_level})
            next_level: list[str] = []
            for sid in current_level:
                for neighbor in adjacency[sid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)
            current_level = next_level
            level_idx += 1

        if sum(in_degree.values()) > 0:
            cycle_nodes = [sid for sid, deg in in_degree.items() if deg > 0]
            raise ValueError(
                f"Circular dependency detected involving stages: {cycle_nodes}"
            )

        max_level = max(levels.values()) if levels else 0
        parallel_groups: list[list[str]] = [[] for _ in range(max_level + 1)]
        for sid, level in levels.items():
            parallel_groups[level].append(sid)

        self._execution_order = [sid for group in parallel_groups for sid in group]
        self._parallel_groups = parallel_groups
        return parallel_groups

    def get_stage(self, stage_id: str) -> StageConfig | None:
        """Retrieves a stage configuration by ID.

        Args:
            stage_id: The stage identifier.

        Returns:
            The stage configuration or None if not found.
        """
        return self._stages.get(stage_id)

    def list_stages(self) -> list[str]:
        """Returns all registered stage IDs in execution order.

        Returns:
            List of stage IDs.
        """
        return list(self._execution_order)

    def execute(
        self,
        context: dict[str, Any],
        output_dir: str | Path | None = None,
        client: Any = None,
    ) -> dict[str, StageExecutionResult]:
        """Executes all registered stages according to their dependencies.

        Args:
            context: Shared execution context passed to each stage.
            output_dir: Directory for saving stage outputs.
            client: The GenAI client instance.

        Returns:
            A dictionary mapping stage IDs to their execution results.
        """
        output_path = ensure_output_dir(output_dir or "outputs")
        results: dict[str, StageExecutionResult] = {}
        resolved_context = dict(context)

        parallel_groups = self.resolve_execution_order()
        logger.info(
            "Pipeline execution starting: %d stages in %d parallel groups",
            len(self._stages),
            len(parallel_groups),
        )

        for group_idx, group in enumerate(parallel_groups):
            group_start = time.perf_counter()
            logger.info(
                "Executing parallel group %d/%d: stages=%s",
                group_idx + 1,
                len(parallel_groups),
                group,
            )

            runnable = [
                sid for sid in group
                if self._stages[sid].should_run(resolved_context)
            ]
            skipped = [sid for sid in group if sid not in runnable]

            for sid in skipped:
                results[sid] = StageExecutionResult(
                    stage_id=sid,
                    status=StageStatus.CONDITIONAL_SKIP,
                    metadata={"reason": "condition_not_met"},
                )
                logger.info("Stage %s skipped (condition)", sid)

            if not runnable:
                continue

            max_workers = min(
                self._stages[sid].parallel_hints.get("max_workers", 4)
                for sid in runnable
            )
            max_workers = max(max_workers, 1)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._execute_single_stage,
                        sid,
                        resolved_context,
                        output_path,
                        client,
                    ): sid
                    for sid in runnable
                }

                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        result = future.result()
                        results[sid] = result
                        if result.output is not None:
                            resolved_context[sid] = result.output
                    except Exception as e:
                        logger.exception("Stage %s raised exception", sid)
                        results[sid] = StageExecutionResult(
                            stage_id=sid,
                            status=StageStatus.FAILED,
                            error=str(e),
                        )

            critical_failures = [
                sid for sid in runnable
                if results[sid].status == StageStatus.FAILED
                and self._stages[sid].critical
            ]
            if critical_failures:
                logger.error(
                    "Critical stage failures detected: %s. Halting pipeline.",
                    critical_failures,
                )
                for sid in self._stages:
                    if sid not in results:
                        results[sid] = StageExecutionResult(
                            stage_id=sid,
                            status=StageStatus.SKIPPED,
                            metadata={"reason": "upstream_critical_failure"},
                        )
                break

            group_elapsed = time.perf_counter() - group_start
            logger.info(
                "Parallel group %d completed in %.3fs: %d/%d succeeded",
                group_idx + 1,
                group_elapsed,
                sum(
                    1 for sid in runnable
                    if results[sid].status == StageStatus.COMPLETED
                ),
                len(runnable),
            )

        logger.info("Pipeline execution completed: %d stages processed", len(results))
        return results

    def _execute_single_stage(
        self,
        stage_id: str,
        context: dict[str, Any],
        output_path: Path,
        client: Any,
    ) -> StageExecutionResult:
        """Executes a single stage with retry logic.

        The stage config's agent_factory should produce an agent that has
        either an 'execute(context)' method or specific methods matching
        the stage's operation (e.g., 'create_brief(...)', 'generate_variants(...)').

        For backward compatibility, if the agent has neither, this method
        falls back to calling 'run(context)'.

        Args:
            stage_id: The ID of the stage to execute.
            context: The current execution context.
            output_path: Directory for saving outputs.
            client: The GenAI client.

        Returns:
            The stage execution result.
        """
        config = self._stages[stage_id]
        policy = config.retry_policy
        last_error: str | None = None
        start_time = time.perf_counter()

        for attempt_num in range(1, policy.max_attempts + 1):
            try:
                logger.info(
                    "Executing stage %s attempt %d/%d",
                    stage_id,
                    attempt_num,
                    policy.max_attempts,
                )
                stage_start = time.perf_counter()

                agent = config.agent_factory(client)

                if callable(getattr(agent, "execute", None)):
                    output = agent.execute(context)
                elif callable(getattr(agent, "run", None)):
                    output = agent.run(context)
                elif hasattr(agent, stage_id.replace("eval_", "")):
                    stage_method = getattr(agent, stage_id.replace("eval_", ""))
                    output = stage_method(context)
                else:
                    raise AttributeError(
                        f"Agent {agent.__class__.__name__} has no 'execute', 'run', "
                        f"or stage-specific method for stage '{stage_id}'"
                    )

                elapsed = time.perf_counter() - stage_start

                if config.output_filename and output is not None:
                    save_json(output, output_path / config.output_filename)

                return StageExecutionResult(
                    stage_id=stage_id,
                    status=StageStatus.COMPLETED,
                    output=output,
                    elapsed_seconds=time.perf_counter() - start_time,
                    attempt_count=attempt_num,
                    metadata={
                        "stage_elapsed": elapsed,
                        "retry_attempts": attempt_num - 1,
                    },
                )

            except Exception as e:
                last_error = str(e)
                is_retryable = any(
                    retry_exc in last_error for retry_exc in policy.retryable_exceptions
                )

                if attempt_num < policy.max_attempts and is_retryable:
                    delay = min(
                        policy.initial_delay
                        * (policy.backoff_multiplier ** (attempt_num - 1)),
                        policy.max_delay,
                    )
                    logger.warning(
                        "Stage %s attempt %d failed (retryable): %s. Retrying in %.1fs.",
                        stage_id,
                        attempt_num,
                        last_error[:100],
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Stage %s failed permanently after %d attempts: %s",
                        stage_id,
                        attempt_num,
                        last_error,
                    )
                    return StageExecutionResult(
                        stage_id=stage_id,
                        status=StageStatus.FAILED,
                        error=last_error,
                        elapsed_seconds=time.perf_counter() - start_time,
                        attempt_count=attempt_num,
                    )

        return StageExecutionResult(
            stage_id=stage_id,
            status=StageStatus.FAILED,
            error=last_error or "Unknown error after max retries",
            elapsed_seconds=time.perf_counter() - start_time,
            attempt_count=policy.max_attempts,
        )

    @classmethod
    def create_legacy(cls, client: Any) -> PipelineStageRegistry:
        """Creates a stage registry configured with the legacy 13-step pipeline.

        This provides backward compatibility while enabling new features
        like per-stage retry policies and conditional execution.

        Args:
            client: The GenAI client.

        Returns:
            A configured PipelineStageRegistry.
        """
        from ads_ai.agents import (
            AssetProductionAgent,
            AssetProductionReport,
            AttentionEvaluation,
            AttentionHeuristicAgent,
            AudienceAgent,
            AudienceSegments,
            BrandLinkageAgent,
            BrandLinkageEvaluation,
            BudgetInferenceAgent,
            BudgetInferenceReport,
            ClarityEvaluation,
            ComplianceRiskAgent,
            ComplianceRiskReport,
            CompositeReadinessReport,
            CreativeAgent,
            CreativeVariants,
            DeploymentExperimentationAgent,
            DeploymentExperimentationReport,
            DiagnosticsAgent,
            DiagnosticsEvaluation,
            ExternalValidationAgent,
            ExternalValidationPlan,
            ExtractedInputs,
            IntentEvaluation,
            IntentSimulationAgent,
            IterationControllerAgent,
            IterationControlReport,
            KnowledgeLearningAgent,
            KnowledgeLearningReport,
            MessageClarityAgent,
            PlatformAdaptationAgent,
            PlatformVariant,
            ScoringAgent,
            StrategyAgent,
            StrategyBrief,
            URLIntelligenceAgent,
            VideoGenerationAgent,
            VideoGenerationResult,
        )

        registry = cls()

        # Stage 0: URL Intelligence
        registry.register(StageConfig(
            id="url_intelligence",
            description="Extract product context from landing page URL",
            agent_factory=lambda c: URLIntelligenceAgent(c),
            output_schema=ExtractedInputs,
            output_filename="step_0_url_intelligence.json",
            critical=True,
            retry_policy=RetryPolicy(max_attempts=3),
        ))

        # Stage 0.5: Budget Inference (conditional on budget="TBD")
        registry.register(StageConfig(
            id="budget_inference",
            description="Auto-infer budget when TBD is provided",
            agent_factory=lambda c: BudgetInferenceAgent(c),
            output_schema=BudgetInferenceReport,
            output_filename="step_0_5_budget_inference.json",
            depends_on=["url_intelligence"],
            condition=lambda ctx: ctx.get("budget") == "TBD",
            critical=False,
            retry_policy=RetryPolicy(max_attempts=2),
        ))

        # Stage 1: Strategy
        registry.register(StageConfig(
            id="strategy",
            description="Generate strategy brief with KPIs and pillars",
            agent_factory=lambda c: StrategyAgent(c),
            output_schema=StrategyBrief,
            output_filename="step_1_strategy.json",
            depends_on=["url_intelligence"],
            critical=True,
            retry_policy=RetryPolicy(max_attempts=3),
        ))

        # Stage 2: Audience
        registry.register(StageConfig(
            id="audience",
            description="Model behavioral personas and segments",
            agent_factory=lambda c: AudienceAgent(c),
            output_schema=AudienceSegments,
            output_filename="step_2_audience.json",
            depends_on=["strategy"],
            critical=True,
            retry_policy=RetryPolicy(max_attempts=3),
        ))

        # Stage 3: Creative Generation
        registry.register(StageConfig(
            id="creative",
            description="Generate 3-5 ad script variants",
            agent_factory=lambda c: CreativeAgent(c),
            output_schema=CreativeVariants,
            output_filename="step_3_creative_base.json",
            depends_on=["strategy", "audience"],
            critical=True,
            retry_policy=RetryPolicy(max_attempts=3),
        ))

        # Evaluation stages (4a-4e) - run in parallel
        eval_parallel_hints = {"max_workers": 4}

        registry.register(StageConfig(
            id="eval_clarity",
            description="Evaluate message clarity",
            agent_factory=lambda c: MessageClarityAgent(c),
            output_schema=ClarityEvaluation,
            output_filename="step_4a_clarity.json",
            depends_on=["creative"],
            parallel_hints=eval_parallel_hints,
            critical=True,
        ))

        registry.register(StageConfig(
            id="eval_brand",
            description="Evaluate brand linkage",
            agent_factory=lambda c: BrandLinkageAgent(c),
            output_schema=BrandLinkageEvaluation,
            output_filename="step_4b_brand.json",
            depends_on=["creative"],
            parallel_hints=eval_parallel_hints,
            critical=True,
        ))

        registry.register(StageConfig(
            id="eval_diagnostics",
            description="Evaluate creative diagnostics",
            agent_factory=lambda c: DiagnosticsAgent(c),
            output_schema=DiagnosticsEvaluation,
            output_filename="step_4c_diagnostics.json",
            depends_on=["creative"],
            parallel_hints=eval_parallel_hints,
            critical=True,
        ))

        registry.register(StageConfig(
            id="eval_attention",
            description="Evaluate attention heuristics",
            agent_factory=lambda c: AttentionHeuristicAgent(c),
            output_schema=AttentionEvaluation,
            output_filename="step_4d_attention.json",
            depends_on=["creative"],
            parallel_hints=eval_parallel_hints,
            critical=True,
        ))

        registry.register(StageConfig(
            id="eval_intent",
            description="Evaluate behavioral intent simulation",
            agent_factory=lambda c: IntentSimulationAgent(c),
            output_schema=IntentEvaluation,
            output_filename="step_4e_intent.json",
            depends_on=["creative"],
            parallel_hints=eval_parallel_hints,
            critical=True,
        ))

        # Stage 5: Scoring
        registry.register(StageConfig(
            id="scoring",
            description="Aggregate scores into GO/NO-GO decisions",
            agent_factory=lambda c: ScoringAgent(c),
            output_schema=CompositeReadinessReport,
            output_filename="step_5_scoring.json",
            depends_on=[
                "eval_clarity",
                "eval_brand",
                "eval_diagnostics",
                "eval_attention",
                "eval_intent",
            ],
            critical=True,
        ))

        # Stage 6: Iteration Controller
        registry.register(StageConfig(
            id="iteration",
            description="Generate refinement directives for failing variants",
            agent_factory=lambda c: IterationControllerAgent(c),
            output_schema=IterationControlReport,
            output_filename="step_6_iteration.json",
            depends_on=["scoring"],
            condition=lambda ctx: not all(
                d.get("status") == "GO"
                for d in ctx.get("scoring", {}).get("variant_decisions", [])
            ),
            critical=False,
        ))

        # Post-approval stages (7-12)
        registry.register(StageConfig(
            id="platform_adaptation",
            description="Adapt scripts per platform",
            agent_factory=lambda c: PlatformAdaptationAgent(c),
            output_schema=list[PlatformVariant],
            output_filename="step_7_adaptations.json",
            depends_on=["scoring"],
            condition=lambda ctx: len(ctx.get("approved_variants", [])) > 0,
        ))

        registry.register(StageConfig(
            id="compliance",
            description="Validate legal and brand safety compliance",
            agent_factory=lambda c: ComplianceRiskAgent(c),
            output_schema=ComplianceRiskReport,
            output_filename="step_8_compliance.json",
            depends_on=["platform_adaptation"],
        ))

        registry.register(StageConfig(
            id="asset_production",
            description="Plan shot-by-shot production scenes",
            agent_factory=lambda c: AssetProductionAgent(c),
            output_schema=AssetProductionReport,
            output_filename="step_9_production.json",
            depends_on=["compliance"],
        ))

        registry.register(StageConfig(
            id="validation",
            description="Design A/B test and validation experiments",
            agent_factory=lambda c: ExternalValidationAgent(c),
            output_schema=ExternalValidationPlan,
            output_filename="step_10_validation.json",
            depends_on=["asset_production"],
        ))

        registry.register(StageConfig(
            id="deployment",
            description="Plan launch timeline and scaling triggers",
            agent_factory=lambda c: DeploymentExperimentationAgent(c),
            output_schema=DeploymentExperimentationReport,
            output_filename="step_11_deployment.json",
            depends_on=["validation"],
        ))

        registry.register(StageConfig(
            id="learning",
            description="Capture patterns from historical campaigns",
            agent_factory=lambda c: KnowledgeLearningAgent(c),
            output_schema=KnowledgeLearningReport,
            output_filename="step_12_learning.json",
            depends_on=["deployment"],
        ))

        # Stage 13: Video Generation
        registry.register(StageConfig(
            id="video_generation",
            description="Synthesize Veo prompts and generate final videos",
            agent_factory=lambda c: VideoGenerationAgent(c),
            output_schema=list[VideoGenerationResult],
            output_filename="step_13_video.json",
            depends_on=["learning"],
            condition=lambda ctx: ctx.get("generate_videos", True),
            critical=False,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay=5.0),
        ))

        return registry
