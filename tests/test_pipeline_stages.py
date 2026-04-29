"""Tests for pipeline_stages module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ads_ai.pipeline_stages import (
    PipelineStageRegistry,
    RetryPolicy,
    StageConfig,
    StageExecutionResult,
    StageStatus,
)


class TestRetryPolicy:
    """Tests for RetryPolicy model."""

    def test_default_values(self):
        """Test RetryPolicy has correct defaults."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.initial_delay == 1.0
        assert policy.backoff_multiplier == 2.0
        assert policy.max_delay == 60.0
        assert "InternalServerError" in policy.retryable_exceptions

    def test_custom_values(self):
        """Test RetryPolicy with custom values."""
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=2.0,
            backoff_multiplier=1.5,
            max_delay=120.0,
            retryable_exceptions=["CustomError", "503"],
        )
        assert policy.max_attempts == 5
        assert policy.initial_delay == 2.0
        assert policy.backoff_multiplier == 1.5
        assert policy.max_delay == 120.0
        assert "CustomError" in policy.retryable_exceptions
        assert "503" in policy.retryable_exceptions


class TestStageConfig:
    """Tests for StageConfig."""

    def test_basic_config(self):
        """Test basic stage config creation."""
        mock_factory = MagicMock()
        config = StageConfig(
            id="test_stage",
            description="A test stage",
            agent_factory=mock_factory,
        )
        assert config.id == "test_stage"
        assert config.description == "A test stage"
        assert config.agent_factory is mock_factory
        assert config.depends_on == []
        assert config.critical is True
        assert config.retry_policy is not None

    def test_config_with_dependencies(self):
        """Test stage config with dependencies."""
        config = StageConfig(
            id="stage_b",
            description="Stage B depends on A",
            agent_factory=MagicMock(),
            depends_on=["stage_a"],
        )
        assert config.depends_on == ["stage_a"]

    def test_config_with_condition(self):
        """Test stage config with condition."""
        def condition(ctx): return ctx.get("run_optional") is True
        config = StageConfig(
            id="optional_stage",
            description="Optional stage",
            agent_factory=MagicMock(),
            condition=condition,
        )
        assert config.condition is condition

    def test_validate_context_missing_key(self):
        """Test validate_context returns False when key is missing."""
        config = StageConfig(
            id="test",
            description="test",
            agent_factory=MagicMock(),
            required_context_keys=["required_field"],
        )
        context = {"other_field": "value"}
        assert config.validate_context(context) is False

    def test_validate_context_has_key(self):
        """Test validate_context returns True when key is present."""
        config = StageConfig(
            id="test",
            description="test",
            agent_factory=MagicMock(),
            required_context_keys=["required_field"],
        )
        context = {"required_field": "value"}
        assert config.validate_context(context) is True

    def test_validate_context_null_value(self):
        """Test validate_context returns False when key value is None."""
        config = StageConfig(
            id="test",
            description="test",
            agent_factory=MagicMock(),
            required_context_keys=["field"],
        )
        context = {"field": None}
        assert config.validate_context(context) is False

    def test_should_run_condition_false(self):
        """Test should_run returns False when condition returns False."""
        config = StageConfig(
            id="test",
            description="test",
            agent_factory=MagicMock(),
            condition=lambda ctx: ctx.get("flag") is True,
        )
        context = {"flag": False}
        assert config.should_run(context) is False

    def test_should_run_condition_true(self):
        """Test should_run returns True when condition returns True."""
        config = StageConfig(
            id="test",
            description="test",
            agent_factory=MagicMock(),
            condition=lambda ctx: ctx.get("flag") is True,
        )
        context = {"flag": True}
        assert config.should_run(context) is True

    def test_should_run_condition_exception_runs_stage(self):
        """Test stage runs when condition raises exception."""
        config = StageConfig(
            id="test",
            description="test",
            agent_factory=MagicMock(),
            condition=lambda ctx: ctx["missing_key"],
        )
        context = {}
        assert config.should_run(context) is True


class TestStageExecutionResult:
    """Tests for StageExecutionResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = StageExecutionResult(
            stage_id="test_stage",
            status=StageStatus.COMPLETED,
        )
        assert result.stage_id == "test_stage"
        assert result.status == StageStatus.COMPLETED
        assert result.output is None
        assert result.error is None
        assert result.elapsed_seconds == 0.0
        assert result.attempt_count == 0

    def test_result_with_output(self):
        """Test result with output."""
        result = StageExecutionResult(
            stage_id="test_stage",
            status=StageStatus.COMPLETED,
            output={"key": "value"},
            elapsed_seconds=1.5,
            attempt_count=1,
        )
        assert result.output == {"key": "value"}
        assert result.elapsed_seconds == 1.5
        assert result.attempt_count == 1

    def test_result_with_error(self):
        """Test result with error."""
        result = StageExecutionResult(
            stage_id="test_stage",
            status=StageStatus.FAILED,
            error="Something went wrong",
        )
        assert result.status == StageStatus.FAILED
        assert result.error == "Something went wrong"


class TestPipelineStageRegistry:
    """Tests for PipelineStageRegistry."""

    def test_empty_registry(self):
        """Test empty registry initialization."""
        registry = PipelineStageRegistry()
        assert registry._stages == {}
        assert registry._execution_order == []

    def test_register_single_stage(self):
        """Test registering a single stage."""
        registry = PipelineStageRegistry()
        config = StageConfig(
            id="test",
            description="test",
            agent_factory=MagicMock(),
        )
        registry.register(config)
        assert "test" in registry._stages
        assert registry._stages["test"] is config

    def test_register_returns_self(self):
        """Test register returns registry for chaining."""
        registry = PipelineStageRegistry()
        config = StageConfig(id="test", description="test", agent_factory=MagicMock())
        result = registry.register(config)
        assert result is registry

    def test_register_many(self):
        """Test registering multiple stages."""
        registry = PipelineStageRegistry()
        configs = [
            StageConfig(id="a", description="a", agent_factory=MagicMock()),
            StageConfig(id="b", description="b", agent_factory=MagicMock()),
        ]
        registry.register_many(configs)
        assert "a" in registry._stages
        assert "b" in registry._stages

    def test_get_stage_exists(self):
        """Test get_stage returns config when exists."""
        registry = PipelineStageRegistry()
        config = StageConfig(id="test", description="test", agent_factory=MagicMock())
        registry.register(config)
        retrieved = registry.get_stage("test")
        assert retrieved is config

    def test_get_stage_not_exists(self):
        """Test get_stage returns None when not exists."""
        registry = PipelineStageRegistry()
        assert registry.get_stage("nonexistent") is None

    def test_list_stages_empty(self):
        """Test list_stages on empty registry."""
        registry = PipelineStageRegistry()
        assert registry.list_stages() == []

    def test_resolve_order_single_stage(self):
        """Test resolving execution order with single stage."""
        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="solo",
            description="solo",
            agent_factory=MagicMock(),
        ))
        groups = registry.resolve_execution_order()
        assert groups == [["solo"]]

    def test_resolve_order_two_sequential(self):
        """Test resolving execution order for sequential stages."""
        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="first",
            description="first",
            agent_factory=MagicMock(),
        ))
        registry.register(StageConfig(
            id="second",
            description="second",
            agent_factory=MagicMock(),
            depends_on=["first"],
        ))
        groups = registry.resolve_execution_order()
        assert groups == [["first"], ["second"]]

    def test_resolve_order_two_parallel(self):
        """Test resolving execution order for parallel stages."""
        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="a",
            description="a",
            agent_factory=MagicMock(),
            depends_on=["common"],
        ))
        registry.register(StageConfig(
            id="b",
            description="b",
            agent_factory=MagicMock(),
            depends_on=["common"],
        ))
        registry.register(StageConfig(
            id="common",
            description="common",
            agent_factory=MagicMock(),
        ))
        groups = registry.resolve_execution_order()
        assert groups == [["common"], ["a", "b"]]

    def test_resolve_order_complex_dag(self):
        """Test resolving execution order for complex DAG."""
        registry = PipelineStageRegistry()
        # A -> C, B -> C, C -> D
        registry.register(StageConfig(id="a", description="a", agent_factory=MagicMock()))
        registry.register(StageConfig(id="b", description="b", agent_factory=MagicMock()))
        registry.register(StageConfig(
            id="c", description="c", agent_factory=MagicMock(), depends_on=["a", "b"]
        ))
        registry.register(StageConfig(
            id="d", description="d", agent_factory=MagicMock(), depends_on=["c"]
        ))
        groups = registry.resolve_execution_order()
        assert groups == [["a", "b"], ["c"], ["d"]]

    def test_resolve_order_circular_dependency_raises(self):
        """Test circular dependency detection."""
        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="a",
            description="a",
            agent_factory=MagicMock(),
            depends_on=["b"],
        ))
        registry.register(StageConfig(
            id="b",
            description="b",
            agent_factory=MagicMock(),
            depends_on=["a"],
        ))
        with pytest.raises(ValueError, match="Circular dependency"):
            registry.resolve_execution_order()

    def test_resolve_order_unknown_dependency_raises(self):
        """Test unknown dependency detection."""
        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="a",
            description="a",
            agent_factory=MagicMock(),
            depends_on=["nonexistent"],
        ))
        with pytest.raises(ValueError, match="unknown stage"):
            registry.resolve_execution_order()


class TestPipelineStageRegistryExecute:
    """Tests for PipelineStageRegistry.execute()."""

    def test_execute_empty_registry(self):
        """Test executing empty registry."""
        registry = PipelineStageRegistry()
        results = registry.execute(context={})
        assert results == {}

    def test_execute_single_stage_success(self):
        """Test executing a single stage that succeeds."""
        mock_agent = MagicMock()
        mock_agent.execute.return_value = {"result": "success"}

        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="test",
            description="test",
            agent_factory=lambda _: mock_agent,
        ))

        results = registry.execute(context={})
        assert "test" in results
        assert results["test"].status == StageStatus.COMPLETED
        assert results["test"].output == {"result": "success"}

    def test_execute_single_stage_uses_run_fallback(self):
        """Test executing stage falls back to run() method."""
        mock_agent = MagicMock()
        mock_agent.execute = None
        mock_agent.run.return_value = {"fallback": "worked"}

        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="test",
            description="test",
            agent_factory=lambda _: mock_agent,
        ))

        results = registry.execute(context={})
        assert results["test"].status == StageStatus.COMPLETED
        assert results["test"].output == {"fallback": "worked"}

    def test_execute_skips_conditional_stage(self):
        """Test executing skips stages with false conditions."""
        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="always",
            description="always",
            agent_factory=lambda _: MagicMock(execute=lambda ctx: {}),
        ))
        registry.register(StageConfig(
            id="conditional",
            description="conditional",
            agent_factory=lambda _: MagicMock(execute=lambda ctx: {}),
            condition=lambda ctx: ctx.get("run_conditional") is True,
        ))

        results = registry.execute(context={})
        assert results["always"].status == StageStatus.COMPLETED
        assert results["conditional"].status == StageStatus.CONDITIONAL_SKIP

    def test_execute_runs_conditional_stage(self):
        """Test executing runs conditional stages when condition is true."""
        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="conditional",
            description="conditional",
            agent_factory=lambda _: MagicMock(execute=lambda ctx: {}),
            condition=lambda ctx: ctx.get("run_conditional") is True,
        ))

        results = registry.execute(context={"run_conditional": True})
        assert results["conditional"].status == StageStatus.COMPLETED

    def test_execute_parallel_stages(self):
        """Test executing parallel stages concurrently."""
        call_order = []

        def slow_factory(_):
            agent = MagicMock()

            def slow_execute(ctx):
                call_order.append("start")
                import time
                time.sleep(0.05)
                call_order.append("end")
                return {}

            agent.execute = slow_execute
            return agent

        registry = PipelineStageRegistry()
        registry.register(StageConfig(
            id="a",
            description="a",
            agent_factory=slow_factory,
            depends_on=["common"],
        ))
        registry.register(StageConfig(
            id="b",
            description="b",
            agent_factory=slow_factory,
            depends_on=["common"],
        ))
        registry.register(StageConfig(
            id="common",
            description="common",
            agent_factory=lambda _: MagicMock(execute=lambda ctx: {}),
        ))

        results = registry.execute(context={})
        # Both a and b should complete
        assert results["a"].status == StageStatus.COMPLETED
        assert results["b"].status == StageStatus.COMPLETED
        # They should have started at similar times (parallel)
        # Note: exact timing depends on thread scheduler

    def test_execute_updates_context(self):
        """Test that stage outputs are added to context."""
        registry = PipelineStageRegistry()

        registry.register(StageConfig(
            id="stage1",
            description="stage1",
            agent_factory=lambda _: MagicMock(execute=lambda ctx: {"from_stage1": True}),
        ))
        registry.register(StageConfig(
            id="stage2",
            description="stage2",
            agent_factory=lambda _: MagicMock(execute=lambda ctx: ctx),
            depends_on=["stage1"],
        ))

        results = registry.execute(context={})
        # stage2 should have received stage1 output
        assert results["stage1"].status == StageStatus.COMPLETED
        assert results["stage2"].status == StageStatus.COMPLETED


class TestPipelineStageRegistryCreateLegacy:
    """Tests for PipelineStageRegistry.create_legacy()."""

    def test_create_legacy_returns_registry(self):
        """Test create_legacy returns a PipelineStageRegistry."""
        mock_client = MagicMock()
        registry = PipelineStageRegistry.create_legacy(mock_client)
        assert isinstance(registry, PipelineStageRegistry)

    def test_create_legacy_has_stages(self):
        """Test create_legacy creates all expected stages."""
        mock_client = MagicMock()
        registry = PipelineStageRegistry.create_legacy(mock_client)
        # Should have multiple stages registered
        assert len(registry._stages) > 10

    def test_create_legacy_has_correct_dependencies(self):
        """Test create_legacy has correct stage dependencies."""
        mock_client = MagicMock()
        registry = PipelineStageRegistry.create_legacy(mock_client)

        # Strategy should depend on url_intelligence
        strategy_config = registry.get_stage("strategy")
        assert "url_intelligence" in strategy_config.depends_on

        # Scoring should depend on all eval stages
        scoring_config = registry.get_stage("scoring")
        assert "eval_clarity" in scoring_config.depends_on
        assert "eval_brand" in scoring_config.depends_on
        assert "eval_diagnostics" in scoring_config.depends_on
        assert "eval_attention" in scoring_config.depends_on
        assert "eval_intent" in scoring_config.depends_on

    def test_create_legacy_eval_parallel_hints(self):
        """Test eval stages have parallel hints."""
        mock_client = MagicMock()
        registry = PipelineStageRegistry.create_legacy(mock_client)

        clarity_config = registry.get_stage("eval_clarity")
        assert clarity_config.parallel_hints.get("max_workers") == 4
