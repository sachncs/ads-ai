# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-04-29

### Added
- `PipelineStageRegistry`: Dynamic stage registry with declarative stage configurations,
  dependency resolution, topological sorting, and parallel execution hints.
- `StageConfig`: Configurable stages with dependencies, conditions, retry policies, and
  per-stage timeout settings.
- `PipelineStageRegistry.create_legacy()`: Class method factory for the existing 13-step
  pipeline using the new dynamic stage system.
- `RetryPolicy` model: Configurable retry with backoff, max attempts, and retryable
  exception filtering.
- `StageExecutionResult` and `StageStatus`: Structured output for pipeline execution.
- `test_pipeline_stages.py`: Comprehensive test suite with 38 test cases for the new
  pipeline stages module.

### Enhanced
- All agent prompts now include chain-of-thought reasoning traces for systematic problem-solving.
- Few-shot examples added to StrategyAgent, CreativeAgent, AudienceAgent, ScoringAgent,
  IterationControllerAgent, and URLIntelligenceAgent.
- Output validation checklists added to agent prompts to reduce malformed outputs.
- Brand safety guardrails added to CreativeAgent prompts.
- Contextual weighting logic added to ScoringAgent based on strategy goal type.

### Changed
- `OrchestratorPipeline` remains backward-compatible with existing code while the new
  `PipelineStageRegistry` provides an alternative execution model.
- `PipelineStageRegistry.create_legacy` is now a class method instead of standalone function.
- All ruff linting issues fixed across the codebase.
- Comprehensive test coverage raised to >90% (126 tests).
- CI workflow triggers on `master` branch.

### Fixed
- `BaseAgent.generate()` overloads now use a generic `TypeVar` for precise return types.
- `save_json()` protocol simplified to accept `Any`, resolving Pydantic v2 compatibility issues.
- `capture_learnings()` and `design_validation()` signatures relaxed to accept `list[Any]`.

## [0.1.1] - 2024-04-28

### Fixed
- Patch release with minor dependency updates.

## [0.1.0] - 2024-04-04

### Added
- Initial release of the ads-ai multi-agent advertising pipeline.
- 19 specialized agents covering strategy, creative, evaluation, compliance, and video generation.
- Veo 3.1 integration for cinematic video ad generation.
- 13-step orchestrated pipeline with iterative quality gating.
- Pydantic v2 models for strict schema validation across all agent outputs.
