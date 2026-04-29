# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test coverage raised to >90%.
- Added `pytest-cov` to test dependencies with 80% minimum coverage gate in CI.
- Added dedicated test suites for `main.py` CLI entry point and `VideoGenerationAgent`.
- Added `ruff` linting and `mypy` type checking to CI workflow.

### Changed
- CI workflow triggers on `master` branch (was incorrectly set to `main`).
- `BaseAgent.generate()` overloads now use a generic `TypeVar` for precise return types.
- `save_json()` protocol simplified to accept `Any`, resolving Pydantic v2 compatibility issues.
- `capture_learnings()` and `design_validation()` signatures relaxed to accept `list[Any]` for evaluations.

### Fixed
- Fixed 64 `ruff` lint errors across the codebase.
- Fixed 55 `mypy` type errors across 20 source files.
- Removed obsolete `pytest-asyncio` and `asyncio_mode` configuration (no async tests present).

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
