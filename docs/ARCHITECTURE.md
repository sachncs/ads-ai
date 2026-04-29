# Architecture Overview

## System Design

Ads.ai uses a **centralized orchestrator** pattern where a single `OrchestratorPipeline` coordinates state and data flow between 19 specialized agents.

## Pipeline Stages

The pipeline executes in 13 sequential stages (with iterative loops between stages 4-6):

| Stage | Agent | Purpose |
|-------|-------|---------|
| 0 | URLIntelligenceAgent | Extract product context from a landing page URL |
| 0.5 | BudgetInferenceAgent | Auto-infer budget when "TBD" is provided |
| 1 | StrategyAgent | Generate strategy brief with KPIs and pillars |
| 2 | AudienceAgent | Model behavioral personas and segments |
| 3 | CreativeAgent | Generate 3-5 ad script variants |
| 4 | Evaluation Agents (x5) | Parallel evaluation of clarity, brand, diagnostics, attention, intent |
| 5 | ScoringAgent | Aggregate scores into GO/NO-GO decisions |
| 6 | IterationControllerAgent | Generate refinement directives for failing variants |
| 7 | PlatformAdaptationAgent | Adapt scripts per platform (Meta, TikTok, etc.) |
| 8 | ComplianceRiskAgent | Validate legal and brand safety compliance |
| 9 | AssetProductionAgent | Plan shot-by-shot production scenes |
| 10 | ExternalValidationAgent | Design A/B test and validation experiments |
| 11 | DeploymentExperimentationAgent | Plan launch timeline and scaling triggers |
| 12 | KnowledgeLearningAgent | Capture patterns from historical campaigns |
| 13 | VideoGenerationAgent | Synthesize Veo prompts and generate final videos |

## Iteration Loop

Stages 4-6 form an **evaluate-iterate cycle**:

1. All variants are evaluated in parallel (max 4 concurrent threads).
2. ScoringAgent computes a composite readiness score per variant.
3. If any variant scores below the threshold, the IterationControllerAgent produces targeted refinement instructions.
4. The CreativeAgent refines only the failing variants.
5. The cycle repeats up to `max_iterations` (default: 3).

## Key Abstractions

### BaseAgent

Every agent inherits from `BaseAgent`, which provides:

- **LLM Interface**: `generate(prompt, response_schema=None)` with automatic retry on 503 errors.
- **JSON Recursion**: `to_json_dict(obj)` recursively serializes Pydantic models, lists, and dicts.
- **Logging**: Structured `logger.info()` / `logger.exception()` with `perf_counter` timing.

### OrchestratorPipeline

The orchestrator:

1. Initializes all 19 agents with a shared `genai.Client`.
2. Runs stages sequentially, saving artifacts to `outputs/run_YYYYMMDD_HHMMSS/`.
3. Handles fallback logic when no variants pass the GO threshold (selects highest-scoring variant).
4. Parallelizes variant evaluation using `ThreadPoolExecutor`.

### Artifact Management

Each run produces:

```
outputs/
  run_20240429_143052/
    step_1_strategy.json
    step_2_audience.json
    step_3_creative_base.json
    iteration_0_variants.json
    iteration_1_variants.json
    ...
    step_7_adaptations.json
    step_8_compliance.json
    step_9_production.json
    step_10_validation.json
    step_11_deployment.json
    step_12_learning.json
    pipeline_final_report.json
    ad_product_0_20240429_143052.mp4
    ad_product_1_20240429_143052.mp4
```

## Error Handling

- **Agent-level**: Each agent defines a typed exception (e.g., `VideoGenerationError`).
- **Pipeline-level**: The orchestrator catches exceptions and either falls back or terminates gracefully.
- **CLI-level**: `main.py` returns distinct exit codes for missing API key, missing args, and pipeline failure.
