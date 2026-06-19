# Agents Overview

This directory contains the 19 specialized agents that power the ads-ai pipeline. Each agent inherits from `BaseAgent` and implements a single responsibility within the campaign generation workflow.

## Agent Categories

### Intelligence & Strategy

| Agent | Module | Purpose |
|-------|--------|---------|
| **URLIntelligenceAgent** | `intelligence.py` | Extracts product context from a landing page URL |
| **BudgetInferenceAgent** | `budget.py` | Auto-infers budget constraints when not explicitly provided |
| **StrategyAgent** | `strategy.py` | Generates master strategy brief with KPIs and pillars |

### Audience & Creative

| Agent | Module | Purpose |
|-------|--------|---------|
| **AudienceAgent** | `audience.py` | Models behavioral personas and intent simulation |
| **CreativeAgent** | `creative.py` | Generates 3-5 ad script variants with scene breakdowns |

### Evaluation (Parallel)

| Agent | Module | Purpose |
|-------|--------|---------|
| **MessageClarityAgent** | `clarity.py` | Evaluates message clarity and comprehension |
| **BrandLinkageAgent** | `brand.py` | Evaluates brand recall and attribution strength |
| **DiagnosticsAgent** | `diagnostics.py` | Checks for logical consistency and plot holes |
| **AttentionHeuristicAgent** | `attention.py` | Scores hook strength and opening impact |
| **IntentSimulationAgent** | `intent.py` | Simulates user response and conversion likelihood |

### Scoring & Iteration

| Agent | Module | Purpose |
|-------|--------|---------|
| **ScoringAgent** | `scoring.py` | Aggregates evaluation scores into GO/NO-GO decisions |
| **IterationControllerAgent** | `iteration.py` | Generates refinement directives for failing variants |

### Post-Production

| Agent | Module | Purpose |
|-------|--------|---------|
| **PlatformAdaptationAgent** | `adaptation.py` | Adapts scripts per platform (Meta, TikTok, YouTube) |
| **ComplianceRiskAgent** | `compliance.py` | Validates legal and brand safety compliance |
| **AssetProductionAgent** | `asset.py` | Plans shot-by-shot production scenes |
| **ExternalValidationAgent** | `validation.py` | Designs A/B test and validation experiments |
| **DeploymentExperimentationAgent** | `deployment.py` | Plans launch timeline and scaling triggers |
| **KnowledgeLearningAgent** | `learning.py` | Captures patterns from historical campaigns |

### Video Generation

| Agent | Module | Purpose |
|-------|--------|---------|
| **VideoGenerationAgent** | `video.py` | Synthesizes Veo prompts and generates final videos |

## BaseAgent

All agents inherit from `BaseAgent` (`base.py`), which provides:

- **LLM Interface**: `generate(prompt, response_schema=None)` with automatic retry on 503 errors
- **JSON Serialization**: `to_json_dict(obj)` for recursive Pydantic model serialization
- **Logging**: Structured logging with `perf_counter` timing
- **Type Safety**: `@overload` decorators for precise return types

## Models

All agent input/output schemas are defined in `models.py` using Pydantic v2. The module exports 40+ models including:

- `StrategyBrief`, `AudienceSegments`, `CreativeVariants`
- `ClarityEvaluation`, `BrandLinkageEvaluation`, `AttentionEvaluation`
- `CompositeReadinessReport`, `IterationDirective`
- `VideoGenerationResult`, `AssetProductionReport`

## Usage

```python
from ads_ai.agents import StrategyAgent, CreativeAgent
from google import genai

client = genai.Client(api_key="your-key")
strategy = StrategyAgent(client)
creative = CreativeAgent(client)
```
