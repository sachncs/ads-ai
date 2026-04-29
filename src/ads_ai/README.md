# ads_ai

The core module for the AI-driven multi-agent advertising pipeline.

## Overview

`ads_ai` provides an end-to-end pipeline for generating, validating, and deploying advertising campaigns using multiple specialized AI agents.

## Submodules

### `agents/`

Specialized agents for different aspects of campaign generation:
- **adaptation** — Adaptive learning and performance-based adjustments
- **asset** — Creative asset generation and management
- **attention** — Audience attention and engagement optimization
- **audience** — Target audience definition and segmentation
- **base** — Base agent interface and utilities
- **brand** — Brand consistency and voice validation
- **budget** — Budget allocation and optimization
- **clarity** — Message clarity evaluation
- **compliance** — Policy and compliance checking
- **creative** — Creative copy and visuals generation
- **deployment** — Campaign deployment and monitoring
- **diagnostics** — Pipeline diagnostics and health checks
- **intelligence** — Market intelligence and competitive analysis
- **intent** — User intent detection and mapping
- **iteration** — Campaign iteration and improvement
- **learning** — Continuous learning from campaign data
- **models** — Data models and schemas
- **scoring** — Campaign performance scoring
- **strategy** — Campaign strategy formulation
- **validation** — Campaign validation and quality assurance
- **video** — Video ad generation and optimization

### `utils/`

Utility modules:
- **file_ops** — File operations and serialization for run artifacts

### `pipeline.py`

Main `OrchestratorPipeline` class that coordinates all agents.

### `config.py`

Configuration management for the pipeline.

## Usage

```python
from google import genai
from ads_ai.pipeline import OrchestratorPipeline

client = genai.Client(api_key="your-api-key")
pipeline = OrchestratorPipeline(client)

# Run from explicit inputs
result = pipeline.run(
    product="Ergonomic Office Chair",
    goal="Maximize Q4 Sales",
    audience_desc="Remote workers aged 25-45 with back pain",
    platforms=["Meta", "TikTok"],
)

# Or run from a product URL
result = pipeline.run_from_url(
    url="https://example.com/product",
    goal="Maximize Q4 Sales",
    platforms=["Meta", "TikTok"],
)
```

## Requirements

- Python >= 3.10
- google-genai >= 0.1.0
- pydantic >= 2.0
- pydantic-settings >= 2.0