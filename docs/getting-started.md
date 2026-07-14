# Getting Started

This guide walks you through setting up and running your first ad generation pipeline.

## Prerequisites

- **Python 3.10+** — Check with `python3 --version`
- **Google AI Studio API key** — Get one at [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

```bash
# Clone the repository
git clone https://github.com/sachncs/ads-ai.git
cd ads-ai

# Install in editable mode
pip install -e .
```

For development with testing and linting tools:

```bash
pip install -e ".[dev,test,lint]"
```

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and set your API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

3. Optionally configure models and logging:

```bash
DEFAULT_TEXT_MODEL=gemini-3.1-pro-preview
DEFAULT_EVALUATION_MODEL=gemini-3.1-flash-lite-preview
DEFAULT_VIDEO_MODEL=veo-3.1-generate-preview
LOG_LEVEL=INFO
```

## Running Your First Campaign

### URL Mode (Recommended)

Provide a product URL and let the pipeline extract context automatically:

```bash
ads-ai --url "https://example.com/product" --goal "Maximize Sales"
```

### Explicit Mode

Provide product and audience details directly:

```bash
ads-ai \
  --product "Ergonomic Standing Desk" \
  --audience "Remote workers aged 25-40" \
  --goal "Drive website conversions"
```

### With Platform Targeting

```bash
ads-ai \
  --url "https://example.com" \
  --goal "Brand Awareness" \
  --platforms TikTok Meta YouTube
```

### With Budget and Brand Assets

```bash
ads-ai \
  --url "https://example.com" \
  --goal "Maximize Sales" \
  --budget "$50,000" \
  --brand-colors "Blue #0000FF, Gold #FFD700" \
  --brand-images "Modern minimalist logo, product lifestyle shots"
```

## Output

Each pipeline run generates artifacts in a timestamped directory:

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
    pipeline_final_report.json
    ad_product_0_20240429_143052.mp4
```

## Next Steps

- Read the [Architecture](ARCHITECTURE.md) document to understand how the pipeline works
- Review [Quality Standards](QUALITY_STANDARDS.md) for agent-specific expectations
- See the [FAQ](faq.md) for common questions
