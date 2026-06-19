# Frequently Asked Questions

## General

### What is ads-ai?

ads-ai is a production-grade, multi-agent AI framework for end-to-end advertising campaign generation. It extracts intelligence from product URLs, synthesizes strategic briefs, generates creative variants, evaluates them against quality thresholds, and produces video assets.

### How many agents are in the pipeline?

The pipeline uses **19 specialized agents** covering strategy, audience modeling, creative generation, compliance validation, and video asset production.

### What models does it use?

- **Text reasoning**: Gemini 3.1 Pro (default)
- **Evaluation**: Gemini 3.1 Flash Lite (default)
- **Video generation**: Veo 3.1 (Google's video synthesis model)

All models can be configured via environment variables.

## Setup

### Do I need a Google AI Studio API key?

Yes. The pipeline requires a `GEMINI_API_KEY` for both text generation and video synthesis. Get one at [Google AI Studio](https://aistudio.google.com/apikey).

### What Python version is required?

Python 3.10 or higher. The project uses modern Python features like `X | None` union syntax.

### Can I run this without an API key?

No. All agents require the Gemini API for inference. However, you can run the test suite without an API key:

```bash
pip install -e ".[test]"
pytest tests/ -v
```

## Usage

### What input formats are supported?

1. **URL mode** — Provide a product landing page URL; the pipeline extracts product context automatically.
2. **Explicit mode** — Provide `--product` and `--audience` descriptions directly.

### How long does a pipeline run take?

Typical runs take 5-15 minutes depending on:
- Number of creative variants (3-5 by default)
- Iteration rounds (up to 3)
- Video generation (the most time-consuming step)

### Can I customize the number of iterations?

The iteration loop (stages 4-6) runs up to `max_iterations` times (default: 3). This is configurable in the pipeline configuration.

### What platforms are supported?

The `PlatformAdaptationAgent` can adapt scripts for:
- Meta (Facebook/Instagram)
- TikTok
- YouTube
- Custom platforms via CLI flags

## Output

### Where are generated files saved?

All artifacts are saved to `outputs/run_YYYYMMDD_HHMMSS/` relative to the working directory.

### What file formats are produced?

- **JSON**: Strategy briefs, audience segments, creative variants, evaluation reports
- **MP4**: Final video advertisements (via Veo 3.1)

### Can I customize the output directory?

Yes, use the `--output-dir` flag:

```bash
ads-ai --url "https://example.com" --goal "Sales" --output-dir ./my-campaign
```

## Development

### How do I run tests?

```bash
pip install -e ".[test]"
pytest tests/ -v --cov=ads_ai --cov-report=term-missing
```

### How do I run linting?

```bash
pip install -e ".[lint]"
ruff check ads_ai/ tests/
mypy ads_ai/ tests/ --ignore-missing-imports
```

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on forking, branching, commit conventions, and the pull request process.

## Troubleshooting

### "GEMINI_API_KEY is not set"

Ensure your `.env` file exists and contains a valid API key. The pipeline exits immediately without a key.

### "Rate limit exceeded"

Gemini API has rate limits. The pipeline includes automatic retry logic with exponential backoff. If the issue persists, check your API quota in Google AI Studio.

### Video generation times out

Video generation via Veo has a 300-second timeout. If your video isn't ready:
1. Check your API quota
2. Ensure the model name is correct (`veo-3.1-generate-preview`)
3. Review the logs for specific error messages
