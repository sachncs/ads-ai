[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/sachncs/ads-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/ads-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/sachncs/ads-ai/releases)

# Ads.ai — Multi-Agent Advertising Intelligence Pipeline

A production-grade, multi-agent AI framework for end-to-end advertising campaign generation. The system extracts intelligence from product URLs, synthesizes strategic briefs, generates creative variants, evaluates them against quantifiable quality thresholds, and produces video assets ready for deployment.

## Features

- **13-Step Automated Pipeline** — Programmatic evaluation gates ensure every ad variant meets brand, clarity, and intent thresholds before proceeding to production
- **19 Specialized Agents** — Domain-expert agents for strategy, audience modeling, creative generation, compliance validation, and asset production
- **Veo 3.1 Video Synthesis** — Direct integration with Google's Veo 3.1 model for generating high-fidelity video advertisements from structured scripts
- **Quantified Quality Enforcement** — Built-in [Quality Standards](docs/QUALITY_STANDARDS.md) that gate creative production based on measurable, AI-evaluated metrics
- **Dynamic Pipeline Stages** — `PipelineStageRegistry` with declarative configurations, dependency resolution, topological sorting, and configurable retry policies

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Google AI Studio API key ([Get one here](https://aistudio.google.com/apikey))

### Installation

```bash
git clone https://github.com/sachncs/ads-ai.git
cd ads-ai
pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### Run Your First Campaign

```bash
# URL mode — extracts product context automatically
ads-ai --url "https://example.com/product" --goal "Maximize Sales"

# Explicit mode — provide product and audience directly
ads-ai --product "Ergonomic Chair" --audience "Remote Workers" --goal "Drive Conversions"
```

For detailed setup instructions, see [docs/getting-started.md](docs/getting-started.md).

## Architecture

```mermaid
graph TD
    User([User Input]) --> URL[URL Intelligence Agent]
    URL --> Budget[Budget Inference Agent]
    Budget --> Strategy[Strategy Agent]
    Strategy --> Audience[Audience Agent]
    Audience --> Creative[Creative Agent]
    Creative --> Eval[Evaluation Agents x5]
    Eval -- Feedback --> Iteration[Iteration Controller]
    Iteration --> Creative
    Eval -- Success --> Adaptation[Platform Adaptation]
    Adaptation --> Compliance[Compliance Agent]
    Compliance --> Prod[Asset Production Agent]
    Prod --> Video[Video Generation Agent]
    Video --> Output([Final Campaign Assets])
```

| Stage | Agent | Purpose |
|-------|-------|---------|
| 0 | URLIntelligenceAgent | Extract product context from a URL |
| 0.5 | BudgetInferenceAgent | Auto-infer budget when not provided |
| 1 | StrategyAgent | Generate strategy brief with KPIs |
| 2 | AudienceAgent | Model behavioral personas |
| 3 | CreativeAgent | Generate 3-5 ad script variants |
| 4 | Evaluation Agents (x5) | Parallel quality evaluation |
| 5 | ScoringAgent | Composite GO/NO-GO scoring |
| 6 | IterationControllerAgent | Refinement directives for failing variants |
| 7 | PlatformAdaptationAgent | Platform-specific adaptation |
| 8 | ComplianceRiskAgent | Legal and brand safety compliance |
| 9 | AssetProductionAgent | Shot-by-shot production planning |
| 10 | ExternalValidationAgent | A/B test design |
| 11 | DeploymentExperimentationAgent | Launch timeline and scaling |
| 12 | KnowledgeLearningAgent | Pattern capture from campaigns |
| 13 | VideoGenerationAgent | Veo 3.1 video synthesis |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture deep-dive.

## Agent Workforce

| Agent | Responsibility | Output |
|:------|:---------------|:-------|
| **StrategyAgent** | Strategic pillars, KPIs, and pre-release targets | `StrategyBrief` |
| **AudienceAgent** | Behavioral persona modeling and intent simulation | `AudienceSegments` |
| **CreativeAgent** | Narrative structure, hooks, and visual cue design | `CreativeVariants` |
| **ScoringAgent** | Multi-dimensional readiness gating (GO/NO-GO) | `CompositeReadinessReport` |
| **VideoGenAgent** | Cinematic prompt synthesis and Veo rendering | `VideoGenerationResult` |
| *...and 14 more* | See [Agents Overview](ads_ai/agents/README.md) | |

## Project Structure

```
ads-ai/
├── ads_ai/                    # Main package
│   ├── __init__.py            # Version and public API
│   ├── config.py              # Settings from environment variables
│   ├── main.py                # CLI entry point
│   ├── pipeline.py            # OrchestratorPipeline
│   ├── pipeline_stages.py     # Dynamic stage registry
│   ├── agents/                # 19 specialized agents
│   │   ├── base.py            # BaseAgent with LLM interface
│   │   ├── models.py          # 40+ Pydantic models
│   │   ├── strategy.py        # Strategy generation
│   │   ├── creative.py        # Ad script generation
│   │   ├── video.py           # Veo 3.1 integration
│   │   └── ...                # Additional agents
│   └── utils/                 # Shared utilities
│       ├── file_ops.py        # Artifact management
│       ├── prompts.py         # Prompt templates
│       └── timing.py          # Performance instrumentation
├── tests/                     # Test suite (126+ tests)
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System design
│   ├── QUALITY_STANDARDS.md   # Agent quality expectations
│   ├── getting-started.md     # Setup guide
│   └── faq.md                 # Common questions
├── .github/                   # GitHub configuration
│   ├── workflows/ci.yml       # CI pipeline
│   ├── dependabot.yml         # Dependency updates
│   ├── ISSUE_TEMPLATE/        # Issue templates
│   └── PULL_REQUEST_TEMPLATE.md
├── pyproject.toml             # Project metadata and dependencies
├── .env.example               # Environment variable template
├── .editorconfig              # Editor configuration
├── .gitattributes             # Git line ending normalization
├── .gitignore                 # Git ignore rules
├── CONTRIBUTING.md            # Contribution guidelines
├── CODE_OF_CONDUCT.md         # Community code of conduct
├── SECURITY.md                # Security policy
├── CHANGELOG.md               # Release history
└── LICENSE                    # MIT License
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| LLM Framework | Google GenAI (Gemini 3.1 Pro / Flash Lite) |
| Video Generation | Google Veo 3.1 |
| Data Validation | Pydantic v2 |
| Settings | pydantic-settings |
| Build System | Hatchling |
| Linting | Ruff |
| Type Checking | mypy |
| Testing | pytest + pytest-cov |

## Development

### Commands

| Command | Description |
|---------|-------------|
| `pip install -e .` | Install in editable mode |
| `pip install -e ".[dev,test,lint]"` | Install with all dev dependencies |
| `ads-ai --url ... --goal ...` | Run the pipeline |
| `pytest tests/ -v` | Run test suite |
| `pytest tests/ -v --cov=ads_ai` | Run tests with coverage |
| `ruff check ads_ai/ tests/` | Lint code |
| `mypy ads_ai/ --ignore-missing-imports` | Type check |

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run the full suite with coverage
pytest tests/ -v --cov=ads_ai --cov-report=term-missing

# Run standalone tests (no API key required)
python3 tests/standalone_runner.py
```

### Linting and Type Checking

```bash
# Install lint dependencies
pip install -e ".[lint]"

# Lint
ruff check ads_ai/ tests/

# Type check
mypy ads_ai/ --ignore-missing-imports
```

## Roadmap

- [ ] Add support for additional video generation providers
- [ ] Implement campaign performance analytics dashboard
- [ ] Add multi-language support for ad generation
- [ ] Create web UI for non-technical users
- [ ] Add support for image ad generation
- [ ] Implement A/B test result analysis agent
- [ ] Add cost estimation and budget optimization
- [ ] Support batch campaign generation

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Fork and branch workflow
- Commit conventions ([Conventional Commits](https://www.conventionalcommits.org/))
- Pull request process
- Coding standards and quality requirements

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). Please read it before participating.

## Security

For reporting vulnerabilities, see [SECURITY.md](SECURITY.md).

## Community

- [GitHub Discussions](https://github.com/sachncs/ads-ai/discussions) — Ask questions, share ideas, and discuss the project
- [Issue Tracker](https://github.com/sachncs/ads-ai/issues) — Report bugs and request features

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
