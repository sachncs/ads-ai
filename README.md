# Ads.ai - Multi-Agent Advertising Intelligence Pipeline

Ads.ai is a production-grade, multi-agent AI framework for end-to-end advertising campaign generation. The system extracts intelligence from product URLs, synthesizes strategic briefs, generates creative variants, evaluates them against quantifiable quality thresholds, and produces video assets ready for deployment.

## Key Features

- **13-Step Automated Pipeline**: Programmatic evaluation gates ensure every ad variant meets brand, clarity, and intent thresholds before proceeding to production.
- **19 Specialized Agents**: Domain-expert agents for strategy, audience modeling, creative generation, compliance validation, and asset production.
- **Veo 3.1 Video Synthesis**: Direct integration with Google's Veo 3.1 model for generating high-fidelity video advertisements from structured scripts.
- **Quantified Quality Enforcement**: Built-in [Quality Standards](docs/QUALITY_STANDARDS.md) that gate creative production based on measurable, AI-evaluated metrics.

## Technical Architecture

Ads.ai employs a recursive orchestration pattern where a central `OrchestratorPipeline` manages state and data flow between specialized agents through a strict 13-step workflow.

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

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Google AI Studio API key (for Gemini and Veo models)

### Installation

```bash
# Clone the repository
git clone https://github.com/sachn-cs/ads-ai.git
cd ads-ai

# Install in editable mode
pip install -e .
```

### Environment Configuration

Create a `.env` file from the provided example:

```bash
cp .env.example .env
```

Key environment variables:

- `GEMINI_API_KEY`: Your Google GenAI API key.
- `DEFAULT_TEXT_MODEL`: Default model for agent reasoning (default: `gemini-3.1-pro-preview`).
- `DEFAULT_VIDEO_MODEL`: Model for video generation (default: `veo-3.1-generate-preview`).

## Agent Workforce

| Agent | Responsibility | Output |
| :--- | :--- | :--- |
| **StrategyAgent** | Strategic pillars, KPIs, and pre-release targets. | `StrategyBrief` |
| **AudienceAgent** | Behavioral persona modeling and intent simulation. | `AudienceSegments` |
| **CreativeAgent** | Narrative structure, hooks, and visual cue design. | `CreativeVariants` |
| **ScoringAgent** | Multi-dimensional readiness gating (GO/NO-GO). | `CompositeReadinessReport` |
| **VideoGenAgent** | Cinematic prompt synthesis and Veo rendering. | `VideoGenerationResult` |
| *...and 14 more* | See [Agents Overview](ads_ai/agents/README.md) | |

## Quality and Testing

Ads.ai maintains a rigorous test suite with high code coverage, focusing on:

- **Pydantic Validation**: 40+ models enforce strict schema integrity across all agent inputs and outputs.
- **Quantified Quality Metrics**: Tests verify "Cinematic Markers" and "Hook Strength" in agent outputs.
- **Standalone Verification**: Run tests independently without external API dependencies:
    ```bash
    python3 tests/standalone_runner.py
    ```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run the full suite with coverage
pytest tests/ -v --cov=ads_ai --cov-report=term-missing
```

### Linting and Type Checking

```bash
pip install -e ".[lint]"
ruff check ads_ai tests/
mypy ads_ai --ignore-missing-imports
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
