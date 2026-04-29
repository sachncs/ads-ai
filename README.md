# Ads.ai - Multi-Agent Advertising Intelligence Pipeline

![Ads.ai Header](https://source.unsplash.com/1600x400/?technology,ai,advertising)

Ads.ai is a state-of-the-art, multi-agent AI framework designed to automate the entire advertising lifecycle. From raw URL intelligence to cinematic video production using **Veo 3.1**, Ads.ai orchestrates a specialized workforce of 19 AI agents to deliver high-performance creative assets.

## 🚀 Key Features

*   **13-Step Automated Gating**: Programmatic loops ensure every ad variant meets strict brand and clarity thresholds before proceeding.
*   **19 Specialized Agents**: Domain-expert agents for Strategy, Audience, Creative, Compliance, and more.
*   **Cinematic Video Synthesis**: Direct integration with **Veo 3.1** for generating high-fidelity video ads from structured scripts.
*   **Quantified Quality Enforcement**: Built-in [Quality Standards](docs/QUALITY_STANDARDS.md) that gate creative production based on measurable metrics.

## 🏗️ Technical Architecture

Ads.ai uses a recursive orchestration pattern where a central `OrchestratorPipeline` manages the state and data flow between specialized agents.

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

## 🛠️ Getting Started

### 1. Prerequisites
*   Python 3.10+
*   Google AI Studio API Key (for Gemini and Veo)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/sachn-cs/ads-ai.git
cd ads-ai

# Install in editable mode
pip install -e .
```

### 3. Environment Configuration
Create a `.env` file from the provided example:
```bash
cp .env.example .env
```
Key variables:
*   `GEMINI_API_KEY`: Your Google GenAI API key.
*   `DEFAULT_TEXT_MODEL`: Default model for agent reasoning (default: `gemini-3.1-pro-preview`).
*   `DEFAULT_VIDEO_MODEL`: Model for video generation (default: `veo-3.1-generate-preview`).

## 🤖 The Agent Workforce

| Agent | Responsibility | Output |
| :--- | :--- | :--- |
| **StrategyAgent** | Strategic pillars, KPIs, and pre-release targets. | `StrategyBrief` |
| **AudienceAgent** | Behavioral persona modeling and intent simulation. | `AudienceSegments` |
| **CreativeAgent** | Narrative structure, hooks, and visual cue design. | `CreativeVariants` |
| **ScoringAgent** | Multi-dimensional readiness gating (GO/NO-GO). | `CompositeReadinessReport` |
| **VideoGenAgent** | Cinematic prompt synthesis and Veo rendering. | `VideoGenerationResult` |
| *...and 14 more* | See [Agents Overview](src/ads_ai/agents/README.md) | |

## 🧪 Quality & Testing

Ads.ai maintains a rigorous test suite with > 90% code coverage, focusing on:
*   **Pydantic Guardrails**: 40+ models validated for strict schema integrity.
*   **Quantified Quality**: Tests that verify "Cinematic Markers" and "Hook Strength" in agent outputs.
*   **Stand-alone Verification**: Bypass local environment issues using our standalone runner:
    ```bash
    python3 tests/standalone_runner.py
    ```

### Running Tests
```bash
# Install test dependencies
pip install -e ".[test]"

# Run the full suite with coverage
pytest tests/ -v --cov=src/ads_ai --cov-report=term-missing
```

### Linting & Type Checking
```bash
pip install -e ".[lint]"
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
