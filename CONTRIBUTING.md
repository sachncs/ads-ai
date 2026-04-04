# Contributing to Ads.ai

We welcome contributions! Please follow these guidelines to ensure a smooth collaboration.

## Coding Standards

- **Style**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- **Format**: Use 4-space indentation.
- **Docstrings**: Required for all public modules, classes, and functions.
- **Types**: Use Python type hints for all function signatures.

## Development Workflow

1. **Fork & Clone**: Create a feature branch from `main`.
   ```bash
   git clone https://github.com/sachn-cs/ads-ai.git
   ```
2. **Install Dev Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Write Tests**: Ensure coverage remains > 35% and all models are validated.
4. **Linting**:
   ```bash
   ruff check src/ads_ai
   mypy src/ads_ai
   ```

## Pull Request Process

1. Ensure the test suite passes (`pytest`).
2. Update the `README.md` if existing functionality is modified.
3. Submit the PR with a clear description of changes and impact analysis.
