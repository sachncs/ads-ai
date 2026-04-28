# Contributing to ads.ai

We welcome contributions! Please follow these guidelines to ensure a smooth collaboration.

## Getting Started

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/sachn-cs/ads-ai.git
   cd ads-ai
   ```

2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install Dev Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Coding Standards

- **Style**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **Format**: Use 4-space indentation
- **Docstrings**: Required for all public modules, classes, and functions
- **Types**: Use Python type hints for all function signatures

## Quality Requirements

- Minimum test coverage: 35%
- All models must be validated
- All linting must pass with ruff
- All type checking must pass with mypy

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src/ads_ai --cov-report=html
```

## Linting & Type Checking

```bash
# Lint with ruff
ruff check src/ads_ai

# Type check with mypy
mypy src/ads_ai
```

## Pre-commit Hooks

We recommend installing pre-commit hooks:
```bash
pre-commit install
```

## Pull Request Process

1. Ensure the test suite passes (`pytest`)
2. Ensure linting and type checking pass
3. Update the `README.md` and relevant documentation if you modify existing functionality
4. Submit a PR with a clear description of changes and impact analysis
5. Your PR will be reviewed by a maintainer

## Commit Messages

Use clear, descriptive commit messages:
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters
- Add a blank line followed by more detail if needed

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include a minimal reproduction case when reporting bugs
- Check existing issues before creating a new one

## License

By contributing to ads.ai, you agree that your contributions will be licensed under the MIT License.
