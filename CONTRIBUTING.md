# Contributing to ads-ai

We welcome contributions! Please follow these guidelines to ensure a smooth collaboration.

## Getting Started

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/your-username/ads-ai.git
   cd ads-ai
   ```

2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install Dev Dependencies**:
   ```bash
   pip install -e ".[dev,test,lint]"
   ```

## Branch Naming

Use descriptive branch names with conventional prefixes:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

Example: `feat/add-multi-language-support`

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/). Use the format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Formatting, missing semicolons, etc (no code change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Build process or auxiliary tool changes |

### Examples

```
feat(agents): add multi-language support to CreativeAgent
fix(pipeline): handle timeout in video generation stage
docs: update installation instructions
test(scoring): add edge case tests for composite scoring
chore(deps): update pydantic to v2.8
```

## Coding Standards

- **Style**: Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **Format**: Use 4-space indentation
- **Docstrings**: Required for all public modules, classes, and functions
- **Types**: Use Python type hints for all function signatures
- **Max line length**: 100 characters

## Quality Requirements

- Minimum test coverage: 80%
- All models must be validated with Pydantic
- All linting must pass with ruff
- All type checking must pass with mypy

## Testing

### Running the Test Suite

```bash
# Full suite with coverage
pytest tests/ -v --cov=ads_ai --cov-report=term-missing

# Standalone tests (no API key required)
python3 tests/standalone_runner.py

# Specific test file
pytest tests/test_agents.py -v
```

### Writing Tests

- Place tests in `tests/` with the `test_` prefix
- Use descriptive test names: `test_strategy_agent_generates_pillars`
- Test both success and error paths
- Mock external API calls when testing agents in isolation

## Linting & Type Checking

```bash
# Lint with ruff
ruff check ads_ai/ tests/

# Auto-fix lint issues
ruff check ads_ai/ tests/ --fix

# Type check with mypy
mypy ads_ai/ --ignore-missing-imports
```

## Pull Request Process

1. Ensure your branch is up to date with `master`
2. Run the full test suite: `pytest tests/ -v --cov=ads_ai`
3. Run linting: `ruff check ads_ai/ tests/`
4. Run type checking: `mypy ads_ai/ --ignore-missing-imports`
5. Update documentation if you modify existing functionality
6. Submit a PR using the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
7. Your PR will be reviewed by a maintainer

### PR Title Format

Use the same conventional commit format for PR titles:

```
feat(agents): add multi-language support
```

## Pre-commit Hooks

We recommend installing pre-commit hooks for local development:

```bash
pip install pre-commit
pre-commit install
```

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include a minimal reproduction case when reporting bugs
- Check existing issues before creating a new one
- Use the provided issue templates

## License

By contributing to ads-ai, you agree that your contributions will be licensed under the MIT License.
