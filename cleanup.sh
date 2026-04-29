#!/bin/bash
# Cleanup script - removes cache files, build artifacts, and generated outputs

set -e

echo "Cleaning cache files and build artifacts..."

# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name ".coverage" -delete 2>/dev/null || true

# Coverage reports
find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "coverage.xml" -delete 2>/dev/null || true

# Build artifacts
rm -rf build/ 2>/dev/null || true
rm -rf dist/ 2>/dev/null || true
rm -rf *.egg-info/ 2>/dev/null || true
rm -rf .egg/ 2>/dev/null || true

# IDE and editor files
find . -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".vscode" -exec rm -rf {} + 2>/dev/null || true

# Generated outputs
rm -rf outputs/ 2>/dev/null || true

# ruff cache
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

echo "Cleanup complete."
