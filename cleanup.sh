#!/bin/bash
# Cleanup script for ads-ai project
# Removes cache directories and local testing outputs

set -e

echo "Cleaning up cache directories..."

# Remove common cache directories
rm -rf .benchmarks
rm -rf .mypy_cache
rm -rf .pytest_cache
rm -rf .ruff_cache

# Remove __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove .pyc files
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove .coverage if present
rm -rf .coverage 2>/dev/null || true

echo "Cleanup complete."