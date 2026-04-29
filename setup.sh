#!/bin/bash
# Development setup script

set -e

echo "Setting up development environment..."

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        echo "Warning: .env.example not found. Create .env manually."
    fi
fi

# Install in editable mode with all dependencies
echo "Installing package in editable mode..."
pip install -e ".[dev,test,lint]"

echo ""
echo "Setup complete. To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "To lint and type-check:"
echo "  ruff check ads_ai tests/"
echo "  mypy ads_ai --ignore-missing-imports"
