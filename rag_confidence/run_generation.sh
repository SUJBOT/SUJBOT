#!/bin/bash
# Run synthetic dataset generation
# Usage: bash rag_confidence/run_generation.sh [--dry-run]

cd "$(dirname "$0")/.."

echo "=== Generating Synthetic Eval Dataset ==="
echo "Working directory: $(pwd)"
echo ""

# Run with uv
uv run python rag_confidence/generate_synthetic_dataset.py "$@"

echo ""
echo "=== Generation Complete ==="
