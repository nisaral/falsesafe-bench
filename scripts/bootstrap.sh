#!/usr/bin/env bash
# bootstrap.sh — Cross-platform bootstrap for falsesafe-bench (Linux/macOS)
# Usage: bash scripts/bootstrap.sh
set -euo pipefail

echo "=== falsesafe-bench bootstrap ==="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the updated PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv: $(uv --version)"

# Install Python 3.13 if needed
echo "Ensuring Python 3.13 is available..."
uv python install 3.13

# Create virtual environment
echo "Creating virtual environment..."
uv venv --python 3.13

# Activate and install
echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Verify installation:"
echo "  python scripts/run_experiment.py doctor"
echo "  python -m pytest -q -p no:cacheprovider"
echo ""
echo "Run a simulation:"
echo "  python scripts/run_experiment.py simulate --config configs/starters/single_robot.json"
echo ""
