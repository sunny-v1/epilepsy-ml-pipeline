#!/usr/bin/env bash
# Create .venv with Python 3.15 (if missing), install dependencies, and run the smoke test.
set -euo pipefail

# PyYAML has no Python 3.15 wheel yet; build its pure-Python version without Cython.
export PYYAML_FORCE_CYTHON=0 PYYAML_FORCE_LIBYAML=0
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment with Python 3.15..."
    python3.15 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py
