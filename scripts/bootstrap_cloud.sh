#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip compile backend/requirements.in -o backend/requirements.txt --python-version 3.12
uv pip compile frontend/requirements.in -o frontend/requirements.txt --python-version 3.12
uv pip compile requirements-dev.in -o requirements-dev.txt --python-version 3.12
uv pip install --python .venv/bin/python -r requirements-dev.txt

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Cloud environment ready. Run: bash scripts/start_cloud.sh"
