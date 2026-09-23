#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p .logs

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >.logs/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

ollama pull "${OLLAMA_MODEL:-qwen2.5:1.5b}"

if [ ! -f data/vector_store/chroma.sqlite3 ]; then
  python scripts/build_vector_store.py --reset
fi

nohup uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 >.logs/backend.log 2>&1 &
nohup env API_BASE_URL=http://127.0.0.1:8000 streamlit run frontend/app.py \
  --server.address 0.0.0.0 --server.port 8501 >.logs/frontend.log 2>&1 &

echo "Cloud services started: API on 8000, UI on 8501"

