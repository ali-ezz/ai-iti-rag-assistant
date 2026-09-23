.PHONY: install prepare index test run-api run-ui cloud-start

install:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -r requirements-dev.txt

prepare:
	.venv/bin/python scripts/prepare_corpus.py

index:
	.venv/bin/python scripts/build_vector_store.py --reset

test:
	.venv/bin/python -m pytest -q

run-api:
	.venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

run-ui:
	API_BASE_URL=http://127.0.0.1:8000 .venv/bin/streamlit run frontend/app.py --server.port 8501

cloud-start:
	bash scripts/start_cloud.sh
