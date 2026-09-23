# AI-ITI Study Assistant

A **grounded RAG (Retrieval-Augmented Generation)** web application that answers questions about the AI-ITI summer-training course material (OOP, Advanced Python, Computer Vision, Deep Learning, Object Detection, YOLO, Transfer Learning, NLP, Topic Modeling, Arabic NLP, and Transformers). Answers are retrieved from twelve cleaned lab documents, cited by source section, and generated locally with Ollama — never from ungrounded model memory.

**Track:** Core Track (text-only RAG) · **Work mode:** Individual assignment · **Student:** Ali Ezz Ali

---

## Architecture

```
┌─────────────────────┐     POST /query      ┌──────────────────────────────┐
│  Streamlit frontend │ ───────────────────► │        FastAPI backend       │
│   (port 8501)       │ ◄─────────────────── │        (port 8000)           │
│  chat UI + sources  │   {answer, sources}  │                              │
└─────────────────────┘                      │  1. RetrievalService         │
                                             │     └─ Chroma (persisted)    │
                                             │        top-k=4, cosine       │
                                             │  2. GenerationService        │
                                             │     └─ Ollama qwen2.5:1.5b   │
                                             │        temp=0.1, grounded    │
                                             └──────────────┬───────────────┘
                                                            │ loads once
                                                            ▼
                                             data/vector_store/ (built by
                                             notebooks/rag_pipeline.ipynb)
```

**Answer flow:** user question → embedding query → top-4 chunks from Chroma → strict context-only prompt with `[Source N]` labels → Ollama generation → answer + cited sources in the UI.

---

## Tech stack

| Layer | Choice |
|---|---|
| Notebook / pipeline | Jupyter, pandas, section-aware chunking |
| Vector database | ChromaDB (persistent, cosine HNSW) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (EN + AR) |
| LLM | Ollama `qwen2.5:1.5b` (local, CPU-friendly) |
| Backend | FastAPI, pydantic-settings, uvicorn |
| Frontend | Streamlit + requests |
| Tests | pytest + FastAPI TestClient |
| Infra | Docker / docker-compose, GitHub Actions, devcontainer (Codespaces) |
| Tooling | uv, ruff, Makefile |

---

## Project structure

```
ai-iti-rag-assistant/
├── notebooks/
│   └── rag_pipeline.ipynb      # Phase 2 report: load → chunk → embed → retrieve → evaluate
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI app, CORS, lifespan startup
│   │   ├── api/routes/query.py # GET /health, POST /query
│   │   ├── core/config.py      # Settings from .env
│   │   ├── schemas/query.py    # QueryRequest / QueryResponse
│   │   ├── services/
│   │   │   ├── retrieval.py    # load vector store, retrieve chunks
│   │   │   └── generation.py   # grounded Ollama prompt + answer
│   │   └── utils/logging_config.py
│   ├── tests/test_query.py     # happy path + 422 validation tests
│   ├── requirements.in / requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py                  # Streamlit chat UI
│   ├── api_client.py           # API_BASE_URL from env (never hard-coded)
│   ├── requirements.in / requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── data/
│   ├── raw_notebooks/          # gitignored — original lab .ipynb files
│   ├── source_documents/       # cleaned markdown corpus (12 labs)
│   ├── vector_store/           # persisted Chroma index
│   └── evaluation/
│       ├── questions.json      # 12 test questions with expected sources
│       └── results.csv         # produced by the notebook
├── scripts/
│   ├── prepare_corpus.py       # notebooks → markdown documents
│   ├── build_vector_store.py   # markdown → chunked Chroma store
│   ├── create_notebook.py      # generates rag_pipeline.ipynb
│   ├── bootstrap_cloud.sh      # Codespaces: uv + ollama + deps
│   └── start_cloud.sh          # Codespaces: start all services
├── docker-compose.yml
├── Makefile
└── pytest.ini
```

---

## Domain & data description

The assistant is scoped to **AI-ITI Level-2 summer training course content** — a real, meaningful document collection of twelve lab notebooks the student completed during the course:

| Document | Topic |
|---|---|
| LAB-01-OOP | Classes, encapsulation, inheritance, polymorphism, dunders |
| LAB-02-Advanced-Python | Iterators, generators, decorators, context managers |
| LAB-03-Computer-Vision | OpenCV fundamentals, filtering, thresholding |
| LAB-04-Deep-Vision | CNNs, residual connections, ResNet intuition |
| LAB-05-Object-Detection | IoU, Non-Max Suppression, detection metrics |
| LAB-06-YOLO | YOLO architecture, inference, dataset workflow |
| LAB-07-Transfer-Learning | Freezing base models, fine-tuning heads |
| LAB-08-NLP-Preprocessing | Tokenization, stemming vs lemmatization |
| LAB-09-Sentiment-POS-NER | Sentiment, POS tagging, named entities |
| LAB-10-Topic-Modeling | LDA vs LSA, bag-of-words, TF-IDF |
| LAB-11-Arabic-NLP | Arabic preprocessing and summarization |
| LAB-12-Transformers | Attention, positional encoding, encoder stacks |

**Cleaning:** each lab notebook is converted to markdown (`scripts/prepare_corpus.py`) keeping explanatory markdown cells and supplements for NLP topics; outputs and embedded images are dropped. All files are text-extractable — **no OCR needed**.

**Chunking:** documents are split on Markdown section headings, then into **320-word windows with 60-word overlap**. Section boundaries keep topics intact; overlap protects explanations that cross a window edge. Every chunk stores `source`, `section`, and `chunk_id` for citations.

---

## Quick start (local)

### Prerequisites

| Tool | Minimum | Check |
|---|---|---|
| Python | 3.10 (project uses 3.12) | `python --version` |
| Ollama | latest | `ollama --version` |
| Git | recent | `git --version` |
| uv (optional) | latest | `uv --version` |

### 1. Install

```bash
git clone https://github.com/ali-ezz/ai-iti-rag-assistant.git
cd ai-iti-rag-assistant

# with uv (recommended)
make install          # creates .venv (Python 3.12) + installs requirements-dev.txt

# or with plain venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Configure

```bash
cp .env.example .env
# defaults are correct for local development
```

### 3. Build the data pipeline

```bash
# place your lab notebooks into data/raw_notebooks/ (see .gitignore), then:
make prepare          # notebooks → data/source_documents/*.md
make index            # chunks + embeddings → data/vector_store/
```

Or rebuild everything inside the notebook: open `notebooks/rag_pipeline.ipynb` and **Kernel → Restart & Run All**.

### 4. Start the LLM

```bash
ollama serve          # if not already running
ollama pull qwen2.5:1.5b
```

### 5. Run the app

```bash
make run-api          # FastAPI on http://127.0.0.1:8000
make run-ui           # Streamlit on http://127.0.0.1:8501
```

Open http://localhost:8501, ask a question, and expand **Sources** to see the cited sections.

### 6. Tests

```bash
make test             # pytest -q (happy path + validation)
```

### Docker Compose

```bash
docker compose up --build
# API :8000 · UI :8501 · Ollama :11434
```

### GitHub Codespaces

The repo ships a devcontainer; opening a Codespace runs `scripts/bootstrap_cloud.sh` automatically.

```bash
bash scripts/start_cloud.sh   # ollama + backend + frontend
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Runtime environment label |
| `VECTOR_STORE_PATH` | `data/vector_store` | Path to persisted Chroma store |
| `COLLECTION_NAME` | `ai_iti_course` | Chroma collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformer used for indexing **and** querying |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Chat model for grounded answers |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved (1–10) |
| `FRONTEND_ORIGINS` | `http://127.0.0.1:8501,http://localhost:8501` | CORS allow-list (comma-separated) |
| `API_BASE_URL` | `http://127.0.0.1:8000` | **Frontend only** — backend base URL, never hard-coded in app logic |

---

## API reference

Base URL: `http://127.0.0.1:8000` · Interactive docs: `http://127.0.0.1:8000/docs`

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "vector_store": "ready",
  "ollama": "ready",
  "model": "qwen2.5:1.5b"
}
```

`status` is `degraded` if the vector store or Ollama is unavailable.

### `POST /query`

Request:

| Field | Type | Rules |
|---|---|---|
| `question` | string | trimmed, 3–1000 characters |

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Non-Max Suppression remove duplicate detections?"}'
```

Response `200`:

```json
{
  "answer": "Non-Max Suppression ranks candidate boxes by confidence and keeps the highest-scoring box while suppressing overlaps whose IoU exceeds a threshold... [Source 1]",
  "sources": [
    "LAB-05-Object-Detection — Non-Max Suppression — chunk 2"
  ]
}
```

Errors: `422` invalid/blank/missing question · `503` vector store or LLM not ready / generation failed.

---

## Grounding & hallucination control

- System prompt forbids outside knowledge: answer **only** from supplied course context.
- Insufficient context → fixed refusal: *"I do not have enough information in the course material."*
- Temperature `0.1`, short `num_predict`, numbered `[Source N]` context blocks.
- Every response returns a de-duplicated `sources` list (source — section — chunk).
- Evaluation checks retrieval hit-rate @4 and citation presence across **12 test questions** (`data/evaluation/questions.json`).

---

## Evaluation

Run `notebooks/rag_pipeline.ipynb` (section 6) to regenerate `data/evaluation/results.csv`. Metrics reported in the notebook:

| Metric | Meaning |
|---|---|
| Retrieval hit rate @4 | Expected lab document appears in the top-4 chunks |
| Answers with citations | Generated answer contains `[Source N]` markers |

**Failure analysis & mitigations** (notebook section 7): overlapping terminology across labs, questions broader than one section, and a small CPU LLM producing terse answers — mitigated by section-aware chunks, 60-word overlap, top-4 retrieval, explicit source metadata, low-temperature generation, a strict context-only prompt, and an insufficient-context refusal. Retrieval misses are addressed by clearer headings / reranking, never by allowing unsupported model knowledge.

---

## Deliverables checklist

- [x] `notebooks/rag_pipeline.ipynb` — report-style pipeline (load, chunk, embed, retrieve, prompt, evaluate, export)
- [x] `backend/` — FastAPI with `/health` + `/query`, `.env.example`, pinned `requirements.txt`, passing pytest
- [x] `frontend/` — Streamlit chat UI with cited sources, `.env.example`, env-var API URL
- [x] Persisted vector store produced by the notebook / `make index`, loaded once at API startup
- [x] Root README (this file) sufficient for a stranger to run the project
- [x] Public GitHub repository, clean history (no `.venv`, no `.env`, no raw corpus dump)
- [x] End-to-end path: question → API → retrieval → LLM → grounded answer on screen
- [ ] Live demo + recorded video walkthrough (submitted with the form)

---

## License

MIT — see [LICENSE](LICENSE).
