from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "rag_pipeline.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3.12 (AI-ITI RAG)",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
notebook["cells"] = [
    markdown(
        """
# AI-ITI Study Assistant — RAG Pipeline

**Student:** Ali Ezz Ali  
**Track:** Core Track — text RAG  
**Runtime:** GitHub Codespaces (cloud), Python 3.12, Chroma, Ollama

This notebook builds and evaluates a Retrieval-Augmented Generation pipeline over twelve
cleaned AI-ITI lab documents. It is designed to run from top to bottom in the repository's
cloud environment.
"""
    ),
    markdown(
        """
## 1. Load and inspect

The corpus contains one Markdown document per solved lab. Notebook outputs and embedded
images were removed during conversion; explanatory markdown and relevant code were kept.
No OCR is required because all selected files are text-extractable notebooks.
"""
    ),
    code(
        """
import json
import os
import sys
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

DOCUMENTS = ROOT / "data" / "source_documents"
VECTOR_STORE = ROOT / "data" / "vector_store"
QUESTIONS_FILE = ROOT / "data" / "evaluation" / "questions.json"
documents = sorted(DOCUMENTS.glob("*.md"))
inspection = pd.DataFrame({
    "document": [path.name for path in documents],
    "format": [path.suffix for path in documents],
    "characters": [len(path.read_text(encoding="utf-8")) for path in documents],
    "needs_ocr": [False] * len(documents),
})
print(f"Documents: {len(documents)}")
print(f"Total characters: {inspection['characters'].sum():,}")
inspection
"""
    ),
    markdown(
        """
## 2. Chunking strategy

Documents are split by Markdown section, then into **320-word chunks with a 60-word
overlap**. Section boundaries preserve topic meaning and the overlap protects explanations
that cross a window boundary. A 320-word window is large enough for code-plus-explanation
context but small enough to keep retrieval focused. Every chunk stores its source document,
section heading, and chunk number for citations.
"""
    ),
    code(
        """
from scripts.build_vector_store import load_chunks

chunks = load_chunks(DOCUMENTS, size=320, overlap=60)
chunk_stats = pd.Series([len(chunk.text.split()) for chunk in chunks]).describe()
print(f"Chunks: {len(chunks)}")
chunk_stats
"""
    ),
    markdown(
        """
## 3. Embeddings and persistent vector store

The multilingual `paraphrase-multilingual-MiniLM-L12-v2` sentence-transformer creates
embeddings for English and Arabic questions. Chroma stores the vectors on disk so the API
loads them at startup instead of rebuilding them per request.
"""
    ),
    code(
        """
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "ai_iti_course"
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=str(VECTOR_STORE))
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)
collection.add(
    ids=[f"{chunk.source}:{chunk.chunk_id}" for chunk in chunks],
    documents=[chunk.text for chunk in chunks],
    metadatas=[{
        "source": chunk.source,
        "section": chunk.section,
        "chunk_id": chunk.chunk_id,
    } for chunk in chunks],
)
print(f"Persisted {collection.count()} chunks to {VECTOR_STORE}")
"""
    ),
    markdown("## 4. Retrieval and citation grounding"),
    code(
        """
def retrieve(question: str, top_k: int = 4) -> list[dict]:
    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": text,
            "source": metadata["source"],
            "section": metadata["section"],
            "chunk_id": metadata["chunk_id"],
            "distance": round(float(distance), 4),
        }
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]

questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
retrieval_rows = []
for item in questions:
    results = retrieve(item["question"])
    sources = [result["source"] for result in results]
    retrieval_rows.append({
        "question": item["question"],
        "expected_source": item["expected_source"],
        "top_source": sources[0],
        "retrieved_sources": ", ".join(sources),
        "relevant": item["expected_source"] in sources,
    })
retrieval_df = pd.DataFrame(retrieval_rows)
retrieval_df
"""
    ),
    code(
        """
retrieval_accuracy = retrieval_df["relevant"].mean()
print(f"Retrieval hit rate @4: {retrieval_accuracy:.1%}")
"""
    ),
    markdown(
        """
## 5. Prompt template

The generation prompt explicitly limits Ollama to retrieved evidence, numbers every source,
requires inline citations, and provides a refusal sentence for insufficient context. This is
the primary hallucination control.
"""
    ),
    code(
        """
from backend.app.services.generation import SYSTEM_PROMPT
print(SYSTEM_PROMPT)
"""
    ),
    markdown(
        """
## 6. End-to-end evaluation with Ollama

The following evaluation runs all test questions through retrieval and `qwen2.5:1.5b`.
`retrieval_correct` checks whether the expected lab is in the top four results.
`citation_grounded` checks whether the answer uses the required source markers. Final manual
review should inspect whether each statement is actually supported by its retrieved text.
"""
    ),
    code(
        """
from backend.app.core.config import Settings
from backend.app.services.generation import GenerationService
from backend.app.services.retrieval import RetrievedChunk

generator = GenerationService(Settings())
evaluation_rows = []
for item in questions:
    retrieved = retrieve(item["question"])
    service_chunks = [RetrievedChunk(
        text=result["text"],
        source=result["source"],
        section=result["section"],
        chunk_id=result["chunk_id"],
        distance=result["distance"],
    ) for result in retrieved]
    answer = generator.answer(item["question"], service_chunks)
    sources = [result["source"] for result in retrieved]
    evaluation_rows.append({
        "question": item["question"],
        "retrieved_source": sources[0],
        "answer": answer,
        "retrieval_correct": item["expected_source"] in sources,
        "citation_grounded": "[Source" in answer,
    })
evaluation_df = pd.DataFrame(evaluation_rows)
evaluation_df
"""
    ),
    code(
        """
summary = pd.DataFrame({
    "metric": ["Questions", "Retrieval hit rate @4", "Answers with citations"],
    "value": [
        len(evaluation_df),
        f"{evaluation_df['retrieval_correct'].mean():.1%}",
        f"{evaluation_df['citation_grounded'].mean():.1%}",
    ],
})
evaluation_path = ROOT / "data" / "evaluation" / "results.csv"
evaluation_df.to_csv(evaluation_path, index=False)
summary
"""
    ),
    markdown(
        """
## 7. Failure analysis and mitigation

Likely failures include overlapping terminology across labs, questions that are broader than
one section, and a small CPU-friendly LLM producing terse answers. Mitigations are section-aware
chunks, overlap, top-4 retrieval, explicit source metadata, low-temperature generation, a strict
context-only prompt, and an insufficient-context refusal. Retrieval misses should be handled by
improving document headings or adding a reranker—not by allowing unsupported model knowledge.

## 8. Export

The vector store is persisted in `data/vector_store/`; configuration is recorded in `.env.example`.
The FastAPI lifespan loads this existing store once at startup. No indexing occurs during a query.
"""
    ),
    code(
        """
assert (VECTOR_STORE / "chroma.sqlite3").exists()
assert collection.count() == len(chunks)
print("Export verified:", VECTOR_STORE)
print("The backend can load this store without rebuilding it.")
"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Created {OUTPUT}")

