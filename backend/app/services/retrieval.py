from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from backend.app.core.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    section: str
    chunk_id: str
    distance: float | None = None

    @property
    def citation(self) -> str:
        return f"{self.source} — {self.section} — chunk {self.chunk_id}"


class RetrievalService:
    def __init__(self, settings: Settings) -> None:
        store_path = Path(settings.vector_store_path)
        if not store_path.exists() or not any(store_path.iterdir()):
            raise RuntimeError(
                f"Vector store is missing at {store_path}. Run notebooks/rag_pipeline.ipynb first."
            )
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.client = chromadb.PersistentClient(path=str(store_path))
        self.collection = self.client.get_collection(
            name=settings.collection_name,
            embedding_function=embedding_function,
        )
        self.top_k = settings.retrieval_top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_texts=[question],
            n_results=min(self.top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] or {}
            chunks.append(
                RetrievedChunk(
                    text=document,
                    source=str(metadata.get("source", "Unknown source")),
                    section=str(metadata.get("section", "General")),
                    chunk_id=str(metadata.get("chunk_id", index)),
                    distance=float(distances[index]) if index < len(distances) else None,
                )
            )
        return chunks

