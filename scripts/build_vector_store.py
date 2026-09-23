import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    section: str
    chunk_id: str


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("General", text)]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def window_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, len(words), step)]


def load_chunks(directory: Path, size: int = 320, overlap: int = 60) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        number = 0
        for section, body in split_sections(text):
            for window in window_text(body, size=size, overlap=overlap):
                number += 1
                chunks.append(
                    Chunk(
                        text=window,
                        source=path.stem,
                        section=section,
                        chunk_id=str(number),
                    )
                )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the persistent Chroma vector store")
    parser.add_argument("--documents", type=Path, default=Path("data/source_documents"))
    parser.add_argument("--output", type=Path, default=Path("data/vector_store"))
    parser.add_argument("--collection", default="ai_iti_course")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(args.documents)
    if not chunks:
        raise SystemExit(f"No chunks produced from {args.documents}")

    embedding_function = SentenceTransformerEmbeddingFunction(model_name=args.embedding_model)
    client = chromadb.PersistentClient(path=str(args.output))
    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[f"{chunk.source}:{chunk.chunk_id}" for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "source": chunk.source,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks from {len(list(args.documents.glob('*.md')))} documents")
    print(f"Vector store: {args.output.resolve()}")


if __name__ == "__main__":
    main()
