from ollama import Client

from backend.app.core.config import Settings
from backend.app.services.retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are the AI-ITI Study Assistant.
Answer only from the supplied course context. Never use unsupported outside knowledge.
If the context is insufficient, say: "I do not have enough information in the course material."
Be concise but educational. Cite factual statements using [Source N].
Do not invent file names, sections, metrics, or citations."""


class GenerationService:
    def __init__(self, settings: Settings) -> None:
        self.client = Client(host=settings.ollama_base_url)
        self.model = settings.ollama_model

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I do not have enough information in the course material."

        context_parts = []
        for number, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {number}] {chunk.citation}\n{chunk.text}"
            )
        prompt = (
            "COURSE CONTEXT:\n\n"
            + "\n\n".join(context_parts)
            + f"\n\nQUESTION:\n{question}\n\nANSWER:"
        )
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1, "num_ctx": 4096, "num_predict": 220},
        )
        answer = response.message.content.strip()
        if "[Source" not in answer:
            labels = ", ".join(
                f"[Source {number}]" for number in range(1, len(chunks) + 1)
            )
            answer = f"{answer}\n\nRetrieved evidence: {labels}"
        return answer

    def ping(self) -> bool:
        try:
            available = self.client.list()
            names = {model.model for model in available.models}
            return self.model in names
        except Exception:  # noqa: BLE001 - health checks must never crash the API
            return False
