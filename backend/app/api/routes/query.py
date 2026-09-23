from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from backend.app.services.generation import GenerationService
from backend.app.services.retrieval import RetrievalService

router = APIRouter()


def get_retrieval_service(request: Request) -> RetrievalService:
    service = getattr(request.app.state, "retrieval", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready")
    return service


def get_generation_service(request: Request) -> GenerationService:
    service = getattr(request.app.state, "generation", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Ollama is not ready")
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    retrieval_ready = getattr(request.app.state, "retrieval", None) is not None
    generation = getattr(request.app.state, "generation", None)
    ollama_ready = bool(generation and generation.ping())
    settings = request.app.state.settings
    return HealthResponse(
        status="ok" if retrieval_ready and ollama_ready else "degraded",
        vector_store="ready" if retrieval_ready else "unavailable",
        ollama="ready" if ollama_ready else "unavailable",
        model=settings.ollama_model,
    )


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    retrieval: Annotated[RetrievalService, Depends(get_retrieval_service)],
    generation: Annotated[GenerationService, Depends(get_generation_service)],
) -> QueryResponse:
    try:
        chunks = retrieval.retrieve(payload.question)
        answer = generation.answer(payload.question, chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant could not generate an answer. Please try again.",
        ) from exc

    sources = list(dict.fromkeys(chunk.citation for chunk in chunks))
    return QueryResponse(answer=answer, sources=sources)

