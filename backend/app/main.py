import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.query import router as query_router
from backend.app.core.config import get_settings
from backend.app.services.generation import GenerationService
from backend.app.services.retrieval import RetrievalService
from backend.app.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    try:
        app.state.retrieval = RetrievalService(settings)
        logger.info("Vector store loaded from %s", settings.vector_store_path)
    except Exception as exc:  # noqa: BLE001 - startup reports any unavailable store safely
        app.state.retrieval = None
        logger.warning("Vector store unavailable: %s", exc)

    app.state.generation = GenerationService(settings)
    yield
    app.state.retrieval = None
    app.state.generation = None


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Grounded RAG API over Ali Ezz's AI-ITI course material.",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(query_router)
    return application


app = create_app()
