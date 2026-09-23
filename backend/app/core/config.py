from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "AI-ITI Study Assistant API"
    app_version: str = "1.0.0"
    environment: str = "development"
    vector_store_path: Path = PROJECT_ROOT / "data" / "vector_store"
    collection_name: str = "ai_iti_course"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:1.5b"
    retrieval_top_k: int = Field(default=4, ge=1, le=10)
    frontend_origins: str = "http://127.0.0.1:8501,http://localhost:8501"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

