from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("Question must contain at least 3 non-space characters")
        return cleaned


class QueryResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "answer": "Non-Max Suppression removes overlapping detections...",
            "sources": ["LAB-5 — Non-Max Suppression — chunk 2"],
        }
    })

    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    ollama: str
    model: str

