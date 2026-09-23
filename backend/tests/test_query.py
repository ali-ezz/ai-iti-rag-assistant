import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes.query import get_generation_service, get_retrieval_service
from backend.app.main import app
from backend.app.services.retrieval import RetrievedChunk


class FakeRetrieval:
    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                text="NMS keeps the highest-confidence box and removes overlaps.",
                source="LAB-5",
                section="Non-Max Suppression",
                chunk_id="2",
                distance=0.1,
            )
        ]


class FakeGeneration:
    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return "NMS removes overlapping detections. [Source 1]"


@pytest.fixture(autouse=True)
def override_services():
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrieval()
    app.dependency_overrides[get_generation_service] = lambda: FakeGeneration()
    yield
    app.dependency_overrides.clear()


def test_query_happy_path() -> None:
    response = TestClient(app).post("/query", json={"question": "What is NMS?"})

    assert response.status_code == 200
    assert response.json()["answer"].endswith("[Source 1]")
    assert response.json()["sources"] == ["LAB-5 — Non-Max Suppression — chunk 2"]


def test_query_rejects_blank_question() -> None:
    response = TestClient(app).post("/query", json={"question": "  "})
    assert response.status_code == 422


def test_query_rejects_missing_question() -> None:
    response = TestClient(app).post("/query", json={})
    assert response.status_code == 422
