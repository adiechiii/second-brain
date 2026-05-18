"""Reflection API route tests."""

from uuid import uuid4

from app.api.routes.reflections import get_reflection_service
from app.main import create_app


class FakeReflectionService:
    def __init__(self):
        self.query = None
        self.limit = None
        self.memory_ids = None
        self.reflection = {
            "summary": "Grounded reflection.",
            "themes": ["database"],
            "insights": ["'database' appears in 2 retrieved memories."],
            "questions": ["What database decision needs review?"],
        }

    def generate_from_query(self, query: str, limit: int) -> dict:
        self.query = query
        self.limit = limit
        return self.reflection

    def generate_from_ids(self, memory_ids: list) -> dict:
        self.memory_ids = memory_ids
        return self.reflection


def test_post_reflections_with_query_returns_reflection():
    app = create_app()
    service = FakeReflectionService()
    app.dependency_overrides[get_reflection_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/reflections",
        json={"query": "database", "limit": 2},
    )

    assert response.status_code == 200
    assert response.json() == service.reflection
    assert service.query == "database"
    assert service.limit == 2


def test_post_reflections_with_memory_ids_returns_reflection():
    app = create_app()
    service = FakeReflectionService()
    app.dependency_overrides[get_reflection_service] = lambda: service
    memory_id = uuid4()

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/reflections",
        json={"memory_ids": [str(memory_id)]},
    )

    assert response.status_code == 200
    assert response.json() == service.reflection
    assert service.memory_ids == [memory_id]


def test_post_reflections_requires_query_or_memory_ids():
    app = create_app()
    service = FakeReflectionService()
    app.dependency_overrides[get_reflection_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).post("/reflections", json={})

    assert response.status_code == 422
