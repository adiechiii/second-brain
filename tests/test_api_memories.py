"""Memory API route tests."""

from datetime import date
from uuid import uuid4

from app.api.routes.memories import get_memory_compression_service, get_memory_service
from app.main import create_app
from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_compression import MemoryCompressionError
from app.services.memory_service import MemoryNotFoundError


def build_memory(summary: str = "Captured memory summary.") -> Memory:
    return Memory(
        id=uuid4(),
        raw_text="  Captured memory text.  ",
        clean_text="Captured memory text.",
        summary=summary,
        tags=["memory", "capture"],
        importance_score=0.2,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


class FakeMemoryService:
    def __init__(self):
        self.created_raw_text = None
        self.search_query = None
        self.search_limit = None
        self.memory = build_memory()

    def create_memory(self, raw_text: str) -> Memory:
        self.created_raw_text = raw_text
        return self.memory

    def search_memories(self, query: str, limit: int) -> list[Memory]:
        self.search_query = query
        self.search_limit = limit
        return [self.memory]

    def get_memory(self, id):
        if id != self.memory.id:
            raise MemoryNotFoundError("not found")
        return self.memory


class FailingMemoryService(FakeMemoryService):
    def create_memory(self, raw_text: str) -> Memory:
        raise RuntimeError("database exploded")

    def search_memories(self, query: str, limit: int) -> list[Memory]:
        raise RuntimeError("search exploded")


class FakeCompressionService:
    def __init__(self):
        self.daily_day = None
        self.weekly_start = None
        self.daily_memory = build_memory("Daily compression summary.")
        self.daily_memory.topic = "daily-summary"
        self.daily_memory.tags = [
            "compression",
            "daily-summary",
            "period:2026-05-19",
        ]
        self.weekly_memory = build_memory("Weekly compression summary.")
        self.weekly_memory.topic = "weekly-summary"
        self.weekly_memory.tags = [
            "compression",
            "weekly-summary",
            "period:2026-05-18..2026-05-24",
        ]

    def create_daily_summary(self, day: date) -> Memory:
        self.daily_day = day
        return self.daily_memory

    def create_weekly_summary(self, week_start: date) -> Memory:
        self.weekly_start = week_start
        return self.weekly_memory


class InvalidWeeklyCompressionService(FakeCompressionService):
    def create_weekly_summary(self, week_start: date) -> Memory:
        raise MemoryCompressionError("week_start must be a Monday")


def test_post_memories_creates_memory():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service
    client = app.test_client() if hasattr(app, "test_client") else None

    from fastapi.testclient import TestClient

    response = TestClient(app).post("/memories", json={"raw_text": "  hello  "})

    assert response.status_code == 201
    assert response.json() == {
        "id": str(service.memory.id),
        "summary": "Captured memory summary.",
    }
    assert service.created_raw_text == "  hello  "


def test_search_memories_returns_results():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get("/memories/search", params={"query": "memory", "limit": 3})

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == str(service.memory.id)
    assert service.search_query == "memory"
    assert service.search_limit == 3


def test_post_daily_compression_returns_compression_response():
    app = create_app()
    service = FakeCompressionService()
    app.dependency_overrides[get_memory_compression_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/memories/compressions/daily",
        json={"day": "2026-05-19"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": str(service.daily_memory.id),
        "summary": "Daily compression summary.",
        "topic": "daily-summary",
        "tags": [
            "compression",
            "daily-summary",
            "period:2026-05-19",
        ],
    }
    assert service.daily_day == date(2026, 5, 19)


def test_post_weekly_compression_returns_compression_response():
    app = create_app()
    service = FakeCompressionService()
    app.dependency_overrides[get_memory_compression_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/memories/compressions/weekly",
        json={"week_start": "2026-05-18"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": str(service.weekly_memory.id),
        "summary": "Weekly compression summary.",
        "topic": "weekly-summary",
        "tags": [
            "compression",
            "weekly-summary",
            "period:2026-05-18..2026-05-24",
        ],
    }
    assert service.weekly_start == date(2026, 5, 18)


def test_post_weekly_compression_returns_400_for_invalid_week_start():
    app = create_app()
    app.dependency_overrides[
        get_memory_compression_service
    ] = lambda: InvalidWeeklyCompressionService()

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/memories/compressions/weekly",
        json={"week_start": "2026-05-19"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "week_start must be a Monday"}


def test_get_memory_returns_stored_memory():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get(f"/memories/{service.memory.id}")

    assert response.status_code == 200
    assert response.json()["summary"] == "Captured memory summary."


def test_get_memory_returns_404_for_missing_memory():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get(f"/memories/{uuid4()}")

    assert response.status_code == 404


def test_post_memories_debug_errors_returns_exception_payload(monkeypatch, caplog):
    monkeypatch.setenv("DEBUG_ERRORS", "true")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_memory_service] = lambda: FailingMemoryService()

    from fastapi.testclient import TestClient

    response = TestClient(app, raise_server_exceptions=False).post(
        "/memories",
        json={"raw_text": "hello"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error_type": "RuntimeError",
        "error": "database exploded",
    }
    assert "Unhandled memory endpoint error: RuntimeError: database exploded" in caplog.text

    get_settings.cache_clear()


def test_search_memories_debug_errors_returns_exception_payload(monkeypatch):
    monkeypatch.setenv("DEBUG_ERRORS", "true")
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_memory_service] = lambda: FailingMemoryService()

    from fastapi.testclient import TestClient

    response = TestClient(app, raise_server_exceptions=False).get(
        "/memories/search",
        params={"query": "memory"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error_type": "RuntimeError",
        "error": "search exploded",
    }

    get_settings.cache_clear()


def test_post_memories_preserves_normal_500_when_debug_errors_disabled(monkeypatch):
    monkeypatch.delenv("DEBUG_ERRORS", raising=False)
    from app.core.config import get_settings

    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_memory_service] = lambda: FailingMemoryService()

    from fastapi.testclient import TestClient

    response = TestClient(app, raise_server_exceptions=False).post(
        "/memories",
        json={"raw_text": "hello"},
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"

    get_settings.cache_clear()
