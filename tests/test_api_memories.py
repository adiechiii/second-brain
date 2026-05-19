"""Memory API route tests."""

from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from app.api.routes.memories import get_memory_compression_service, get_memory_service
from app.main import create_app
from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_compression import MemoryCompressionError
from app.services.memory_service import MemoryNotFoundError


def build_memory(summary: str = "Captured memory summary.") -> Memory:
    return Memory(
        id=uuid4(),
        raw_text="  Captured memory text.  ",
        memory_type="memory",
        decision_data=None,
        clean_text="Captured memory text.",
        summary=summary,
        tags=["memory", "capture"],
        importance_score=0.2,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


def build_decision_memory() -> Memory:
    return Memory(
        id=uuid4(),
        raw_text="Decision note",
        memory_type="decision",
        decision_data={
            "context": "I need to pick a launch date.",
            "reasoning": "A smaller release is safer.",
            "expected_outcome": "Ship earlier with fewer defects.",
        },
        clean_text="Decision note",
        summary="Decision note",
        tags=["decision"],
        importance_score=0.7,
        processing_state=ProcessingState.EMBEDDED,
        record_state=RecordState.ACTIVE,
    )


class FakeMemoryService:
    def __init__(self):
        self.created_raw_text = None
        self.created_memory_type = None
        self.created_context = None
        self.created_reasoning = None
        self.created_expected_outcome = None
        self.search_query = None
        self.search_limit = None
        self.memory = build_memory()
        self.decision_memory = build_decision_memory()
        self.updated_outcome_id = None
        self.updated_actual_outcome = None
        self.updated_outcome_timestamp = None
        self.updated_outcome_evaluation = None
        self.update_count = 0
        self.evaluation_summary = {
            "accuracy": {
                "total_evaluated": 0,
                "correct_count": 0,
                "incorrect_count": 0,
                "uncertain_count": 0,
                "accuracy_rate": None,
                "incorrect_rate": None,
                "uncertain_rate": None,
                "trend": "insufficient_data",
            },
            "repeated_incorrect_patterns": [],
        }
        self.intervention_warning = {
            "warning": False,
            "risk_level": "none",
            "reason": "No repeated negative decision pattern matched",
            "reference_pattern": None,
        }
        self.evaluation_call_count = 0
        self.intervention_memory = None

    def create_memory(
        self,
        raw_text: str | None,
        memory_type: str = "memory",
        context: str | None = None,
        reasoning: str | None = None,
        expected_outcome: str | None = None,
    ) -> Memory:
        self.created_raw_text = raw_text
        self.created_memory_type = memory_type
        self.created_context = context
        self.created_reasoning = reasoning
        self.created_expected_outcome = expected_outcome
        return self.memory

    def search_memories(self, query: str, limit: int) -> list[Memory]:
        self.search_query = query
        self.search_limit = limit
        return [self.memory]

    def get_memory(self, id):
        if id == self.memory.id:
            return self.memory
        if id == self.decision_memory.id:
            return self.decision_memory
        raise MemoryNotFoundError("not found")

    def update_decision_outcome(
        self,
        id,
        actual_outcome: str,
        outcome_timestamp=None,
        outcome_evaluation: str | None = None,
    ) -> Memory:
        if id != self.decision_memory.id:
            raise MemoryNotFoundError("not found")
        self.update_count += 1
        self.updated_outcome_id = id
        self.updated_actual_outcome = actual_outcome
        self.updated_outcome_timestamp = outcome_timestamp or datetime(
            2026,
            5,
            19,
            12,
            0,
            tzinfo=timezone.utc,
        )
        self.updated_outcome_evaluation = outcome_evaluation
        self.decision_memory.actual_outcome = actual_outcome
        self.decision_memory.outcome_timestamp = self.updated_outcome_timestamp
        self.decision_memory.outcome_evaluation = outcome_evaluation
        return self.decision_memory

    def get_decision_evaluation_summary(self) -> dict:
        self.evaluation_call_count += 1
        return self.evaluation_summary

    def get_intervention_warning_for_memory(self, memory: Memory) -> dict:
        self.intervention_memory = memory
        return self.intervention_warning


class FailingMemoryService(FakeMemoryService):
    def create_memory(
        self,
        raw_text: str | None,
        memory_type: str = "memory",
        context: str | None = None,
        reasoning: str | None = None,
        expected_outcome: str | None = None,
    ) -> Memory:
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


def test_post_decision_memory_creates_structured_decision():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).post(
        "/memories",
        json={
            "memory_type": "decision",
            "context": "I need to pick a launch date.",
            "reasoning": "A smaller release is safer.",
            "expected_outcome": "Ship earlier with fewer defects.",
        },
    )

    assert response.status_code == 201
    assert service.created_raw_text is None
    assert service.created_memory_type == "decision"
    assert service.created_context == "I need to pick a launch date."
    assert service.created_reasoning == "A smaller release is safer."
    assert service.created_expected_outcome == "Ship earlier with fewer defects."


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


def test_get_decision_evaluation_returns_empty_summary():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get("/memories/decisions/evaluation")

    assert response.status_code == 200
    assert response.json() == {
        "accuracy": {
            "total_evaluated": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "uncertain_count": 0,
            "accuracy_rate": None,
            "incorrect_rate": None,
            "uncertain_rate": None,
            "trend": "insufficient_data",
        },
        "failure_loops": [],
    }
    assert service.evaluation_call_count == 1


def test_get_decision_evaluation_returns_accuracy_summary():
    app = create_app()
    service = FakeMemoryService()
    service.evaluation_summary = {
        "accuracy": {
            "total_evaluated": 3,
            "correct_count": 1,
            "incorrect_count": 1,
            "uncertain_count": 1,
            "accuracy_rate": 0.5,
            "incorrect_rate": 0.5,
            "uncertain_rate": 1 / 3,
            "trend": "insufficient_data",
        },
        "repeated_incorrect_patterns": [],
    }
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get("/memories/decisions/evaluation")

    assert response.status_code == 200
    assert response.json()["accuracy"] == {
        "total_evaluated": 3,
        "correct_count": 1,
        "incorrect_count": 1,
        "uncertain_count": 1,
        "accuracy_rate": 0.5,
        "incorrect_rate": 0.5,
        "uncertain_rate": 1 / 3,
        "trend": "insufficient_data",
    }


def test_get_decision_evaluation_returns_failure_loops():
    app = create_app()
    service = FakeMemoryService()
    service.evaluation_summary = {
        "accuracy": {
            "total_evaluated": 2,
            "correct_count": 0,
            "incorrect_count": 2,
            "uncertain_count": 0,
            "accuracy_rate": 0.0,
            "incorrect_rate": 1.0,
            "uncertain_rate": 0.0,
            "trend": "insufficient_data",
        },
        "repeated_incorrect_patterns": [
            {
                "pattern_key": "tag:launch",
                "incorrect_count": 2,
                "decision_ids": ["decision-1", "decision-2"],
                "latest_outcome_timestamp": datetime(
                    2026,
                    5,
                    19,
                    12,
                    0,
                    tzinfo=timezone.utc,
                ),
                "reason": "Repeated incorrect decisions share this grouping key",
            }
        ],
    }
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get("/memories/decisions/evaluation")

    assert response.status_code == 200
    assert response.json()["failure_loops"] == [
        {
            "pattern_key": "tag:launch",
            "incorrect_count": 2,
            "decision_ids": ["decision-1", "decision-2"],
            "latest_outcome_timestamp": "2026-05-19T12:00:00Z",
            "reason": "Repeated incorrect decisions share this grouping key",
        }
    ]


def test_get_decision_evaluation_does_not_mutate_memories():
    app = create_app()
    service = FakeMemoryService()
    original_decision_data = dict(service.decision_memory.decision_data)
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get("/memories/decisions/evaluation")

    assert response.status_code == 200
    assert service.decision_memory.decision_data == original_decision_data


def test_patch_decision_outcome_updates_existing_decision():
    app = create_app()
    service = FakeMemoryService()
    original_decision_data = dict(service.decision_memory.decision_data)
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).patch(
        f"/memories/{service.decision_memory.id}/outcome",
        json={
            "actual_outcome": "The release shipped safely.",
            "outcome_timestamp": "2026-05-19T12:00:00Z",
            "outcome_evaluation": "correct",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(service.decision_memory.id)
    assert body["actual_outcome"] == "The release shipped safely."
    assert body["outcome_timestamp"] == "2026-05-19T12:00:00Z"
    assert body["outcome_evaluation"] == "correct"
    assert body["decision_data"] == original_decision_data
    assert service.updated_outcome_id == service.decision_memory.id
    assert service.updated_actual_outcome == "The release shipped safely."
    assert service.updated_outcome_evaluation == "correct"
    assert service.update_count == 1


def test_patch_decision_outcome_uses_service_timestamp_default():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).patch(
        f"/memories/{service.decision_memory.id}/outcome",
        json={
            "actual_outcome": "The release shipped safely.",
        },
    )

    assert response.status_code == 200
    assert response.json()["outcome_timestamp"] == "2026-05-19T12:00:00Z"
    assert service.updated_outcome_timestamp == datetime(
        2026,
        5,
        19,
        12,
        0,
        tzinfo=timezone.utc,
    )


@pytest.mark.parametrize("evaluation", ["correct", "incorrect", "uncertain"])
def test_patch_decision_outcome_accepts_valid_evaluations(evaluation):
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).patch(
        f"/memories/{service.decision_memory.id}/outcome",
        json={
            "actual_outcome": "The release shipped safely.",
            "outcome_evaluation": evaluation,
        },
    )

    assert response.status_code == 200
    assert response.json()["outcome_evaluation"] == evaluation


def test_patch_decision_outcome_rejects_invalid_evaluation():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).patch(
        f"/memories/{service.decision_memory.id}/outcome",
        json={
            "actual_outcome": "The release shipped safely.",
            "outcome_evaluation": "maybe",
        },
    )

    assert response.status_code == 422
    assert service.update_count == 0


def test_get_memory_intervention_returns_no_warning():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get(
        f"/memories/{service.decision_memory.id}/intervention",
    )

    assert response.status_code == 200
    assert response.json() == {
        "warning": False,
        "risk_level": "none",
        "reason": "No repeated negative decision pattern matched",
        "reference_pattern": None,
    }
    assert service.intervention_memory is service.decision_memory


def test_get_memory_intervention_returns_warning():
    app = create_app()
    service = FakeMemoryService()
    service.intervention_warning = {
        "warning": True,
        "risk_level": "high",
        "reason": "This matches 2 past decision outcomes marked incorrect.",
        "reference_pattern": {
            "pattern_key": "tag:launch",
            "incorrect_count": 2,
            "decision_ids": ["decision-1", "decision-2"],
            "latest_outcome_timestamp": "2026-05-19T12:00:00+00:00",
        },
    }
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get(
        f"/memories/{service.decision_memory.id}/intervention",
    )

    assert response.status_code == 200
    assert response.json() == service.intervention_warning


def test_get_memory_intervention_returns_404_for_missing_memory():
    app = create_app()
    service = FakeMemoryService()
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get(f"/memories/{uuid4()}/intervention")

    assert response.status_code == 404


def test_get_memory_intervention_returns_no_warning_for_non_decision_memory():
    app = create_app()
    service = FakeMemoryService()
    service.intervention_warning = {
        "warning": True,
        "risk_level": "high",
        "reason": "This should not be returned for non-decision memories.",
        "reference_pattern": {
            "pattern_key": "tag:memory",
            "incorrect_count": 2,
            "decision_ids": ["decision-1", "decision-2"],
            "latest_outcome_timestamp": None,
        },
    }
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get(f"/memories/{service.memory.id}/intervention")

    assert response.status_code == 200
    assert response.json() == {
        "warning": False,
        "risk_level": "none",
        "reason": "No repeated negative decision pattern matched",
        "reference_pattern": None,
    }
    assert service.intervention_memory is None


def test_get_memory_intervention_does_not_mutate_memories():
    app = create_app()
    service = FakeMemoryService()
    original_decision_data = dict(service.decision_memory.decision_data)
    app.dependency_overrides[get_memory_service] = lambda: service

    from fastapi.testclient import TestClient

    response = TestClient(app).get(
        f"/memories/{service.decision_memory.id}/intervention",
    )

    assert response.status_code == 200
    assert service.decision_memory.decision_data == original_decision_data


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
