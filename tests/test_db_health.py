"""Database health endpoint tests."""

import pytest

from app.core.config import get_settings
from app.api.routes.health import get_health_db_session


class FakeSession:
    def __init__(self):
        self.closed = False

    def execute(self, statement):
        assert str(statement) == "SELECT 1"

    def close(self):
        self.closed = True


def test_database_health_returns_ok(client):
    fake_session = FakeSession()

    def override_db_session():
        try:
            yield fake_session
        finally:
            fake_session.close()

    client.app.dependency_overrides[get_health_db_session] = override_db_session

    response = client.get("/health/db")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake_session.closed is True


def test_database_url_loads_from_environment(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/second_brain",
    )

    assert get_settings().database_url == (
        "postgresql+psycopg://postgres:postgres@localhost:5432/second_brain"
    )

    get_settings.cache_clear()


def test_openai_api_key_loads_from_environment(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert get_settings().openai_api_key == "test-key"

    get_settings.cache_clear()
