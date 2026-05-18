"""Database health endpoint tests."""

from app.core.config import get_settings, normalize_database_url


def test_database_health_returns_ok(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.health.verify_database_connection",
        lambda: None,
    )

    response = client.get("/health/db")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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


def test_database_url_normalizes_render_postgres_url():
    assert normalize_database_url("postgres://user:pass@host/db") == (
        "postgresql+psycopg://user:pass@host/db"
    )


def test_database_url_normalizes_postgresql_url_to_psycopg():
    assert normalize_database_url("postgresql://user:pass@host/db") == (
        "postgresql+psycopg://user:pass@host/db"
    )


def test_auto_create_tables_loads_from_environment(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AUTO_CREATE_TABLES", "true")

    assert get_settings().auto_create_tables is True

    get_settings.cache_clear()


def test_debug_errors_loads_from_environment(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DEBUG_ERRORS", "true")

    assert get_settings().debug_errors is True

    get_settings.cache_clear()


def test_openai_api_key_loads_from_environment(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert get_settings().openai_api_key == "test-key"

    get_settings.cache_clear()
