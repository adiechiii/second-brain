"""Environment-backed application configuration."""

from dataclasses import dataclass
from functools import lru_cache
import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_database_url(value: str | None) -> str | None:
    if value is None:
        return None

    url = value.strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


@dataclass(frozen=True)
class Settings:
    app_name: str = "Second Brain API"
    environment: str = "development"
    debug: bool = False
    debug_errors: bool = False
    database_url: str | None = None
    auto_create_tables: bool = False
    embedding_provider: str | None = None
    llm_provider: str | None = None
    openai_api_key: str | None = None
    enable_openai_reflections: bool = False


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        app_name=os.getenv("APP_NAME", Settings.app_name),
        environment=os.getenv("APP_ENV", Settings.environment),
        debug=_get_bool("APP_DEBUG", Settings.debug),
        debug_errors=_get_bool("DEBUG_ERRORS", Settings.debug_errors),
        database_url=normalize_database_url(os.getenv("DATABASE_URL")),
        auto_create_tables=_get_bool("AUTO_CREATE_TABLES", Settings.auto_create_tables),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER"),
        llm_provider=os.getenv("LLM_PROVIDER"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        enable_openai_reflections=_get_bool(
            "ENABLE_OPENAI_REFLECTIONS",
            Settings.enable_openai_reflections,
        ),
    )
