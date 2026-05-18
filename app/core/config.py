"""Environment-backed application configuration."""

from dataclasses import dataclass
from functools import lru_cache
import os


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "Second Brain API"
    environment: str = "development"
    debug: bool = False
    database_url: str | None = None
    embedding_provider: str | None = None
    llm_provider: str | None = None
    openai_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        app_name=os.getenv("APP_NAME", Settings.app_name),
        environment=os.getenv("APP_ENV", Settings.environment),
        debug=_get_bool("APP_DEBUG", Settings.debug),
        database_url=os.getenv("DATABASE_URL"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER"),
        llm_provider=os.getenv("LLM_PROVIDER"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
