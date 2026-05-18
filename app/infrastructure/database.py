"""Database engine infrastructure.

The application expects PostgreSQL with the pgvector extension available. This
module does not create extensions, schemas, tables, or ORM models.
"""

from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, inspect, text, create_engine

from app.core.config import get_settings
from app.models.base import Base

PGVECTOR_EXTENSION_NAME = "vector"


class DatabaseConfigurationError(RuntimeError):
    """Raised when database infrastructure is used without configuration."""


@lru_cache
def get_engine() -> Engine:
    """Create the SQLAlchemy engine from DATABASE_URL."""
    settings = get_settings()
    if not settings.database_url:
        raise DatabaseConfigurationError("DATABASE_URL is not configured")

    return create_engine(settings.database_url, pool_pre_ping=True)


def verify_database_connection() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def list_database_tables() -> list[str]:
    return inspect(get_engine()).get_table_names()


def create_pgvector_extension() -> None:
    with get_engine().begin() as connection:
        connection.execute(
            text(f'CREATE EXTENSION IF NOT EXISTS "{PGVECTOR_EXTENSION_NAME}"')
        )


def create_database_tables() -> None:
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


__all__ = [
    "DatabaseConfigurationError",
    "PGVECTOR_EXTENSION_NAME",
    "Vector",
    "create_database_tables",
    "create_pgvector_extension",
    "get_engine",
    "list_database_tables",
    "verify_database_connection",
]
