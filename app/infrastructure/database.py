"""Database engine infrastructure.

The application expects PostgreSQL with the pgvector extension available. This
module does not create extensions, schemas, tables, or ORM models.
"""

from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import Engine, create_engine

from app.core.config import get_settings

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


__all__ = [
    "DatabaseConfigurationError",
    "PGVECTOR_EXTENSION_NAME",
    "Vector",
    "get_engine",
]
