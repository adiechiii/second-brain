"""Database session infrastructure."""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.database import get_engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Create a SQLAlchemy session factory bound to the configured engine."""
    return sessionmaker(
        bind=get_engine(),
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
