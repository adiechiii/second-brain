"""Create database tables from existing SQLAlchemy metadata."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.infrastructure.database import (
    create_database_tables,
    create_pgvector_extension,
    list_database_tables,
)


def main() -> int:
    try:
        create_pgvector_extension()
        create_database_tables()
        tables = list_database_tables()
    except Exception as exc:
        print(f"Database initialization failed: {exc}")
        return 1

    print("pgvector extension ensured.")
    print("Database tables initialized.")
    print("Tables:")
    if not tables:
        print("- none")
    for table in tables:
        print(f"- {table}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
