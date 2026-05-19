"""Application service for manually triggered memory compression."""

from datetime import date, datetime, time, timedelta, timezone

from app.models.memory import Memory
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_compression import (
    MemoryCompressionError,
    build_daily_summary,
    build_weekly_summary,
)

DAILY_SUMMARY_TYPE = "daily-summary"
WEEKLY_SUMMARY_TYPE = "weekly-summary"


class MemoryCompressionService:
    """Coordinates manual compression reads, idempotency, and persistence."""

    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def create_daily_summary(self, day: date) -> Memory:
        period_label = day.isoformat()
        existing = self.repository.find_compression_summary(
            DAILY_SUMMARY_TYPE,
            f"period:{period_label}",
        )
        if existing is not None:
            return existing

        start_at = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end_at = start_at + timedelta(days=1)
        sources = self.repository.list_active_by_created_at_range(start_at, end_at)
        summary = build_daily_summary(sources, day)
        return self.repository.save(summary)

    def create_weekly_summary(self, week_start: date) -> Memory:
        if week_start.weekday() != 0:
            raise MemoryCompressionError("week_start must be a Monday")

        period_label = (
            f"{week_start.isoformat()}.."
            f"{(week_start + timedelta(days=6)).isoformat()}"
        )
        existing = self.repository.find_compression_summary(
            WEEKLY_SUMMARY_TYPE,
            f"period:{period_label}",
        )
        if existing is not None:
            return existing

        start_at = datetime.combine(week_start, time.min, tzinfo=timezone.utc)
        end_at = start_at + timedelta(days=7)
        sources = self.repository.list_active_by_created_at_range(start_at, end_at)
        summary = build_weekly_summary(sources, week_start)
        return self.repository.save(summary)
