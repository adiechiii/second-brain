"""Memory API schemas."""

from datetime import date
from uuid import UUID

from pydantic import BaseModel, Field


class CreateMemoryRequest(BaseModel):
    raw_text: str = Field(min_length=1)


class CreateMemoryResponse(BaseModel):
    id: UUID
    summary: str | None


class MemoryResponse(BaseModel):
    id: UUID
    raw_text: str
    clean_text: str | None
    summary: str | None
    tags: list[str] | None
    topic: str | None
    importance_score: float | None
    processing_state: str
    record_state: str


class SearchMemoriesResponse(BaseModel):
    results: list[MemoryResponse]


class DailyCompressionRequest(BaseModel):
    day: date


class WeeklyCompressionRequest(BaseModel):
    week_start: date


class CompressionMemoryResponse(BaseModel):
    id: UUID
    summary: str
    topic: str | None
    tags: list[str] | None
