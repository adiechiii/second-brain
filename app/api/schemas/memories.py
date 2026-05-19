"""Memory API schemas."""

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class CreateMemoryRequest(BaseModel):
    raw_text: str | None = Field(default=None, min_length=1)
    memory_type: str = "memory"
    context: str | None = Field(default=None, min_length=1)
    reasoning: str | None = Field(default=None, min_length=1)
    expected_outcome: str | None = Field(default=None, min_length=1)


class CreateMemoryResponse(BaseModel):
    id: UUID
    summary: str | None


class MemoryResponse(BaseModel):
    id: UUID
    raw_text: str
    memory_type: str
    decision_data: dict | None
    actual_outcome: str | None
    outcome_timestamp: datetime | None
    outcome_evaluation: str | None
    clean_text: str | None
    summary: str | None
    tags: list[str] | None
    topic: str | None
    importance_score: float | None
    processing_state: str
    record_state: str


class SearchMemoriesResponse(BaseModel):
    results: list[MemoryResponse]


class DecisionOutcomeRequest(BaseModel):
    actual_outcome: str = Field(min_length=1)
    outcome_timestamp: datetime | None = None
    outcome_evaluation: Literal["correct", "incorrect", "uncertain"] | None = None


class DailyCompressionRequest(BaseModel):
    day: date


class WeeklyCompressionRequest(BaseModel):
    week_start: date


class CompressionMemoryResponse(BaseModel):
    id: UUID
    summary: str
    topic: str | None
    tags: list[str] | None
