"""Reflection API schemas."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ReflectionRequest(BaseModel):
    query: str | None = None
    memory_ids: list[UUID] | None = None
    limit: int = Field(default=10, ge=1, le=50)
    depth: Literal["quick", "standard", "deep"] = "standard"

    @model_validator(mode="after")
    def validate_input(self):
        has_query = bool(self.query and self.query.strip())
        has_ids = bool(self.memory_ids)
        if has_query == has_ids:
            raise ValueError("Provide exactly one of query or memory_ids")
        return self


class ReflectionResponse(BaseModel):
    summary: str
    themes: list[str]
    insights: list[str]
    questions: list[str]
    dominant_patterns: list[dict] = Field(default_factory=list)
    belief_statements: list[dict] = Field(default_factory=list)
    detected_loops: list[dict] = Field(default_factory=list)
    detected_tensions: list[dict] = Field(default_factory=list)
    grounded_questions: list[str] = Field(default_factory=list)
    decision_feedback: dict = Field(default_factory=dict)
