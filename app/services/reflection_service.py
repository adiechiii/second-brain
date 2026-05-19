"""Reflection application service."""

from uuid import UUID

from app.models.memory import Memory
from app.services.memory_service import MemoryService
from app.services.reflection_engine import generate_reflection_with_ai
from app.services.vector_search import DEFAULT_SEARCH_LIMIT


class ReflectionInputError(ValueError):
    """Raised when reflection input is not usable."""


class ReflectionService:
    """Coordinates memory retrieval and reflection generation."""

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    def generate_for_memories(
        self,
        memories: list[Memory],
        depth: str = "standard",
    ) -> dict:
        return generate_reflection_with_ai(memories, depth=depth)

    def generate_from_query(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        depth: str = "standard",
    ) -> dict:
        if not query or not query.strip():
            raise ReflectionInputError("query must not be empty")
        memories = self.memory_service.search_memories(query, limit)
        return self.generate_for_memories(memories, depth=depth)

    def generate_from_ids(
        self,
        memory_ids: list[UUID],
        depth: str = "standard",
    ) -> dict:
        if not memory_ids:
            raise ReflectionInputError("memory_ids must not be empty")
        memories = self.memory_service.get_memories(memory_ids)
        return self.generate_for_memories(memories, depth=depth)
