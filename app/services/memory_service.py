"""Memory application service."""

from uuid import UUID

from app.models.memory import Memory
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_embedding import embed_memory
from app.services.memory_enrichment import enrich_memory
from app.services.memory_ingestion import MemoryIngestionService
from app.services.vector_search import DEFAULT_SEARCH_LIMIT, VectorSearchService


class MemoryNotFoundError(LookupError):
    """Raised when a memory cannot be found."""


class MemoryService:
    """Coordinates memory workflows across lower-level services."""

    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def create_memory(
        self,
        raw_text: str | None,
        memory_type: str = "memory",
        context: str | None = None,
        reasoning: str | None = None,
        expected_outcome: str | None = None,
    ) -> Memory:
        memory = MemoryIngestionService(self.repository).create_memory(
            raw_text=raw_text,
            memory_type=memory_type,
            context=context,
            reasoning=reasoning,
            expected_outcome=expected_outcome,
        )
        enrich_memory(memory)
        embed_memory(memory)
        return self.repository.save(memory)

    def search_memories(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Memory]:
        return VectorSearchService(self.repository).vector_search(query, limit)

    def get_memory(self, id: UUID) -> Memory:
        memory = self.repository.get_by_id(id)
        if memory is None:
            raise MemoryNotFoundError(f"Memory {id} was not found")
        return memory

    def get_memories(self, ids: list[UUID]) -> list[Memory]:
        return self.repository.get_by_ids(ids)
