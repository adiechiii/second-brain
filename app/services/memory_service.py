"""Memory application service."""

from datetime import datetime, timezone
from uuid import UUID

from app.models.memory import Memory
from app.repositories.memory_repository import MemoryRepository
from app.services.decision_evaluation import (
    detect_decision_mismatch_patterns,
    get_decision_accuracy_patterns,
)
from app.services.intervention import assess_intervention_risk
from app.services.memory_embedding import embed_memory
from app.services.memory_enrichment import enrich_memory
from app.services.memory_ingestion import MemoryIngestionService
from app.services.vector_search import DEFAULT_SEARCH_LIMIT, VectorSearchService


class MemoryNotFoundError(LookupError):
    """Raised when a memory cannot be found."""


class MemoryOutcomeError(ValueError):
    """Raised when a decision outcome update is invalid."""


VALID_OUTCOME_EVALUATIONS = {"correct", "incorrect", "uncertain"}


def _outcome_timestamp(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


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

    def update_decision_outcome(
        self,
        id: UUID,
        actual_outcome: str,
        outcome_timestamp: datetime | None = None,
        outcome_evaluation: str | None = None,
    ) -> Memory:
        memory = self.get_memory(id)
        if memory.memory_type != "decision":
            raise MemoryOutcomeError("memory must be a decision")

        if not actual_outcome or not actual_outcome.strip():
            raise MemoryOutcomeError("actual_outcome must not be empty")

        if (
            outcome_evaluation is not None
            and outcome_evaluation not in VALID_OUTCOME_EVALUATIONS
        ):
            raise MemoryOutcomeError("outcome_evaluation is not supported")

        timestamp = _outcome_timestamp(outcome_timestamp)
        return self.repository.update_decision_outcome(
            memory,
            actual_outcome,
            timestamp,
            outcome_evaluation,
        )

    def get_decision_evaluation_summary(self) -> dict:
        memories = self.repository.list_decisions_with_outcomes()
        return {
            "accuracy": get_decision_accuracy_patterns(memories),
            "repeated_incorrect_patterns": detect_decision_mismatch_patterns(memories),
        }

    def get_intervention_warning_for_memory(self, memory: Memory) -> dict:
        past_memories = self.repository.list_decisions_with_outcomes()
        return assess_intervention_risk(memory, past_memories)
