"""Repository operations for memory records."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.memory import Memory


class MemoryRepository:
    """Persistence boundary for memory records."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, memory: Memory) -> Memory:
        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)
        return memory

    def get_by_id(self, id: UUID) -> Memory | None:
        return self.session.get(Memory, id)

    def get_by_ids(self, ids: list[UUID]) -> list[Memory]:
        if not ids:
            return []

        statement = select(Memory).where(Memory.id.in_(ids))
        memories = list(self.session.scalars(statement))
        memory_by_id = {memory.id: memory for memory in memories}
        return [memory_by_id[id] for id in ids if id in memory_by_id]

    def save(self, memory: Memory) -> Memory:
        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)
        return memory

    def search_by_embedding(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[Memory]:
        distance = Memory.embedding.cosine_distance(query_embedding)
        statement = (
            select(Memory)
            .where(Memory.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )

        return list(self.session.scalars(statement))
