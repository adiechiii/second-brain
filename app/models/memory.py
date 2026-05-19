"""Memory record schema structure."""

from datetime import datetime
from enum import Enum as PythonEnum
import uuid

from sqlalchemy import DateTime, Enum, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database import Vector
from app.models.base import Base


class ProcessingState(str, PythonEnum):
    CAPTURED = "captured"
    ENRICHED = "enriched"
    EMBEDDED = "embedded"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RecordState(str, PythonEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Memory(Base):
    """Stored memory record.

    raw_text is the preserved source text. clean_text and other classification
    fields are derived later by separate workflows.
    """

    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    processing_state: Mapped[ProcessingState] = mapped_column(
        Enum(
            ProcessingState,
            name="processing_state",
            native_enum=False,
            length=32,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
        default=ProcessingState.CAPTURED,
    )
    record_state: Mapped[RecordState] = mapped_column(
        Enum(
            RecordState,
            name="record_state",
            native_enum=False,
            length=32,
            values_callable=lambda enum: [item.value for item in enum],
        ),
        nullable=False,
        default=RecordState.ACTIVE,
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="memory",
        server_default="memory",
    )
    decision_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    clean_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    topic: Mapped[str | None] = mapped_column(String(255), nullable=True)
    importance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    classification_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    tone_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    topic_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)
