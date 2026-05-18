"""Memory model structure tests."""

from app.models.memory import Memory, ProcessingState, RecordState


def test_memory_model_instantiates_with_raw_text():
    memory = Memory(raw_text="Original note text")

    assert memory.raw_text == "Original note text"
    assert memory.clean_text is None
    assert memory.summary is None
    assert memory.tags is None
    assert memory.topic is None
    assert memory.importance_score is None
    assert memory.classification_confidence is None
    assert memory.tone_confidence is None
    assert memory.topic_confidence is None
    assert memory.embedding is None


def test_memory_model_required_fields():
    columns = Memory.__table__.columns

    assert columns.raw_text.nullable is False
    assert columns.processing_state.nullable is False
    assert columns.record_state.nullable is False
    assert columns.created_at.nullable is False
    assert columns.updated_at.nullable is False


def test_memory_model_defaults_are_defined():
    memory = Memory(raw_text="Original note text")

    assert memory.processing_state is None
    assert memory.record_state is None
    assert Memory.__table__.columns.processing_state.default.arg == ProcessingState.CAPTURED
    assert Memory.__table__.columns.record_state.default.arg == RecordState.ACTIVE


def test_memory_model_uses_enum_columns():
    columns = Memory.__table__.columns

    assert columns.processing_state.type.enums == [
        ProcessingState.CAPTURED.value,
        ProcessingState.ENRICHED.value,
        ProcessingState.EMBEDDED.value,
        ProcessingState.PENDING.value,
        ProcessingState.PROCESSING.value,
        ProcessingState.COMPLETED.value,
        ProcessingState.FAILED.value,
    ]
    assert columns.record_state.type.enums == [
        RecordState.ACTIVE.value,
        RecordState.ARCHIVED.value,
        RecordState.DELETED.value,
    ]


def test_memory_model_has_embedding_vector_column():
    columns = Memory.__table__.columns

    assert columns.embedding.nullable is True
    assert columns.embedding.type.dim == 1536
