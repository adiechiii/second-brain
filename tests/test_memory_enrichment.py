"""Memory enrichment service tests."""

from app.models.memory import Memory, ProcessingState, RecordState
from app.services.memory_enrichment import SUMMARY_WORD_LIMIT, enrich_memory


def build_memory(clean_text: str) -> Memory:
    return Memory(
        raw_text=clean_text,
        clean_text=clean_text,
        processing_state=ProcessingState.CAPTURED,
        record_state=RecordState.ACTIVE,
    )


def test_enrichment_fills_fields():
    memory = build_memory(
        "Project planning notes include database schema decisions and API review."
    )

    enriched = enrich_memory(memory)

    assert enriched is memory
    assert memory.summary
    assert memory.tags
    assert memory.importance_score is not None
    assert memory.processing_state == ProcessingState.ENRICHED


def test_summary_uses_first_20_words():
    clean_text = (
        "one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
        "twentyone twentytwo"
    )
    memory = build_memory(clean_text)

    enrich_memory(memory)

    assert len(memory.summary.split()) == SUMMARY_WORD_LIMIT
    assert memory.summary.split() == clean_text.split()[:SUMMARY_WORD_LIMIT]


def test_tags_not_empty_for_valid_input():
    memory = build_memory("Architecture review captured database service repository boundaries.")

    enrich_memory(memory)

    assert memory.tags
    assert "architecture" in memory.tags


def test_importance_score_is_in_range():
    memory = build_memory("important memory " * 300)

    enrich_memory(memory)

    assert 0.0 <= memory.importance_score <= 1.0
