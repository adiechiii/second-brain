"""Deterministic memory enrichment service.

This module intentionally avoids external APIs, embeddings, retrieval, or LLM
usage. It prepares the enrichment pipeline with simple local heuristics.
"""

import re

from app.models.memory import Memory, ProcessingState

SUMMARY_WORD_LIMIT = 20
MAX_TAGS = 8
IMPORTANCE_FULL_SCORE_WORDS = 200

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _build_summary(clean_text: str) -> str:
    return " ".join(clean_text.split()[:SUMMARY_WORD_LIMIT])


def _extract_tags(clean_text: str) -> list[str]:
    tags: list[str] = []
    for word in _words(clean_text):
        if len(word) < 3 or word in _STOP_WORDS or word in tags:
            continue
        tags.append(word)
        if len(tags) == MAX_TAGS:
            break
    return tags


def _score_importance(clean_text: str) -> float:
    word_count = len(clean_text.split())
    score = word_count / IMPORTANCE_FULL_SCORE_WORDS
    return round(max(0.0, min(score, 1.0)), 3)


def enrich_memory(memory: Memory) -> Memory:
    clean_text = memory.clean_text or ""

    memory.summary = _build_summary(clean_text)
    memory.tags = _extract_tags(clean_text)
    memory.importance_score = _score_importance(clean_text)

    if memory.processing_state == ProcessingState.CAPTURED:
        memory.processing_state = ProcessingState.ENRICHED

    return memory
