"""Grounded deterministic reflection engine.

Reflections are derived only from retrieved memory fields. No external models,
long-term scheduling, or unsupported claims are used here.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import re

from app.infrastructure.llm_client import generate_completion
from app.models.memory import Memory

MAX_SUMMARY_MEMORIES = 3
MAX_THEMES = 5
MAX_INSIGHTS = 5
RESPONSE_KEYS = ("summary", "themes", "insights", "questions")


def _memory_summary(memory: Memory) -> str:
    return (memory.summary or memory.clean_text or "").strip()


def _normalized_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalized_tags(memory: Memory) -> list[str]:
    tags = memory.tags or []
    return [
        tag.strip().lower()
        for tag in tags
        if isinstance(tag, str) and tag.strip()
    ]


def _normalized_topic(memory: Memory) -> str | None:
    if not isinstance(memory.topic, str):
        return None
    topic = memory.topic.strip().lower()
    return topic or None


def _safe_importance(memory: Memory) -> float:
    try:
        return float(memory.importance_score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _recency_sort_value(memory: Memory) -> float:
    timestamp = memory.updated_at or memory.created_at
    if not isinstance(timestamp, datetime):
        return float("-inf")

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return timestamp.timestamp()


def _summary_theme_coverage(memory: Memory, themes: list[str]) -> int:
    theme_set = set(themes)
    return len(_memory_themes(memory) & theme_set)


def _top_summaries(memories: list[Memory], themes: list[str]) -> list[str]:
    ranked = sorted(
        memories,
        key=lambda memory: (
            -_summary_theme_coverage(memory, themes),
            -_safe_importance(memory),
            -_recency_sort_value(memory),
            _normalized_summary_text(_memory_summary(memory)),
        ),
    )
    summaries: list[str] = []
    seen: set[str] = set()
    for memory in ranked:
        summary = _memory_summary(memory)
        normalized = _normalized_summary_text(summary)
        if not summary or normalized in seen:
            continue
        summaries.append(summary)
        seen.add(normalized)
        if len(summaries) == MAX_SUMMARY_MEMORIES:
            break
    return summaries


def _memory_themes(memory: Memory) -> set[str]:
    themes = set(_normalized_tags(memory))
    if topic := _normalized_topic(memory):
        themes.add(topic)
    return themes


def _theme_counts(memories: list[Memory]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for memory in memories:
        counts.update(_memory_themes(memory))
    return counts


def _top_themes(memories: list[Memory]) -> list[str]:
    counts = _theme_counts(memories)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [theme for theme, _ in ranked[:MAX_THEMES]]


def _evidence_count(theme: str, memories: list[Memory]) -> int:
    return sum(1 for memory in memories if theme in _memory_themes(memory))


def _grounding_terms(memories: list[Memory], deterministic: dict) -> set[str]:
    terms = set(deterministic.get("themes", []))
    for memory in memories:
        terms.update(_normalized_tags(memory))
        if topic := _normalized_topic(memory):
            terms.add(topic)
    return {term for term in terms if term}


def _build_insights(memories: list[Memory], themes: list[str]) -> list[str]:
    if not themes:
        return [
            "Retrieved memories do not share tags or topics, so there is not enough evidence for a named pattern."
        ]

    insights: list[str] = []
    for theme in themes:
        count = _evidence_count(theme, memories)
        if count > 1:
            insights.append(f"'{theme}' is supported by {count} retrieved memories.")
        elif count == 1:
            insights.append(
                f"'{theme}' appears in one retrieved memory, so treat it as a lead rather than a pattern."
            )
        if len(insights) == MAX_INSIGHTS:
            break

    return insights


def _build_questions(themes: list[str], memories: list[Memory]) -> list[str]:
    memory_count = len(memories)
    if memory_count < 2:
        return ["What additional memories would help confirm whether this pattern matters?"]

    if not themes:
        return ["What additional note would clarify whether these retrieved memories share a theme?"]

    questions = []
    for theme in themes[:3]:
        count = _evidence_count(theme, memories)
        if count > 1:
            questions.append(
                f"Which retrieved memory best supports the next step for '{theme}'?"
            )
        else:
            questions.append(f"What additional note would clarify '{theme}'?")
    return questions


def generate_reflection(memories: list[Memory]) -> dict:
    if not memories:
        return {
            "summary": "No retrieved memories were provided, so there is not enough evidence for a reflection.",
            "themes": [],
            "insights": [],
            "questions": ["Which memories should be retrieved before reflecting?"],
        }

    themes = _top_themes(memories)
    summaries = _top_summaries(memories, themes)
    insights = _build_insights(memories, themes)
    questions = _build_questions(themes, memories)

    if len(memories) < 2:
        summary = "Only one retrieved memory is available: "
        summary += summaries[0] if summaries else "no summary was available."
        insights = ["There is not enough retrieved evidence to identify a repeated pattern."]
    else:
        if summaries:
            summary = (
                f"Across {len(memories)} retrieved memories, the strongest evidence points to: "
                + " | ".join(summaries)
            )
        else:
            summary = "Retrieved memories did not include summaries to combine."

    return {
        "summary": summary,
        "themes": themes,
        "insights": insights,
        "questions": questions,
    }


def _build_ai_prompt(memories: list[Memory], deterministic: dict) -> str:
    memory_lines = []
    for index, memory in enumerate(memories, start=1):
        summary = _memory_summary(memory) or "No summary available."
        tags = ", ".join(_normalized_tags(memory)) or "none"
        topic = _normalized_topic(memory) or "none"
        memory_lines.append(
            f"{index}. summary: {summary}\n"
            f"   tags: {tags}\n"
            f"   topic: {topic}"
        )

    return (
        "You are improving a grounded reflection over retrieved memories.\n"
        "Use ONLY the provided memories and deterministic baseline.\n"
        "Do NOT invent facts, causes, emotions, personality traits, diagnoses, or events.\n"
        "If evidence is weak, remain cautious.\n"
        "No therapy or coaching tone. No personality analysis. No unsupported emotional claims.\n"
        "Return only valid JSON with exactly these keys: summary, themes, insights, questions.\n"
        "Each value must match this schema: summary string; themes list of strings; "
        "insights list of strings; questions list of strings.\n\n"
        "Retrieved memories:\n"
        + "\n".join(memory_lines)
        + "\n\nDeterministic baseline JSON:\n"
        + json.dumps(deterministic, sort_keys=True)
    )


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _has_grounding(value: str, terms: set[str]) -> bool:
    if not terms:
        return True
    lowered = value.lower()
    return any(term in lowered for term in terms)


def _validate_ai_reflection(
    candidate: object,
    deterministic: dict,
    grounding_terms: set[str],
) -> dict:
    if not isinstance(candidate, dict):
        raise ValueError("AI reflection must be a JSON object")
    if set(candidate) != set(RESPONSE_KEYS):
        raise ValueError("AI reflection schema mismatch")
    if not isinstance(candidate["summary"], str):
        raise ValueError("AI reflection summary must be a string")
    if not _is_string_list(candidate["themes"]):
        raise ValueError("AI reflection themes must be a list of strings")
    if not _is_string_list(candidate["insights"]):
        raise ValueError("AI reflection insights must be a list of strings")
    if not _is_string_list(candidate["questions"]):
        raise ValueError("AI reflection questions must be a list of strings")

    deterministic_themes = set(deterministic.get("themes", []))
    themes = [theme for theme in candidate["themes"] if theme in deterministic_themes]
    insights = [
        insight
        for insight in candidate["insights"]
        if _has_grounding(insight, grounding_terms)
    ]
    questions = [
        question
        for question in candidate["questions"]
        if _has_grounding(question, grounding_terms)
    ]

    return {
        "summary": deterministic["summary"],
        "themes": themes or deterministic["themes"],
        "insights": insights or deterministic["insights"],
        "questions": questions or deterministic["questions"],
    }


def generate_reflection_with_ai(memories: list[Memory]) -> dict:
    deterministic = generate_reflection(memories)
    prompt = _build_ai_prompt(memories, deterministic)

    try:
        response_text = generate_completion(prompt)
        candidate = json.loads(response_text)
        return _validate_ai_reflection(
            candidate,
            deterministic,
            _grounding_terms(memories, deterministic),
        )
    except Exception:
        return deterministic
