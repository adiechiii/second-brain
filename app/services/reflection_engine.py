"""Grounded deterministic reflection engine.

Reflections are derived only from retrieved memory fields. No external models,
long-term scheduling, or unsupported claims are used here.
"""

from collections import Counter
import json

from app.infrastructure.llm_client import generate_completion
from app.models.memory import Memory

MAX_SUMMARY_MEMORIES = 3
MAX_THEMES = 5
MAX_INSIGHTS = 5
RESPONSE_KEYS = ("summary", "themes", "insights", "questions")


def _memory_summary(memory: Memory) -> str:
    return (memory.summary or memory.clean_text or "").strip()


def _normalized_tags(memory: Memory) -> list[str]:
    tags = memory.tags or []
    return [str(tag).strip().lower() for tag in tags if str(tag).strip()]


def _normalized_topic(memory: Memory) -> str | None:
    if not memory.topic:
        return None
    topic = memory.topic.strip().lower()
    return topic or None


def _top_summaries(memories: list[Memory]) -> list[str]:
    ranked = sorted(
        memories,
        key=lambda memory: memory.importance_score or 0.0,
        reverse=True,
    )
    return [summary for memory in ranked[:MAX_SUMMARY_MEMORIES] if (summary := _memory_summary(memory))]


def _top_themes(memories: list[Memory]) -> list[str]:
    tag_counts = Counter(tag for memory in memories for tag in _normalized_tags(memory))
    return [theme for theme, _ in tag_counts.most_common(MAX_THEMES)]


def _grounding_terms(memories: list[Memory], deterministic: dict) -> set[str]:
    terms = set(deterministic.get("themes", []))
    for memory in memories:
        terms.update(_normalized_tags(memory))
        if topic := _normalized_topic(memory):
            terms.add(topic)
    return {term for term in terms if term}


def _build_insights(memories: list[Memory], themes: list[str]) -> list[str]:
    insights: list[str] = []
    tag_counts = Counter(tag for memory in memories for tag in _normalized_tags(memory))
    topic_counts = Counter(
        topic for memory in memories if (topic := _normalized_topic(memory))
    )

    for tag in themes:
        count = tag_counts[tag]
        if count > 1:
            insights.append(f"'{tag}' appears in {count} retrieved memories.")
        if len(insights) == MAX_INSIGHTS:
            return insights

    for topic, count in topic_counts.most_common(MAX_INSIGHTS - len(insights)):
        if count > 1:
            insights.append(f"The topic '{topic}' appears in {count} retrieved memories.")

    return insights


def _build_questions(themes: list[str], memory_count: int) -> list[str]:
    if memory_count < 2:
        return ["What additional memories would help confirm whether this pattern matters?"]

    return [f"What would make '{theme}' easier to act on next?" for theme in themes[:3]]


def generate_reflection(memories: list[Memory]) -> dict:
    if not memories:
        return {
            "summary": "No retrieved memories were provided, so there is not enough evidence for a reflection.",
            "themes": [],
            "insights": [],
            "questions": ["Which memories should be retrieved before reflecting?"],
        }

    summaries = _top_summaries(memories)
    themes = _top_themes(memories)
    insights = _build_insights(memories, themes)
    questions = _build_questions(themes, len(memories))

    if len(memories) < 2:
        summary = "Only one retrieved memory is available: "
        summary += summaries[0] if summaries else "no summary was available."
        insights = ["There is not enough retrieved evidence to identify a repeated pattern."]
    else:
        summary = " | ".join(summaries)
        if not summary:
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
