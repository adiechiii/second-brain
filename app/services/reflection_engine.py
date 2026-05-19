"""Grounded deterministic reflection engine.

Reflections are derived only from retrieved memory fields. No external models,
long-term scheduling, or unsupported claims are used here.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import re

from app.core.config import get_settings
from app.infrastructure.llm_client import generate_completion
from app.models.memory import Memory

MAX_SUMMARY_MEMORIES = 3
MAX_THEMES = 5
MAX_INSIGHTS = 5
MAX_AI_PROMPT_MEMORIES = 10
MAX_AI_MEMORY_SUMMARY_CHARS = 300
RESPONSE_KEYS = ("summary", "themes", "insights", "questions")
UNSAFE_AI_OUTPUT_TERMS = (
    "personality",
    "trait",
    "character",
    "feels",
    "felt",
    "anxious",
    "sad",
    "angry",
    "stressed",
    "burnout",
    "caused",
    "because",
    "therefore",
    "proves",
    "should",
    "must",
    "need to",
    "have to",
)


def _memory_summary(memory: Memory) -> str:
    return (memory.summary or memory.clean_text or "").strip()


def _normalized_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _truncate_ai_text(
    text: str,
    max_chars: int = MAX_AI_MEMORY_SUMMARY_CHARS,
) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3].rstrip() + "..."


LOW_SIGNAL_MEMORY_TEXTS = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "noted",
    "done",
    "yes",
    "no",
}


def _is_high_signal_memory(memory: Memory) -> bool:
    summary = _memory_summary(memory)
    if len(summary) < 40:
        return False
    return _normalized_summary_text(summary) not in LOW_SIGNAL_MEMORY_TEXTS


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
        return ["What follow-up memory would show whether this is isolated or recurring?"]

    if not themes:
        return ["What label, topic, or follow-up note would clarify the shared signal here?"]

    questions = []
    for theme in themes[:3]:
        count = _evidence_count(theme, memories)
        if count > 1:
            questions.append(
                f"What decision, action, or open question keeps recurring around '{theme}'?"
            )
        else:
            questions.append(
                f"What extra memory would confirm whether '{theme}' is important or just a one-off lead?"
            )
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
    prompt_memories = memories[:MAX_AI_PROMPT_MEMORIES]

    def format_memory_lines(selected_memories: list[Memory]) -> str:
        memory_lines = []
        for index, memory in enumerate(selected_memories, start=1):
            summary = _truncate_ai_text(_memory_summary(memory)) or "No summary available."
            tags = ", ".join(_normalized_tags(memory)) or "none"
            topic = _normalized_topic(memory) or "none"
            signal = "strong" if _is_high_signal_memory(memory) else "weak"
            memory_lines.append(
                f"{index}. summary: {summary}\n"
                f"   signal: {signal}\n"
                f"   tags: {tags}\n"
                f"   topic: {topic}"
            )
        return "\n".join(memory_lines) if memory_lines else "None."

    return (
        "You are the reflection layer of Second Brain.\n"
        "Your job is not to summarize everything. Your job is to help the user think better using only retrieved memories.\n"
        "Use ONLY the provided memories and deterministic baseline.\n"
        "Do NOT invent facts, causes, motives, emotions, personality traits, diagnoses, or events.\n"
        "Do NOT make broad claims like 'you are the kind of person who'.\n"
        "Prefer patterns supported by repeated signals.\n"
        "A repeated signal means the same theme appears across multiple memories, or one memory is unusually clear and specific.\n"
        "If evidence is weak, say so cautiously. Do not force an insight.\n"
        "Focus on what may be useful for the user's next thinking step.\n"
        "Avoid generic coaching language, motivational filler, therapy-like interpretation, and personality analysis.\n"
        "Good style: 'A recurring signal is...', 'There may be a tension between...', 'The strongest evidence points to...', 'This seems worth watching, but the evidence is light...'.\n"
        "Bad style: 'You are clearly...', 'You always...', 'This proves that...', 'You need to...', 'Your personality is...'.\n"
        "Return only valid JSON with exactly these keys: summary, themes, insights, questions.\n"
        "Each value must match this schema: summary string; themes list of strings; insights list of strings; questions list of strings.\n"
        "Keep the response concise, grounded, and specific.\n\n"
        "Retrieved memory context with signal labels:\n"
        + format_memory_lines(prompt_memories)
        + "\n\nTreat weak memories as lower-confidence evidence. Use them only if they support a pattern already visible in stronger or repeated context.\n"
        "Generate sharper questions grounded in the actual memory themes.\n\n"
        "Deterministic baseline JSON:\n"
        + json.dumps(deterministic, sort_keys=True)
    )


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _has_grounding(value: str, terms: set[str]) -> bool:
    if not terms:
        return True
    lowered = value.lower()
    return any(term in lowered for term in terms)


def _is_safe_ai_output(value: str) -> bool:
    lowered = value.lower()
    return not any(term in lowered for term in UNSAFE_AI_OUTPUT_TERMS)


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
        if _has_grounding(insight, grounding_terms) and _is_safe_ai_output(insight)
    ]
    questions = [
        question
        for question in candidate["questions"]
        if _has_grounding(question, grounding_terms) and _is_safe_ai_output(question)
    ]

    return {
        "summary": deterministic["summary"],
        "themes": themes or deterministic["themes"],
        "insights": insights or deterministic["insights"],
        "questions": questions or deterministic["questions"],
    }


def generate_reflection_with_ai(memories: list[Memory]) -> dict:
    deterministic = generate_reflection(memories)
    if not get_settings().enable_openai_reflections:
        return deterministic

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
