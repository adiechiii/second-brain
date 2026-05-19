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
from app.services.behavior_loops import detect_behavior_loops
from app.services.belief_consolidation import convert_patterns_to_beliefs
from app.services.decision_evaluation import (
    decision_grouping_keys,
    detect_decision_mismatch_patterns,
    get_decision_accuracy_patterns,
)
from app.services.decision_patterns import detect_decision_patterns
from app.services.tension_detection import detect_tensions

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


def _safe_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


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

LOW_SIGNAL_THEME_TERMS = {
    "after",
    "before",
    "brain",
    "captured",
    "daily",
    "debug",
    "error",
    "fake",
    "final",
    "key",
    "local",
    "planned",
    "production",
    "reviewed",
    "second",
    "test",
    "traceback",
    "weekly",
}

LOW_SIGNAL_MEMORY_MARKERS = (
    "debug error test",
    "fake openai key test",
    "local traceback test",
    "test after deploy",
)

LOW_SIGNAL_MEMORY_TOKEN_GROUPS = (
    {"debug", "test"},
    {"fake", "openai", "key"},
    {"local", "traceback"},
    {"smoke", "test"},
)


def _is_useful_theme(theme: str) -> bool:
    normalized = theme.strip().lower()
    if not normalized:
        return False
    return normalized not in LOW_SIGNAL_THEME_TERMS


def _has_low_signal_memory_marker(memory: Memory) -> bool:
    normalized = _normalized_summary_text(_memory_summary(memory))
    if any(marker in normalized for marker in LOW_SIGNAL_MEMORY_MARKERS):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return any(group <= tokens for group in LOW_SIGNAL_MEMORY_TOKEN_GROUPS)


def _is_high_signal_memory(memory: Memory) -> bool:
    summary = _memory_summary(memory)
    normalized = _normalized_summary_text(summary)
    if len(summary) < 40:
        return False
    if normalized in LOW_SIGNAL_MEMORY_TEXTS:
        return False
    return not _has_low_signal_memory_marker(memory)


def _normalized_tags(memory: Memory) -> list[str]:
    tags = memory.tags or []
    return [
        tag.strip().lower()
        for tag in tags
        if isinstance(tag, str) and _is_useful_theme(tag)
    ]


def _normalized_topic(memory: Memory) -> str | None:
    if not isinstance(memory.topic, str):
        return None
    topic = memory.topic.strip().lower()
    if not _is_useful_theme(topic):
        return None
    return topic


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
            insights.append(
                f"A recurring signal around '{theme}' appears across {count} retrieved memories."
            )
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


def build_grounded_questions(
    patterns: list[dict],
    beliefs: list[dict],
    loops: list[dict],
    tensions: list[dict],
    limit: int = 5,
) -> list[str]:
    if limit < 1:
        return []

    questions: list[str] = []
    seen: set[str] = set()

    def add(question: str) -> None:
        if len(questions) >= limit:
            return
        if question in seen:
            return
        seen.add(question)
        questions.append(question)

    for tension in tensions:
        if not isinstance(tension, dict):
            continue
        label = _safe_text(tension.get("label"))
        if label is not None:
            add(f"Where is this tension showing up in your current decisions: {label}")

    for loop in loops:
        if not isinstance(loop, dict):
            continue
        label = _safe_text(loop.get("label"))
        if label is not None:
            add(f"What would help you interrupt this loop: {label}")

    for belief in beliefs:
        if not isinstance(belief, dict):
            continue
        statement = _safe_text(belief.get("statement"))
        if statement is not None:
            add(f"What evidence would strengthen or challenge this belief: {statement}")

    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        label = _safe_text(pattern.get("label"))
        if label is not None:
            add(f"What small decision could test this pattern: {label}")

    return questions


def _cognitive_leverage_payload(memories: list[Memory]) -> dict:
    dominant_patterns = detect_decision_patterns(memories)
    belief_statements = convert_patterns_to_beliefs(dominant_patterns)
    detected_loops = detect_behavior_loops(memories)
    detected_tensions = detect_tensions(dominant_patterns, detected_loops)
    grounded_questions = build_grounded_questions(
        patterns=dominant_patterns,
        beliefs=belief_statements,
        loops=detected_loops,
        tensions=detected_tensions,
    )

    return {
        "dominant_patterns": dominant_patterns,
        "belief_statements": belief_statements,
        "detected_loops": detected_loops,
        "detected_tensions": detected_tensions,
        "grounded_questions": grounded_questions,
    }


def _risk_signals(accuracy: dict, failure_loops: list[dict]) -> list[dict]:
    signals: list[dict] = []
    for failure_loop in failure_loops:
        signals.append(
            {
                "type": "repeated_incorrect_decisions",
                "severity": "high",
                "reason": "Repeated incorrect outcomes detected for the same decision pattern",
                "pattern_key": failure_loop["pattern_key"],
                "incorrect_count": failure_loop["incorrect_count"],
            }
        )

    if accuracy.get("trend") == "declining":
        signals.append(
            {
                "type": "declining_decision_accuracy",
                "severity": "medium",
                "reason": "Recent evaluated decisions are less accurate than earlier decisions",
            }
        )

    return signals


def _reinforced_patterns(memories: list[Memory], min_count: int = 2) -> list[dict]:
    grouped: dict[str, dict] = {}
    for memory in memories:
        if (
            getattr(memory, "memory_type", None) != "decision"
            or getattr(memory, "outcome_evaluation", None) != "correct"
        ):
            continue

        for pattern_key in decision_grouping_keys(memory):
            if pattern_key not in grouped:
                grouped[pattern_key] = {
                    "pattern_key": pattern_key,
                    "decision_ids": [],
                }
            memory_id = str(getattr(memory, "id", ""))
            if memory_id not in grouped[pattern_key]["decision_ids"]:
                grouped[pattern_key]["decision_ids"].append(memory_id)

    results = []
    for pattern in grouped.values():
        correct_count = len(pattern["decision_ids"])
        if correct_count < min_count:
            continue
        results.append(
            {
                "pattern_key": pattern["pattern_key"],
                "correct_count": correct_count,
                "decision_ids": pattern["decision_ids"],
                "reason": "Repeated correct decisions share this grouping key",
            }
        )

    return sorted(
        results,
        key=lambda item: (
            -item["correct_count"],
            item["pattern_key"],
        ),
    )


def _broken_patterns(failure_loops: list[dict]) -> list[dict]:
    return [
        {
            "pattern_key": failure_loop["pattern_key"],
            "incorrect_count": failure_loop["incorrect_count"],
            "decision_ids": failure_loop["decision_ids"],
            "reason": failure_loop["reason"],
        }
        for failure_loop in failure_loops
    ]


def _decision_feedback_payload(memories: list[Memory]) -> dict:
    accuracy = get_decision_accuracy_patterns(memories)
    failure_loops = detect_decision_mismatch_patterns(memories)
    return {
        "accuracy": accuracy,
        "failure_loops": failure_loops,
        "risk_signals": _risk_signals(accuracy, failure_loops),
        "behavior_reinforcement": {
            "reinforced_patterns": _reinforced_patterns(memories),
            "broken_patterns": _broken_patterns(failure_loops),
        },
    }


def _deterministic_extension_payload(memories: list[Memory]) -> dict:
    payload = _cognitive_leverage_payload(memories)
    payload["decision_feedback"] = _decision_feedback_payload(memories)
    return payload


def generate_reflection(memories: list[Memory]) -> dict:
    if not memories:
        reflection = {
            "summary": "No retrieved memories were provided, so there is not enough evidence for a reflection.",
            "themes": [],
            "insights": [],
            "questions": ["Which memories should be retrieved before reflecting?"],
        }
        reflection.update(_deterministic_extension_payload(memories))
        return reflection

    theme_counts = _theme_counts(memories)
    has_repeated_useful_theme = any(count > 1 for count in theme_counts.values())
    all_memories_are_debug_notes = all(
        _has_low_signal_memory_marker(memory) for memory in memories
    )
    if all_memories_are_debug_notes and not has_repeated_useful_theme:
        reflection = {
            "summary": "Retrieved memories are mostly low-signal notes, so there is not enough meaningful evidence for a useful reflection.",
            "themes": [],
            "insights": [],
            "questions": ["Which more substantive memories should be retrieved before reflecting?"],
        }
        reflection.update(_deterministic_extension_payload(memories))
        return reflection

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

    reflection = {
        "summary": summary,
        "themes": themes,
        "insights": insights,
        "questions": questions,
    }
    reflection.update(_deterministic_extension_payload(memories))
    return reflection


def _reflection_depth_instruction(depth: str) -> str:
    if depth == "quick":
        return (
            "Depth mode: quick. Keep the reflection very short. "
            "Prioritize one core signal, up to two themes, one insight, and one question."
        )
    if depth == "deep":
        return (
            "Depth mode: deep. Look for repeated themes, possible tensions, and useful next questions. "
            "Stay grounded and cautious; do not add unsupported conclusions."
        )
    return (
        "Depth mode: standard. Provide themes, grounded insights, and useful questions without overexplaining."
    )


def _build_ai_prompt(
    memories: list[Memory],
    deterministic: dict,
    depth: str = "standard",
) -> str:
    prompt_memories = memories[:MAX_AI_PROMPT_MEMORIES]
    depth_instruction = _reflection_depth_instruction(depth)

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
        "Keep the response concise, grounded, and specific.\n"
        + depth_instruction
        + "\n\nRetrieved memory context with signal labels:\n"
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

    reflection = {
        "summary": deterministic["summary"],
        "themes": themes or deterministic["themes"],
        "insights": insights or deterministic["insights"],
        "questions": questions or deterministic["questions"],
    }
    reflection.update(
        {
            "dominant_patterns": deterministic.get("dominant_patterns", []),
            "belief_statements": deterministic.get("belief_statements", []),
            "detected_loops": deterministic.get("detected_loops", []),
            "detected_tensions": deterministic.get("detected_tensions", []),
            "grounded_questions": deterministic.get("grounded_questions", []),
            "decision_feedback": deterministic.get(
                "decision_feedback",
                _decision_feedback_payload([]),
            ),
        }
    )
    return reflection


def generate_reflection_with_ai(
    memories: list[Memory],
    depth: str = "standard",
) -> dict:
    deterministic = generate_reflection(memories)
    if not get_settings().enable_openai_reflections:
        return deterministic

    prompt = _build_ai_prompt(memories, deterministic, depth=depth)

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
