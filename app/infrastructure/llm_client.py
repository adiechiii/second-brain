"""OpenAI LLM client infrastructure."""

from openai import OpenAI

from app.core.config import get_settings

OPENAI_REFLECTION_MODEL = "gpt-4.1-mini"
LLM_TIMEOUT_SECONDS = 15.0


class LLMConfigurationError(RuntimeError):
    """Raised when LLM infrastructure is used without configuration."""


class LLMGenerationError(RuntimeError):
    """Raised when LLM generation fails or returns no text."""


def generate_completion(prompt: str) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is not configured")

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.with_options(timeout=LLM_TIMEOUT_SECONDS).responses.create(
            model=OPENAI_REFLECTION_MODEL,
            input=prompt,
        )
    except TimeoutError as exc:
        raise LLMGenerationError("LLM request timed out") from exc
    except Exception as exc:
        raise LLMGenerationError("LLM request failed") from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise LLMGenerationError("LLM response did not include output text")

    return output_text
