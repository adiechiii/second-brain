"""OpenAI embedding infrastructure."""

from openai import OpenAI

from app.core.config import get_settings

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSIONS = 1536


class EmbeddingConfigurationError(RuntimeError):
    """Raised when embedding infrastructure is used without configuration."""


def get_embedding(text: str) -> list[float]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise EmbeddingConfigurationError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
    )

    return list(response.data[0].embedding)
