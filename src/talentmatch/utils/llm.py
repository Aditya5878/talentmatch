import logging
from typing import Any

from litellm import acompletion
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from talentmatch.config import settings

logger = logging.getLogger("talentmatch.llm")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def llm_completion(messages: list[dict[str, str]], **kwargs: Any) -> Any:
    """Send a chat completion request to the configured LLM with automatic retry.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) on any exception.
    Logs a warning before each retry.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        **kwargs: Additional parameters passed to litellm.acompletion
                  (e.g. temperature, max_tokens, response_format).

    Returns:
        The litellm completion response object.

    Raises:
        The last exception after all retry attempts are exhausted.
    """
    return await acompletion(
        model=settings.llm_model,
        messages=messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def llm_embedding(model: str, input: list[str]) -> Any:
    """Send an embedding request to the configured provider with automatic retry.

    Retries up to 3 times with exponential backoff on any exception.

    Args:
        model: The embedding model identifier (e.g. "text-embedding-3-small").
        input: List of text strings to embed.

    Returns:
        The litellm embedding response object.

    Raises:
        The last exception after all retry attempts are exhausted.
    """
    from litellm import aembedding

    return await aembedding(model=model, input=input)
