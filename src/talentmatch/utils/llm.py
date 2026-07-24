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
    from litellm import aembedding

    return await aembedding(model=model, input=input)
