"""Groq direct model provider using litellm."""

from __future__ import annotations

import os
import time

from litellm import acompletion
from litellm.exceptions import (
    AuthenticationError as LitellmAuthError,
    RateLimitError as LitellmRateLimitError,
)

from core_gb.types import CompletionResult
from models.base import ModelProvider
from models.errors import AuthError, ProviderError, RateLimitError


class GroqProvider(ModelProvider):
    """LLM provider that routes requests directly to Groq via litellm.

    Uses the ``groq/`` prefix for litellm model routing. Free tier allows
    roughly 30 RPM.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise AuthError(
                "Missing API key: set GROQ_API_KEY or pass api_key",
                provider="groq",
                model="",
            )

    @property
    def name(self) -> str:
        """Provider name."""
        return "groq"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        """Run a chat completion via Groq."""
        prefixed_model = f"groq/{model}"
        start = time.perf_counter()
        try:
            response = await acompletion(
                model=prefixed_model,
                messages=messages,
                api_key=self._api_key,
                **kwargs,
            )
        except LitellmRateLimitError as exc:
            raise RateLimitError(
                str(exc), provider="groq", model=model
            ) from exc
        except LitellmAuthError as exc:
            raise AuthError(
                str(exc), provider="groq", model=model
            ) from exc
        except Exception as exc:
            raise ProviderError(
                str(exc), provider="groq", model=model
            ) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage
        cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0

        return CompletionResult(
            content=choice.message.content or "",
            model=response.model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            latency_ms=elapsed_ms,
            cost=float(cost),
        )
