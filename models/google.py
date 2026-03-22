"""Google AI Studio (Gemini) model provider using litellm."""

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


class GoogleProvider(ModelProvider):
    """LLM provider that routes requests through Google AI Studio via litellm.

    Uses the ``gemini/`` prefix for litellm model routing. Free tier allows
    roughly 5 RPM on Gemini 2.5 Pro.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise AuthError(
                "Missing API key: set GOOGLE_API_KEY or pass api_key",
                provider="google",
                model="",
            )

    @property
    def name(self) -> str:
        """Provider name."""
        return "google"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        """Run a chat completion via Google AI Studio."""
        prefixed_model = f"gemini/{model}"
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
                str(exc), provider="google", model=model
            ) from exc
        except LitellmAuthError as exc:
            raise AuthError(
                str(exc), provider="google", model=model
            ) from exc
        except Exception as exc:
            raise ProviderError(
                str(exc), provider="google", model=model
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
