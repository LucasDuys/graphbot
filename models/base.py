"""Abstract base class for GraphBot model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core_gb.types import CompletionResult


class ModelProvider(ABC):
    """Abstract interface for LLM completion providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'openrouter')."""

    @abstractmethod
    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        """Run a chat completion and return a structured result."""
