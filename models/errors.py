"""Custom exceptions for GraphBot model providers."""

from __future__ import annotations


class ProviderError(Exception):
    """Base error for model provider failures."""

    def __init__(self, message: str, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(message)


class RateLimitError(ProviderError):
    """Raised when the provider returns a rate-limit response."""


class AuthError(ProviderError):
    """Raised when authentication with the provider fails."""
