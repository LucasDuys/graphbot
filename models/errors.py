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


class AllProvidersExhaustedError(Exception):
    """Raised when every provider in the rotation has failed."""

    def __init__(self, errors: list[ProviderError]) -> None:
        self.errors = errors
        names = [e.provider for e in errors]
        super().__init__(f"All providers exhausted: {', '.join(names)}")
