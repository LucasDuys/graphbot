"""LangSmith observability setup via LiteLLM callback."""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def setup_langsmith() -> bool:
    """Enable LangSmith tracing via litellm callback.

    Reads LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2 from env.
    If LANGCHAIN_API_KEY is missing, silently disables tracing (returns False).
    Sets LANGCHAIN_PROJECT to 'graphbot' if not set.
    Sets LANGCHAIN_TRACING_V2 to 'true' if not set.

    Returns True if tracing was enabled, False otherwise.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        logger.debug("LANGCHAIN_API_KEY not set -- LangSmith tracing disabled")
        return False

    try:
        import langsmith  # noqa: F401
    except Exception:
        logger.debug("langsmith package not installed -- tracing disabled")
        return False

    os.environ.setdefault("LANGCHAIN_PROJECT", "graphbot")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "graphbot")
    # EU endpoint for LangSmith -- litellm uses LANGCHAIN_ENDPOINT
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://eu.api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_BASE_URL", "https://eu.api.smith.langchain.com")

    import litellm

    if "langsmith" not in litellm.success_callback:
        litellm.success_callback.append("langsmith")

    logger.info("LangSmith tracing enabled for project '%s'", os.environ["LANGCHAIN_PROJECT"])
    return True
