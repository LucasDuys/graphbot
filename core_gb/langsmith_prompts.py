"""LangSmith prompt management with versioning and A/B testing support.

Manages prompt templates in LangSmith hub, enabling version tracking,
retrieval by name/version, and trace tagging for observability.

Five prompt templates are maintained:
  simple_qa, hard_reasoning, tool_execution, creative, analysis

Each is parameterized with {context}, {examples}, {task}.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt version tracking
# ---------------------------------------------------------------------------

CURRENT_VERSION: str = "v1"

_PROMPT_NAMES: tuple[str, ...] = (
    "simple_qa",
    "hard_reasoning",
    "tool_execution",
    "creative",
    "analysis",
)


@dataclass(frozen=True)
class PromptVersion:
    """Tracks a prompt template version and its LangSmith URL."""

    name: str
    version: str
    url: str = ""


# ---------------------------------------------------------------------------
# Prompt template definitions
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: dict[str, ChatPromptTemplate] = {
    "simple_qa": ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise, factual question-answering assistant. "
            "Answer the user's question directly and concisely using the "
            "provided context. If the context does not contain the answer, "
            "say so clearly.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "<examples>\n{examples}\n</examples>",
        ),
        ("human", "{task}"),
    ]),
    "hard_reasoning": ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert analytical reasoner. Break down complex "
            "problems step by step. Use the provided context to ground "
            "your reasoning. Think carefully before answering and show "
            "your work inside <thinking> tags, then give a final answer "
            "inside <answer> tags.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "<examples>\n{examples}\n</examples>",
        ),
        ("human", "{task}"),
    ]),
    "tool_execution": ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a tool-use specialist. You have access to various "
            "tools for file operations, web search, code execution, and "
            "browser automation. Plan which tools to use, execute them in "
            "the correct order, and synthesize the results into a clear "
            "answer. Always verify tool outputs before reporting.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "<examples>\n{examples}\n</examples>",
        ),
        ("human", "{task}"),
    ]),
    "creative": ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a creative writing and ideation assistant. Generate "
            "original, engaging content that matches the requested style "
            "and tone. Use the provided context for inspiration and "
            "grounding, but do not be constrained by it. Prioritize "
            "clarity, originality, and audience appropriateness.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "<examples>\n{examples}\n</examples>",
        ),
        ("human", "{task}"),
    ]),
    "analysis": ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a data analysis and insight extraction specialist. "
            "Examine the provided context thoroughly, identify patterns "
            "and anomalies, and present your findings in a structured "
            "format. Quantify claims where possible and distinguish "
            "between observations and inferences.\n\n"
            "<context>\n{context}\n</context>\n\n"
            "<examples>\n{examples}\n</examples>",
        ),
        ("human", "{task}"),
    ]),
}

# Descriptions for LangSmith hub metadata.
_PROMPT_DESCRIPTIONS: dict[str, str] = {
    "simple_qa": "Simple factual Q&A with context grounding",
    "hard_reasoning": "Complex multi-step reasoning with chain-of-thought",
    "tool_execution": "Tool-use planning and execution synthesis",
    "creative": "Creative writing and ideation with style matching",
    "analysis": "Data analysis and structured insight extraction",
}


# ---------------------------------------------------------------------------
# LangSmithPromptManager
# ---------------------------------------------------------------------------


class LangSmithPromptManager:
    """Manage prompt templates in LangSmith hub.

    Handles pushing templates, retrieving by name/version, and tagging
    LangSmith traces with prompt version metadata.
    """

    def __init__(self, *, prefix: str = "graphbot") -> None:
        self._prefix = prefix
        self._versions: dict[str, list[PromptVersion]] = {}

    # -- Helpers -------------------------------------------------------------

    def _prompt_identifier(self, name: str) -> str:
        """Build a LangSmith prompt identifier with prefix."""
        return f"{self._prefix}-{name}"

    def _get_client(self) -> Any:
        """Lazy-import and return a LangSmith Client instance."""
        from langsmith import Client

        return Client()

    # -- Push ----------------------------------------------------------------

    def push_templates(
        self, *, version: str = CURRENT_VERSION
    ) -> list[PromptVersion]:
        """Push all 5 prompt templates to LangSmith hub.

        Each push creates a new version if the content changed, or is a
        no-op if the content is identical to the latest version.

        Args:
            version: Version label to record (e.g. "v1", "v2").

        Returns:
            List of PromptVersion records for the pushed templates.
        """
        client = self._get_client()
        pushed: list[PromptVersion] = []

        for name, template in PROMPT_TEMPLATES.items():
            identifier = self._prompt_identifier(name)
            description = _PROMPT_DESCRIPTIONS.get(name, "")
            tags = [version, name, self._prefix]

            try:
                url = client.push_prompt(
                    identifier,
                    object=template,
                    description=description,
                    tags=tags,
                    is_public=False,
                )
                url_str = str(url) if url else ""
            except Exception:
                logger.exception("Failed to push prompt %s", identifier)
                url_str = ""

            pv = PromptVersion(name=name, version=version, url=url_str)
            pushed.append(pv)

            # Track in internal version history.
            self._versions.setdefault(name, []).append(pv)
            logger.info(
                "Pushed prompt %s %s -> %s", identifier, version, url_str
            )

        return pushed

    # -- Retrieve ------------------------------------------------------------

    def get_template(
        self,
        name: str,
        version: str | None = None,
    ) -> ChatPromptTemplate:
        """Retrieve a prompt template by name.

        If version is None, returns the local template (latest).
        If version is specified, attempts to pull from LangSmith hub.

        Args:
            name: One of the 5 registered prompt names.
            version: Optional version string. None returns the local copy.

        Returns:
            The ChatPromptTemplate.

        Raises:
            KeyError: If the name is not a registered prompt.
        """
        if name not in PROMPT_TEMPLATES:
            raise KeyError(
                f"Unknown prompt name '{name}'. "
                f"Available: {', '.join(PROMPT_TEMPLATES)}"
            )

        if version is None:
            return PROMPT_TEMPLATES[name]

        # Pull specific version from LangSmith hub.
        identifier = self._prompt_identifier(name)
        try:
            client = self._get_client()
            prompt = client.pull_prompt(identifier)
            return prompt  # type: ignore[return-value]
        except Exception:
            logger.warning(
                "Could not pull prompt %s from hub, falling back to local",
                identifier,
            )
            return PROMPT_TEMPLATES[name]

    # -- Version tracking ----------------------------------------------------

    def get_versions(self, name: str) -> list[PromptVersion]:
        """Return recorded version history for a prompt name.

        Args:
            name: The prompt name.

        Returns:
            List of PromptVersion records, oldest first.
        """
        return list(self._versions.get(name, []))

    def get_all_names(self) -> tuple[str, ...]:
        """Return the tuple of all registered prompt names."""
        return _PROMPT_NAMES

    # -- Trace tagging -------------------------------------------------------

    @staticmethod
    def tag_trace(
        run_id: str,
        prompt_name: str,
        version: str,
    ) -> dict[str, str]:
        """Build metadata dict for tagging a LangSmith trace with prompt info.

        This returns a metadata dict that should be passed to litellm or
        langsmith run creation so the prompt version is recorded in traces.

        Args:
            run_id: The LangSmith run ID (UUID string).
            prompt_name: Which prompt template was used.
            version: The prompt version string.

        Returns:
            Metadata dict with prompt_name, prompt_version, and run_id.
        """
        return {
            "run_id": run_id,
            "prompt_name": prompt_name,
            "prompt_version": version,
        }

    @staticmethod
    def make_litellm_metadata(
        prompt_name: str,
        version: str,
    ) -> dict[str, Any]:
        """Build litellm metadata kwargs for prompt version tracing.

        Pass the returned dict as ``metadata`` to litellm.acompletion()
        so LangSmith traces include prompt version info.

        Args:
            prompt_name: Which prompt template was used.
            version: The prompt version string.

        Returns:
            Dict suitable for litellm metadata parameter.
        """
        return {
            "prompt_name": prompt_name,
            "prompt_version": version,
            "tags": [f"prompt:{prompt_name}", f"version:{version}"],
        }
