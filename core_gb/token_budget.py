"""Token budget enforcer for single-call context assembly.

Allocates a fixed token budget across context sections by priority,
trimming lowest-priority sections first to fit within the budget.
Uses a simple word_count / 0.75 heuristic for token estimation
to avoid external tokenizer dependencies.
"""

from __future__ import annotations

import math


# Priority order: lower number = higher priority (kept first).
SECTION_PRIORITIES: dict[str, int] = {
    "conversation": 1,
    "reflections": 2,
    "entities": 3,
    "patterns": 4,
}

# Unknown sections get a priority worse than any known section.
_DEFAULT_PRIORITY: int = 99


class TokenBudget:
    """Enforces a token budget for single-call context assembly.

    Reserves space for system prompt, user message, and response tokens,
    then allocates the remaining budget to context sections by priority.

    Args:
        max_tokens: Total token limit for the model call. Defaults to 4096
            (suitable for 8B parameter models).
        system_prompt_reserve: Tokens reserved for the system prompt.
        user_message_reserve: Tokens reserved for the user message.
        response_reserve: Tokens reserved for the model response.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        system_prompt_reserve: int = 100,
        user_message_reserve: int = 200,
        response_reserve: int = 500,
    ) -> None:
        self.max_tokens: int = max_tokens
        self.system_prompt_reserve: int = system_prompt_reserve
        self.user_message_reserve: int = user_message_reserve
        self.response_reserve: int = response_reserve

    @property
    def available_budget(self) -> int:
        """Tokens available for context sections after reserves."""
        remaining = (
            self.max_tokens
            - self.system_prompt_reserve
            - self.user_message_reserve
            - self.response_reserve
        )
        return max(0, remaining)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using word_count / 0.75 heuristic.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count, rounded up to the nearest integer.
        """
        if not text or not text.strip():
            return 0
        word_count = len(text.split())
        return math.ceil(word_count / 0.75)

    def _section_priority(self, section_name: str) -> int:
        """Return the priority for a section name (lower = higher priority)."""
        return SECTION_PRIORITIES.get(section_name, _DEFAULT_PRIORITY)

    def trim_to_budget(self, sections: dict[str, str]) -> dict[str, str]:
        """Trim context sections to fit within the available token budget.

        Sections are allocated by priority (conversation > reflections >
        entities > patterns). When the budget is exceeded, the lowest-priority
        sections are dropped entirely (replaced with empty strings) until the
        total fits.

        Args:
            sections: Mapping of section name to section text content.

        Returns:
            A new dict with the same keys. Sections that do not fit are set
            to empty strings. The input dict is not mutated.
        """
        if not sections:
            return {}

        budget = self.available_budget

        # Sort sections by priority (highest priority first = lowest number).
        sorted_names = sorted(sections.keys(), key=self._section_priority)

        # Calculate token cost for each section.
        token_costs: dict[str, int] = {
            name: self.estimate_tokens(text) for name, text in sections.items()
        }

        # Greedily include sections in priority order.
        result: dict[str, str] = {}
        used: int = 0
        included: set[str] = set()

        for name in sorted_names:
            cost = token_costs[name]
            if used + cost <= budget:
                result[name] = sections[name]
                used += cost
                included.add(name)
            else:
                result[name] = ""

        return result
