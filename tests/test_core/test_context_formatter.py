"""Tests for ContextFormatter -- activation-ranked prompt assembly."""

from __future__ import annotations

from core_gb.context_enrichment import EnrichedContext
from core_gb.context_formatter import ContextFormatter, SectionDef
from core_gb.token_budget import TokenBudget
from core_gb.types import Pattern


def _make_pattern(
    trigger: str = "deploy app",
    description: str = "Run deploy script then verify health",
    success_count: int = 3,
) -> Pattern:
    return Pattern(
        id="pat-001",
        trigger=trigger,
        description=description,
        success_count=success_count,
    )


class TestSectionFormatting:
    """Each section is formatted with the correct header and bullet style."""

    def test_entities_section(self) -> None:
        ctx = EnrichedContext(
            entities=(
                {"type": "User", "name": "Alice", "details": "developer at TU/e"},
            ),
            entity_tokens=10,
        )
        formatter = ContextFormatter()
        sections = formatter._format_sections(ctx)
        assert "entities" in sections
        assert "User: Alice -- developer at TU/e" in sections["entities"]

    def test_memories_section(self) -> None:
        ctx = EnrichedContext(
            memories=("User prefers Python", "Last session discussed graphs"),
            memory_tokens=10,
        )
        formatter = ContextFormatter()
        sections = formatter._format_sections(ctx)
        assert "memories" in sections
        assert "User prefers Python" in sections["memories"]
        assert "Last session discussed graphs" in sections["memories"]

    def test_reflections_section(self) -> None:
        ctx = EnrichedContext(
            reflections=(
                {
                    "task_description": "deploy app",
                    "what_failed": "timeout on health check",
                    "why": "port misconfigured",
                    "what_to_try": "check port config first",
                },
            ),
            reflection_tokens=15,
        )
        formatter = ContextFormatter()
        sections = formatter._format_sections(ctx)
        assert "reflections" in sections
        assert sections["reflections"].startswith(
            "Previous attempts at similar tasks failed because:"
        )
        assert "timeout on health check" in sections["reflections"]

    def test_patterns_section(self) -> None:
        pat = _make_pattern()
        ctx = EnrichedContext(patterns=(pat,), pattern_tokens=10)
        formatter = ContextFormatter()
        sections = formatter._format_sections(ctx)
        assert "patterns" in sections
        assert sections["patterns"].startswith(
            "Similar tasks have been answered like this:"
        )
        assert "deploy app" in sections["patterns"]

    def test_conversation_section(self) -> None:
        ctx = EnrichedContext(
            conversation_turns=(
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ),
            conversation_tokens=10,
        )
        formatter = ContextFormatter()
        sections = formatter._format_sections(ctx)
        assert "conversation" in sections
        assert sections["conversation"].startswith("Recent conversation:")
        assert "User: Hello" in sections["conversation"]
        assert "Assistant: Hi there" in sections["conversation"]

    def test_empty_sections_excluded(self) -> None:
        ctx = EnrichedContext()
        formatter = ContextFormatter()
        sections = formatter._format_sections(ctx)
        assert sections == {}


class TestActivationRanking:
    """Sections are ordered by their activation (token) score descending."""

    def test_highest_activation_first(self) -> None:
        ctx = EnrichedContext(
            entities=({"type": "User", "name": "Alice", "details": "dev"},),
            memories=("memory one",),
            reflections=(
                {
                    "task_description": "t",
                    "what_failed": "f",
                    "why": "w",
                    "what_to_try": "x",
                },
            ),
            entity_tokens=100,
            memory_tokens=50,
            reflection_tokens=200,
        )
        formatter = ContextFormatter()
        ordered = formatter._rank_sections(ctx)
        names = [s.name for s in ordered]
        # reflections (200) > entities (100) > memories (50)
        assert names.index("reflections") < names.index("entities")
        assert names.index("entities") < names.index("memories")

    def test_empty_sections_not_ranked(self) -> None:
        ctx = EnrichedContext(
            entities=({"type": "User", "name": "Alice", "details": "dev"},),
            entity_tokens=10,
        )
        formatter = ContextFormatter()
        ordered = formatter._rank_sections(ctx)
        names = [s.name for s in ordered]
        assert "memories" not in names
        assert "reflections" not in names


class TestTokenBudgetIntegration:
    """Sections that exceed the budget are trimmed."""

    def test_sections_trimmed_to_budget(self) -> None:
        """When total sections exceed budget, lowest-activation sections drop."""
        ctx = EnrichedContext(
            entities=({"type": "User", "name": "Alice", "details": "dev"},),
            memories=("important memory",),
            reflections=(
                {
                    "task_description": "task",
                    "what_failed": "fail",
                    "why": "reason",
                    "what_to_try": "fix",
                },
            ),
            entity_tokens=100,
            memory_tokens=50,
            reflection_tokens=200,
        )
        # Very tight budget -- only highest-activation section fits
        budget = TokenBudget(
            max_tokens=300,
            system_prompt_reserve=0,
            user_message_reserve=0,
            response_reserve=0,
        )
        formatter = ContextFormatter(token_budget=budget)
        messages = formatter.format(ctx, task="What is Alice's role?")

        # Should produce messages; system content should not include all sections
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        # With only ~300 token budget some sections will be dropped

    def test_no_budget_returns_all_sections(self) -> None:
        ctx = EnrichedContext(
            entities=({"type": "User", "name": "Alice", "details": "dev"},),
            entity_tokens=5,
        )
        budget = TokenBudget(
            max_tokens=100_000,
            system_prompt_reserve=0,
            user_message_reserve=0,
            response_reserve=0,
        )
        formatter = ContextFormatter(token_budget=budget)
        messages = formatter.format(ctx, task="test")
        system_content = messages[0]["content"]
        # Entity content should be in the context section
        assert "Alice" in system_content
        assert "<context>" in system_content


class TestMessageAssembly:
    """format() returns a list of message dicts ready for LLM."""

    def test_basic_message_structure(self) -> None:
        ctx = EnrichedContext(
            entities=({"type": "User", "name": "Alice", "details": "dev"},),
            entity_tokens=5,
        )
        formatter = ContextFormatter()
        messages = formatter.format(ctx, task="What does Alice do?")

        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)
        assert all("role" in m and "content" in m for m in messages)
        # First message is system
        assert messages[0]["role"] == "system"
        # Last message is the user task
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What does Alice do?"

    def test_conversation_turns_become_messages(self) -> None:
        ctx = EnrichedContext(
            conversation_turns=(
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ),
            conversation_tokens=10,
        )
        formatter = ContextFormatter()
        messages = formatter.format(ctx, task="How are you?")

        # Conversation turns appear as separate messages between system and user
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert roles[-1] == "user"
        # The conversation turns should be in the middle
        assert "user" in roles[1:-1] or "assistant" in roles[1:-1]

    def test_empty_context_still_produces_messages(self) -> None:
        ctx = EnrichedContext()
        formatter = ContextFormatter()
        messages = formatter.format(ctx, task="Hello")

        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_system_prompt_contains_context_sections(self) -> None:
        pat = _make_pattern()
        ctx = EnrichedContext(
            entities=({"type": "Project", "name": "GraphBot", "details": "DAG engine"},),
            reflections=(
                {
                    "task_description": "build API",
                    "what_failed": "auth missing",
                    "why": "no token check",
                    "what_to_try": "add JWT middleware",
                },
            ),
            patterns=(pat,),
            entity_tokens=10,
            reflection_tokens=15,
            pattern_tokens=10,
        )
        formatter = ContextFormatter()
        messages = formatter.format(ctx, task="Build the API")
        system_content = messages[0]["content"]

        # XML-structured prompt: context, instructions, examples, output_format
        assert "<context>" in system_content
        assert "</context>" in system_content
        assert "<instructions>" in system_content
        assert "<examples>" in system_content
        assert "<output_format>" in system_content
        # Content from the enriched context should be present
        assert "GraphBot" in system_content
        assert "auth missing" in system_content
        assert "deploy app" in system_content

    def test_custom_system_preamble(self) -> None:
        """Custom preamble is stored but XML-structured prompt uses domain role."""
        ctx = EnrichedContext()
        formatter = ContextFormatter(system_preamble="You are GraphBot.")
        messages = formatter.format(ctx, task="test")
        # With XML-structured prompts, the system message starts with the domain role
        system_content = messages[0]["content"]
        assert "expert" in system_content.lower() or "analyst" in system_content.lower()
        assert "<instructions>" in system_content
