"""Tests for DecompositionPrompt (T014, R003)."""

from __future__ import annotations

import json
import re

from core_gb.decomposer import DecompositionPrompt, validate_decomposition
from core_gb.types import GraphContext


def _extract_json_blocks(text: str) -> list[dict]:
    """Extract all JSON objects from text (greedy brace matching)."""
    blocks: list[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    blocks.append(json.loads(text[start : i + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1
    return blocks


class TestBuildWithoutContext:
    """test_build_without_context: verify messages structure (system + user)."""

    def test_returns_list_of_dicts(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Write a poem")
        assert isinstance(messages, list)
        assert len(messages) >= 2

    def test_first_message_is_system(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Write a poem")
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Write a poem")
        assert messages[-1]["role"] == "user"

    def test_user_message_contains_task(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Write a poem about cats")
        assert "Write a poem about cats" in messages[-1]["content"]


class TestBuildWithContext:
    """test_build_with_context: verify context appears in system message."""

    def test_context_in_system_message(self) -> None:
        ctx = GraphContext(
            user_summary="Lucas is a CS student",
            active_memories=("prefers Python",),
        )
        prompt = DecompositionPrompt()
        messages = prompt.build("Write code", context=ctx)
        system_content = messages[0]["content"]
        assert "Lucas is a CS student" in system_content

    def test_no_context_block_without_context(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Write code", context=None)
        system_content = messages[0]["content"]
        # Should not have context section when no context
        assert "<context>" not in system_content


class TestPromptContainsXmlTags:
    """test_prompt_contains_xml_tags: verify <rules>, <output_schema>, <examples>."""

    def test_rules_tag(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "<rules>" in system
        assert "</rules>" in system

    def test_output_schema_tag(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "<output_schema>" in system
        assert "</output_schema>" in system

    def test_examples_tag(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "<examples>" in system
        assert "</examples>" in system


class TestPromptContainsExamples:
    """test_prompt_contains_examples: verify both good and bad examples."""

    def test_good_examples_present(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "GOOD" in system
        # At least 2 good examples
        good_count = system.count("GOOD")
        assert good_count >= 2, f"Expected >=2 good examples, found {good_count}"

    def test_bad_example_present(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "BAD" in system
        assert "WRONG" in system


class TestPromptSandwichDefense:
    """test_prompt_sandwich_defense: output format restated after task."""

    def test_format_reminder_in_user_message(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Build a dashboard")
        user_content = messages[-1]["content"]
        # Must contain the task
        assert "Build a dashboard" in user_content
        # Must restate output format AFTER the task
        task_pos = user_content.index("Build a dashboard")
        # There should be a JSON format reminder after the task
        remainder = user_content[task_pos:]
        assert "json" in remainder.lower() or "JSON" in remainder


class TestPromptTokenBudget:
    """test_prompt_token_budget: estimate tokens, verify < 1500."""

    def test_token_count_under_budget(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Write a poem about cats")
        total_text = " ".join(m["content"] for m in messages)
        # Estimate: words * 1.3
        word_count = len(total_text.split())
        estimated_tokens = int(word_count * 1.3)
        assert estimated_tokens < 1500, (
            f"Estimated {estimated_tokens} tokens (from {word_count} words), budget is 1500"
        )

    def test_token_count_with_context_under_budget(self) -> None:
        ctx = GraphContext(
            user_summary="Software developer working on GraphBot",
            active_memories=("uses Python", "prefers TDD"),
        )
        prompt = DecompositionPrompt()
        messages = prompt.build("Write a poem about cats", context=ctx)
        total_text = " ".join(m["content"] for m in messages)
        word_count = len(total_text.split())
        estimated_tokens = int(word_count * 1.3)
        assert estimated_tokens < 1500, (
            f"Estimated {estimated_tokens} tokens (from {word_count} words), budget is 1500"
        )


class TestExampleTreesValidate:
    """test_example_trees_validate: extract JSON from examples, validate against schema."""

    def test_all_json_examples_validate(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Test task")
        system = messages[0]["content"]

        # Extract examples section
        examples_match = re.search(
            r"<examples>(.*?)</examples>", system, re.DOTALL
        )
        assert examples_match, "No <examples> section found"
        examples_text = examples_match.group(1)

        # Extract all JSON blocks
        json_blocks = _extract_json_blocks(examples_text)
        assert len(json_blocks) >= 2, (
            f"Expected >=2 JSON examples, found {len(json_blocks)}"
        )

        # Find GOOD examples and validate them
        good_section = examples_text.split("BAD")[0]
        good_blocks = _extract_json_blocks(good_section)
        assert len(good_blocks) >= 2, (
            f"Expected >=2 GOOD JSON examples, found {len(good_blocks)}"
        )

        for i, block in enumerate(good_blocks):
            errors = validate_decomposition(block)
            assert errors == [], (
                f"GOOD example {i+1} failed validation: {errors}"
            )

    def test_bad_example_is_present_but_explained(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Test task")
        system = messages[0]["content"]

        examples_match = re.search(
            r"<examples>(.*?)</examples>", system, re.DOTALL
        )
        assert examples_match
        examples_text = examples_match.group(1)

        # BAD section must exist with explanation
        assert "BAD" in examples_text
        assert "WRONG" in examples_text


class TestPromptMentionsOutputTemplate:
    """Verify the prompt instructs the model to produce output_template."""

    def test_rules_mention_output_template(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "output_template" in system

    def test_rules_mention_aggregation_type(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "aggregation_type" in system

    def test_rules_mention_template_fill(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "template_fill" in system

    def test_rules_mention_concatenate(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "concatenate" in system

    def test_rules_mention_slot_definitions(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        assert "slot_definitions" in system

    def test_output_schema_mentions_output_template(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        schema_match = re.search(
            r"<output_schema>(.*?)</output_schema>", system, re.DOTALL
        )
        assert schema_match
        assert "output_template" in schema_match.group(1)

    def test_good_examples_contain_output_template(self) -> None:
        prompt = DecompositionPrompt()
        messages = prompt.build("Do something")
        system = messages[0]["content"]
        examples_match = re.search(
            r"<examples>(.*?)</examples>", system, re.DOTALL
        )
        assert examples_match
        good_section = examples_match.group(1).split("BAD")[0]
        assert "output_template" in good_section
