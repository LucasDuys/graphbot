"""Tests for LangSmith prompt management -- templates, versioning, trace tagging."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from core_gb.langsmith_prompts import (
    CURRENT_VERSION,
    PROMPT_TEMPLATES,
    LangSmithPromptManager,
    PromptVersion,
)


# ---------------------------------------------------------------------------
# Template registry tests
# ---------------------------------------------------------------------------

EXPECTED_NAMES: set[str] = {
    "simple_qa",
    "hard_reasoning",
    "tool_execution",
    "creative",
    "analysis",
}


class TestPromptTemplateRegistry:
    """All 5 prompt templates exist with correct structure."""

    def test_all_five_templates_exist(self) -> None:
        assert set(PROMPT_TEMPLATES.keys()) == EXPECTED_NAMES

    def test_templates_are_chat_prompt_templates(self) -> None:
        for name, template in PROMPT_TEMPLATES.items():
            assert isinstance(template, ChatPromptTemplate), (
                f"{name} is not a ChatPromptTemplate"
            )

    def test_all_templates_have_context_parameter(self) -> None:
        for name, template in PROMPT_TEMPLATES.items():
            variables = template.input_variables
            assert "context" in variables, (
                f"{name} missing {{context}} parameter, has: {variables}"
            )

    def test_all_templates_have_examples_parameter(self) -> None:
        for name, template in PROMPT_TEMPLATES.items():
            variables = template.input_variables
            assert "examples" in variables, (
                f"{name} missing {{examples}} parameter, has: {variables}"
            )

    def test_all_templates_have_task_parameter(self) -> None:
        for name, template in PROMPT_TEMPLATES.items():
            variables = template.input_variables
            assert "task" in variables, (
                f"{name} missing {{task}} parameter, has: {variables}"
            )

    def test_templates_format_without_error(self) -> None:
        for name, template in PROMPT_TEMPLATES.items():
            messages = template.format_messages(
                context="test context",
                examples="test examples",
                task="test task",
            )
            assert len(messages) >= 2, (
                f"{name} produced {len(messages)} messages, expected >= 2"
            )

    def test_formatted_messages_contain_parameters(self) -> None:
        for name, template in PROMPT_TEMPLATES.items():
            messages = template.format_messages(
                context="UNIQUE_CONTEXT_MARKER",
                examples="UNIQUE_EXAMPLES_MARKER",
                task="UNIQUE_TASK_MARKER",
            )
            all_content = " ".join(m.content for m in messages)
            assert "UNIQUE_CONTEXT_MARKER" in all_content, (
                f"{name} did not include context in output"
            )
            assert "UNIQUE_EXAMPLES_MARKER" in all_content, (
                f"{name} did not include examples in output"
            )
            assert "UNIQUE_TASK_MARKER" in all_content, (
                f"{name} did not include task in output"
            )


# ---------------------------------------------------------------------------
# PromptVersion dataclass
# ---------------------------------------------------------------------------

class TestPromptVersion:
    def test_prompt_version_fields(self) -> None:
        pv = PromptVersion(name="simple_qa", version="v1", url="https://example.com")
        assert pv.name == "simple_qa"
        assert pv.version == "v1"
        assert pv.url == "https://example.com"

    def test_prompt_version_frozen(self) -> None:
        pv = PromptVersion(name="test", version="v1")
        with pytest.raises(AttributeError):
            pv.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LangSmithPromptManager
# ---------------------------------------------------------------------------

class TestPromptManagerGetTemplate:
    """Template retrieval by name and version."""

    def test_get_template_by_name(self) -> None:
        mgr = LangSmithPromptManager()
        for name in EXPECTED_NAMES:
            template = mgr.get_template(name)
            assert isinstance(template, ChatPromptTemplate)
            assert template is PROMPT_TEMPLATES[name]

    def test_get_template_unknown_name_raises(self) -> None:
        mgr = LangSmithPromptManager()
        with pytest.raises(KeyError, match="Unknown prompt name"):
            mgr.get_template("nonexistent_prompt")

    def test_get_template_with_version_falls_back_on_error(self) -> None:
        mgr = LangSmithPromptManager()
        with patch.object(mgr, "_get_client") as mock_client:
            mock_client.return_value.pull_prompt.side_effect = Exception("no hub")
            template = mgr.get_template("simple_qa", version="v1")
            assert template is PROMPT_TEMPLATES["simple_qa"]

    def test_get_all_names(self) -> None:
        mgr = LangSmithPromptManager()
        names = mgr.get_all_names()
        assert set(names) == EXPECTED_NAMES


class TestPromptManagerVersionTracking:
    """Version tracking across push operations."""

    def test_versions_empty_initially(self) -> None:
        mgr = LangSmithPromptManager()
        for name in EXPECTED_NAMES:
            assert mgr.get_versions(name) == []

    def test_push_records_versions(self) -> None:
        mgr = LangSmithPromptManager()
        with patch.object(mgr, "_get_client") as mock_client:
            mock_client.return_value.push_prompt.return_value = (
                "https://smith.langchain.com/prompts/test"
            )
            pushed = mgr.push_templates(version="v1")

        assert len(pushed) == 5
        for pv in pushed:
            assert pv.version == "v1"
            assert pv.name in EXPECTED_NAMES
            versions = mgr.get_versions(pv.name)
            assert len(versions) == 1
            assert versions[0].version == "v1"

    def test_push_multiple_versions_tracked(self) -> None:
        mgr = LangSmithPromptManager()
        with patch.object(mgr, "_get_client") as mock_client:
            mock_client.return_value.push_prompt.return_value = "https://url"
            mgr.push_templates(version="v1")
            mgr.push_templates(version="v2")

        for name in EXPECTED_NAMES:
            versions = mgr.get_versions(name)
            assert len(versions) == 2
            assert versions[0].version == "v1"
            assert versions[1].version == "v2"


class TestPromptManagerPush:
    """Push templates to LangSmith hub (mocked)."""

    def test_push_calls_client_for_each_template(self) -> None:
        mgr = LangSmithPromptManager()
        with patch.object(mgr, "_get_client") as mock_client:
            client_instance = MagicMock()
            client_instance.push_prompt.return_value = "https://url"
            mock_client.return_value = client_instance

            pushed = mgr.push_templates(version="v1")

        assert client_instance.push_prompt.call_count == 5
        assert len(pushed) == 5

        # Verify prompt identifiers include prefix.
        call_args = [
            call.args[0]
            for call in client_instance.push_prompt.call_args_list
        ]
        for name in EXPECTED_NAMES:
            assert f"graphbot-{name}" in call_args

    def test_push_uses_custom_prefix(self) -> None:
        mgr = LangSmithPromptManager(prefix="myproject")
        with patch.object(mgr, "_get_client") as mock_client:
            client_instance = MagicMock()
            client_instance.push_prompt.return_value = "https://url"
            mock_client.return_value = client_instance

            mgr.push_templates(version="v1")

        call_args = [
            call.args[0]
            for call in client_instance.push_prompt.call_args_list
        ]
        for name in EXPECTED_NAMES:
            assert f"myproject-{name}" in call_args

    def test_push_handles_client_error_gracefully(self) -> None:
        mgr = LangSmithPromptManager()
        with patch.object(mgr, "_get_client") as mock_client:
            client_instance = MagicMock()
            client_instance.push_prompt.side_effect = Exception("API error")
            mock_client.return_value = client_instance

            pushed = mgr.push_templates(version="v1")

        # Should still return all 5, with empty URLs.
        assert len(pushed) == 5
        for pv in pushed:
            assert pv.url == ""


# ---------------------------------------------------------------------------
# Trace tagging
# ---------------------------------------------------------------------------

class TestTraceTagging:
    """Trace tagging produces correct metadata dicts."""

    def test_tag_trace_returns_metadata(self) -> None:
        meta = LangSmithPromptManager.tag_trace(
            run_id="abc-123",
            prompt_name="simple_qa",
            version="v1",
        )
        assert meta["run_id"] == "abc-123"
        assert meta["prompt_name"] == "simple_qa"
        assert meta["prompt_version"] == "v1"

    def test_make_litellm_metadata(self) -> None:
        meta = LangSmithPromptManager.make_litellm_metadata(
            prompt_name="hard_reasoning",
            version="v2",
        )
        assert meta["prompt_name"] == "hard_reasoning"
        assert meta["prompt_version"] == "v2"
        assert "prompt:hard_reasoning" in meta["tags"]
        assert "version:v2" in meta["tags"]

    def test_tag_trace_is_static(self) -> None:
        # Can be called without an instance.
        meta = LangSmithPromptManager.tag_trace("id", "name", "v1")
        assert isinstance(meta, dict)

    def test_make_litellm_metadata_is_static(self) -> None:
        meta = LangSmithPromptManager.make_litellm_metadata("name", "v1")
        assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# CURRENT_VERSION constant
# ---------------------------------------------------------------------------

class TestCurrentVersion:
    def test_current_version_is_string(self) -> None:
        assert isinstance(CURRENT_VERSION, str)
        assert len(CURRENT_VERSION) > 0

    def test_current_version_starts_with_v(self) -> None:
        assert CURRENT_VERSION.startswith("v")
