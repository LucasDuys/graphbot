"""Tests for LangSmith observability setup."""

from __future__ import annotations

from unittest.mock import patch

import litellm
import pytest

from models.observability import setup_langsmith


@pytest.fixture(autouse=True)
def _clean_litellm_callbacks():
    """Save and restore litellm.success_callback around each test."""
    original = litellm.success_callback[:]
    yield
    litellm.success_callback[:] = original


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove LangSmith env vars before each test."""
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)


class TestSetupLangsmith:
    def test_returns_true_when_api_key_set(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
        assert setup_langsmith() is True

    def test_callback_registered_when_api_key_set(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
        setup_langsmith()
        assert "langsmith" in litellm.success_callback

    def test_returns_false_without_api_key(self):
        assert setup_langsmith() is False

    def test_callback_not_added_without_api_key(self):
        setup_langsmith()
        assert "langsmith" not in litellm.success_callback

    def test_sets_default_project_name(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
        setup_langsmith()
        import os

        assert os.environ["LANGCHAIN_PROJECT"] == "graphbot"

    def test_respects_existing_project_name(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "my-custom-project")
        setup_langsmith()
        import os

        assert os.environ["LANGCHAIN_PROJECT"] == "my-custom-project"

    def test_no_duplicate_callback_on_double_call(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
        setup_langsmith()
        setup_langsmith()
        count = litellm.success_callback.count("langsmith")
        assert count == 1

    def test_does_not_crash_when_langsmith_missing(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_key")
        with patch.dict("sys.modules", {"langsmith": None}):
            result = setup_langsmith()
        # Should not crash; returns False because the import guard catches it
        assert result is False
