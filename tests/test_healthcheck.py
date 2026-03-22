"""Tests for scripts/healthcheck.py."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.healthcheck import (
    CheckResult,
    HealthReport,
    _validate_key_format,
    check_api_keys,
    check_graph_db,
    check_graph_persistence,
    check_langsmith,
    check_multi_provider,
    check_telegram,
    check_whatsapp,
    load_env_file,
    run_healthcheck,
)


# ---------------------------------------------------------------------------
# HealthReport
# ---------------------------------------------------------------------------

class TestHealthReport:
    def test_empty_report_passes(self) -> None:
        report = HealthReport()
        assert report.all_critical_passed is True

    def test_critical_failure_fails_report(self) -> None:
        report = HealthReport()
        report.add(CheckResult(name="x", passed=False, message="bad", critical=True))
        assert report.all_critical_passed is False

    def test_non_critical_failure_still_passes(self) -> None:
        report = HealthReport()
        report.add(CheckResult(name="x", passed=True, message="ok", critical=True))
        report.add(CheckResult(name="y", passed=False, message="warn", critical=False))
        assert report.all_critical_passed is True

    def test_multiple_critical_all_pass(self) -> None:
        report = HealthReport()
        report.add(CheckResult(name="a", passed=True, message="ok", critical=True))
        report.add(CheckResult(name="b", passed=True, message="ok", critical=True))
        assert report.all_critical_passed is True


# ---------------------------------------------------------------------------
# Key format validation
# ---------------------------------------------------------------------------

class TestKeyFormatValidation:
    def test_openrouter_valid(self) -> None:
        key = "sk-or-v1-" + "a" * 64
        passed, reason = _validate_key_format("OPENROUTER_API_KEY", key)
        assert passed is True
        assert reason == ""

    def test_openrouter_invalid(self) -> None:
        passed, reason = _validate_key_format("OPENROUTER_API_KEY", "bad-key")
        assert passed is False
        assert "pattern" in reason

    def test_google_valid(self) -> None:
        key = "AI" + "a" * 37
        passed, _ = _validate_key_format("GOOGLE_API_KEY", key)
        assert passed is True

    def test_google_invalid(self) -> None:
        passed, _ = _validate_key_format("GOOGLE_API_KEY", "not-a-google-key")
        assert passed is False

    def test_groq_valid(self) -> None:
        key = "gsk_" + "a" * 50
        passed, _ = _validate_key_format("GROQ_API_KEY", key)
        assert passed is True

    def test_groq_invalid(self) -> None:
        passed, _ = _validate_key_format("GROQ_API_KEY", "xyz")
        assert passed is False

    def test_telegram_valid(self) -> None:
        key = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ_0123456"
        passed, _ = _validate_key_format("TELEGRAM_BOT_TOKEN", key)
        assert passed is True

    def test_telegram_invalid(self) -> None:
        passed, _ = _validate_key_format("TELEGRAM_BOT_TOKEN", "not-a-token")
        assert passed is False

    def test_langchain_valid(self) -> None:
        key = "lsv2_abc123_def"
        passed, _ = _validate_key_format("LANGCHAIN_API_KEY", key)
        assert passed is True

    def test_langchain_invalid(self) -> None:
        passed, _ = _validate_key_format("LANGCHAIN_API_KEY", "plain-key")
        assert passed is False

    def test_unknown_key_nonempty_passes(self) -> None:
        passed, _ = _validate_key_format("SOME_UNKNOWN_KEY", "any-value")
        assert passed is True

    def test_unknown_key_empty_fails(self) -> None:
        passed, _ = _validate_key_format("SOME_UNKNOWN_KEY", "")
        assert passed is False


# ---------------------------------------------------------------------------
# check_api_keys
# ---------------------------------------------------------------------------

class TestCheckApiKeys:
    def test_all_keys_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 64)
        monkeypatch.setenv("GOOGLE_API_KEY", "AI" + "b" * 37)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "c" * 50)

        report = HealthReport()
        check_api_keys(report)

        # OpenRouter + Google + Groq + "at least one provider"
        assert len(report.results) == 4
        assert all(r.passed for r in report.results)

    def test_no_keys_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        report = HealthReport()
        check_api_keys(report)

        # All individual checks fail + the "at least one" check fails.
        assert not report.all_critical_passed
        at_least_one = [r for r in report.results if "at least" in r.name.lower()]
        assert len(at_least_one) == 1
        assert at_least_one[0].passed is False

    def test_only_openrouter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 64)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        report = HealthReport()
        check_api_keys(report)

        assert report.all_critical_passed
        openrouter = [r for r in report.results if "OpenRouter" in r.name]
        assert openrouter[0].passed is True

    def test_invalid_format_still_counted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A key with wrong format is set but does not count as valid."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "bad-format")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        report = HealthReport()
        check_api_keys(report)

        openrouter = [r for r in report.results if "OpenRouter" in r.name]
        assert openrouter[0].passed is False


# ---------------------------------------------------------------------------
# check_graph_db
# ---------------------------------------------------------------------------

class TestCheckGraphDb:
    def test_valid_directory(self, tmp_path: Path) -> None:
        report = HealthReport()
        with patch("scripts.healthcheck.DB_PATH", tmp_path / "graphbot.db"):
            check_graph_db(report)
        assert report.results[0].passed is True

    def test_missing_directory(self, tmp_path: Path) -> None:
        report = HealthReport()
        missing = tmp_path / "nonexistent" / "graphbot.db"
        with patch("scripts.healthcheck.DB_PATH", missing):
            check_graph_db(report)
        assert report.results[0].passed is False
        assert "does not exist" in report.results[0].message


# ---------------------------------------------------------------------------
# check_graph_persistence
# ---------------------------------------------------------------------------

class TestCheckGraphPersistence:
    def test_persistence_passes(self) -> None:
        report = HealthReport()
        check_graph_persistence(report)
        result = report.results[0]
        assert result.passed is True
        assert "persist" in result.message.lower()

    def test_persistence_without_kuzu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If kuzu is not importable, check should fail gracefully."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name == "kuzu":
                raise ImportError("no kuzu")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        report = HealthReport()
        check_graph_persistence(report)
        assert report.results[0].passed is False
        assert "not installed" in report.results[0].message


# ---------------------------------------------------------------------------
# check_whatsapp
# ---------------------------------------------------------------------------

class TestCheckWhatsApp:
    def test_both_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WHATSAPP_BRIDGE_URL", "ws://localhost:3001")
        monkeypatch.setenv("WHATSAPP_BRIDGE_TOKEN", "secret")

        report = HealthReport()
        check_whatsapp(report)

        assert len(report.results) == 2
        assert all(r.passed for r in report.results)

    def test_neither_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WHATSAPP_BRIDGE_URL", raising=False)
        monkeypatch.delenv("WHATSAPP_BRIDGE_TOKEN", raising=False)

        report = HealthReport()
        check_whatsapp(report)

        # Both should fail but are non-critical.
        assert len(report.results) == 2
        assert all(not r.passed for r in report.results)
        assert all(not r.critical for r in report.results)


# ---------------------------------------------------------------------------
# check_telegram
# ---------------------------------------------------------------------------

class TestCheckTelegram:
    def test_valid_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ_0123456")
        report = HealthReport()
        check_telegram(report)
        assert report.results[0].passed is True

    def test_missing_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        report = HealthReport()
        check_telegram(report)
        assert report.results[0].passed is False

    def test_invalid_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "not-a-valid-token")
        report = HealthReport()
        check_telegram(report)
        assert report.results[0].passed is False
        assert "format" in report.results[0].message.lower() or "pattern" in report.results[0].message.lower()


# ---------------------------------------------------------------------------
# check_multi_provider
# ---------------------------------------------------------------------------

class TestCheckMultiProvider:
    def test_two_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 64)
        monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "b" * 50)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        report = HealthReport()
        check_multi_provider(report)
        assert report.results[0].passed is True
        assert "rotation enabled" in report.results[0].message

    def test_one_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 64)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        report = HealthReport()
        check_multi_provider(report)
        assert report.results[0].passed is False
        assert "1 provider" in report.results[0].message

    def test_no_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        report = HealthReport()
        check_multi_provider(report)
        assert report.results[0].passed is False
        assert "No provider" in report.results[0].message


# ---------------------------------------------------------------------------
# check_langsmith
# ---------------------------------------------------------------------------

class TestCheckLangsmith:
    def test_valid_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_abc123_def")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "my-project")

        report = HealthReport()
        check_langsmith(report)
        assert report.results[0].passed is True
        assert "my-project" in report.results[0].message

    def test_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        report = HealthReport()
        check_langsmith(report)
        assert report.results[0].passed is False
        assert "disabled" in report.results[0].message

    def test_invalid_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGCHAIN_API_KEY", "plain-key")

        report = HealthReport()
        check_langsmith(report)
        assert report.results[0].passed is False


# ---------------------------------------------------------------------------
# load_env_file
# ---------------------------------------------------------------------------

class TestLoadEnvFile:
    def test_loads_from_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env.local"
        env_file.write_text("HC_TEST_VAR=hello_world\n")
        monkeypatch.delenv("HC_TEST_VAR", raising=False)

        load_env_file(env_file)

        assert os.environ.get("HC_TEST_VAR") == "hello_world"
        # Cleanup
        monkeypatch.delenv("HC_TEST_VAR", raising=False)

    def test_skips_comments_and_blanks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env.local"
        env_file.write_text("# comment\n\nHC_TEST_VAR2=value2\n")
        monkeypatch.delenv("HC_TEST_VAR2", raising=False)

        load_env_file(env_file)

        assert os.environ.get("HC_TEST_VAR2") == "value2"
        monkeypatch.delenv("HC_TEST_VAR2", raising=False)

    def test_does_not_override_existing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env.local"
        env_file.write_text("HC_TEST_VAR3=from_file\n")
        monkeypatch.setenv("HC_TEST_VAR3", "already_set")

        load_env_file(env_file)

        assert os.environ.get("HC_TEST_VAR3") == "already_set"

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        # Should not raise.
        load_env_file(missing)


# ---------------------------------------------------------------------------
# run_healthcheck (integration)
# ---------------------------------------------------------------------------

class TestRunHealthcheck:
    def test_full_run_does_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Smoke test: run_healthcheck should complete without raising."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 64)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("WHATSAPP_BRIDGE_URL", raising=False)
        monkeypatch.delenv("WHATSAPP_BRIDGE_TOKEN", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        report = run_healthcheck(load_env=False)
        assert isinstance(report, HealthReport)
        assert len(report.results) > 0

    def test_all_critical_pass_with_valid_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 64)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        report = run_healthcheck(load_env=False)
        assert report.all_critical_passed is True

    def test_critical_fail_without_any_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        report = run_healthcheck(load_env=False)
        assert report.all_critical_passed is False

    def test_report_messages_are_human_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every result message should be a non-empty human-readable string."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 64)

        report = run_healthcheck(load_env=False)
        for result in report.results:
            assert isinstance(result.message, str)
            assert len(result.message) > 0
            # No raw exception tracebacks or empty strings.
            assert result.message != "None"
