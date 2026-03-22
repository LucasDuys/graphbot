"""GraphBot healthcheck -- verify environment, database, and provider readiness.

Checks:
  1. Required/optional API keys are set and have valid format.
  2. Graph database (data/graphbot.db) is accessible and writable.
  3. WhatsApp bridge credentials are present (if channel enabled).
  4. Multi-provider rotation is configured (>1 provider key).
  5. Graph persistence across restarts (write + read-back).

Exit code 0 if all critical checks pass, 1 otherwise.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Resolve project root (two levels up from scripts/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = PROJECT_ROOT / "data" / "graphbot.db"


# -- Result types -----------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single healthcheck."""

    name: str
    passed: bool
    message: str
    critical: bool = True  # If True, failure causes non-zero exit.


@dataclass
class HealthReport:
    """Aggregated healthcheck report."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def all_critical_passed(self) -> bool:
        return all(r.passed for r in self.results if r.critical)


# -- Key format validators --------------------------------------------------

_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "OPENROUTER_API_KEY": re.compile(r"^sk-or-v1-[a-f0-9]{64}$"),
    "GOOGLE_API_KEY": re.compile(r"^AI[a-zA-Z0-9_-]{30,50}$"),
    "GROQ_API_KEY": re.compile(r"^gsk_[a-zA-Z0-9]{48,64}$"),
    "TELEGRAM_BOT_TOKEN": re.compile(r"^\d{8,15}:[A-Za-z0-9_-]{30,50}$"),
    "LANGCHAIN_API_KEY": re.compile(r"^lsv2_[a-zA-Z0-9_]+$"),
}


def _validate_key_format(env_var: str, value: str) -> tuple[bool, str]:
    """Check whether *value* matches the expected format for *env_var*.

    Returns (True, "") on success or (False, reason) on failure.
    """
    pattern = _KEY_PATTERNS.get(env_var)
    if pattern is None:
        # No known format -- accept any non-empty string.
        return (True, "") if value else (False, "value is empty")
    if pattern.match(value):
        return True, ""
    return False, f"does not match expected format (pattern: {pattern.pattern})"


# -- Individual checks -------------------------------------------------------

def check_api_keys(report: HealthReport) -> None:
    """Verify provider API keys are set and have valid format."""
    providers_found: int = 0

    for env_var, label, critical in [
        ("OPENROUTER_API_KEY", "OpenRouter API key", True),
        ("GOOGLE_API_KEY", "Google API key", False),
        ("GROQ_API_KEY", "Groq API key", False),
    ]:
        value = os.environ.get(env_var, "")
        if not value:
            report.add(CheckResult(
                name=label,
                passed=False,
                message=f"{env_var} is not set",
                critical=critical,
            ))
            continue

        valid, reason = _validate_key_format(env_var, value)
        if valid:
            providers_found += 1
            report.add(CheckResult(
                name=label,
                passed=True,
                message=f"{env_var} is set and format looks valid",
                critical=critical,
            ))
        else:
            report.add(CheckResult(
                name=label,
                passed=False,
                message=f"{env_var} is set but {reason}",
                critical=critical,
            ))

    # At least one provider key must be present.
    report.add(CheckResult(
        name="At least one provider key",
        passed=providers_found >= 1,
        message=(
            f"{providers_found} provider key(s) detected"
            if providers_found >= 1
            else "No provider API keys found -- set at least OPENROUTER_API_KEY"
        ),
        critical=True,
    ))


def check_graph_db(report: HealthReport) -> None:
    """Verify the graph database directory is accessible."""
    data_dir = DB_PATH.parent
    if not data_dir.exists():
        report.add(CheckResult(
            name="Graph DB directory",
            passed=False,
            message=f"Directory does not exist: {data_dir}",
            critical=True,
        ))
        return

    # Check the directory is writable.
    try:
        test_file = data_dir / ".healthcheck_probe"
        test_file.write_text("ok")
        test_file.unlink()
        report.add(CheckResult(
            name="Graph DB directory",
            passed=True,
            message=f"Directory exists and is writable: {data_dir}",
            critical=True,
        ))
    except OSError as exc:
        report.add(CheckResult(
            name="Graph DB directory",
            passed=False,
            message=f"Directory exists but is not writable: {exc}",
            critical=True,
        ))


def check_graph_persistence(report: HealthReport) -> None:
    """Write a node, close the store, reopen, and read it back."""
    try:
        import kuzu
    except ImportError:
        report.add(CheckResult(
            name="Graph persistence",
            passed=False,
            message="kuzu package not installed -- cannot verify graph persistence",
            critical=True,
        ))
        return

    with tempfile.TemporaryDirectory(prefix="graphbot_hc_") as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        probe_id = "healthcheck_probe_001"

        try:
            # Write phase: create a table and insert a node.
            db = kuzu.Database(db_path)
            conn = kuzu.Connection(db)
            conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS HCProbe(id STRING, val STRING, PRIMARY KEY(id))"
            )
            conn.execute(
                "CREATE (n:HCProbe {id: $id, val: $val})",
                parameters={"id": probe_id, "val": "persist_test"},
            )
            conn.close()
            db.close()  # Release file lock so reopen works on all platforms.

            # Read phase: reopen and verify.
            db2 = kuzu.Database(db_path)
            conn2 = kuzu.Connection(db2)
            result = conn2.execute(
                "MATCH (n:HCProbe) WHERE n.id = $id RETURN n.val AS val",
                parameters={"id": probe_id},
            )
            if result.has_next():
                row = result.get_next()
                if row[0] == "persist_test":
                    report.add(CheckResult(
                        name="Graph persistence",
                        passed=True,
                        message="Data persists across close/reopen cycle",
                        critical=True,
                    ))
                else:
                    report.add(CheckResult(
                        name="Graph persistence",
                        passed=False,
                        message=f"Read-back value mismatch: expected 'persist_test', got '{row[0]}'",
                        critical=True,
                    ))
            else:
                report.add(CheckResult(
                    name="Graph persistence",
                    passed=False,
                    message="Node not found after close/reopen",
                    critical=True,
                ))
            conn2.close()
            db2.close()
        except Exception as exc:
            report.add(CheckResult(
                name="Graph persistence",
                passed=False,
                message=f"Persistence check failed: {exc}",
                critical=True,
            ))


def check_whatsapp(report: HealthReport) -> None:
    """Verify WhatsApp bridge credentials are present."""
    bridge_url = os.environ.get("WHATSAPP_BRIDGE_URL", "")
    bridge_token = os.environ.get("WHATSAPP_BRIDGE_TOKEN", "")

    if bridge_url:
        report.add(CheckResult(
            name="WhatsApp bridge URL",
            passed=True,
            message=f"WHATSAPP_BRIDGE_URL is set: {bridge_url}",
            critical=False,
        ))
    else:
        report.add(CheckResult(
            name="WhatsApp bridge URL",
            passed=False,
            message="WHATSAPP_BRIDGE_URL is not set (defaults to ws://localhost:3001)",
            critical=False,
        ))

    if bridge_token:
        report.add(CheckResult(
            name="WhatsApp bridge token",
            passed=True,
            message="WHATSAPP_BRIDGE_TOKEN is set",
            critical=False,
        ))
    else:
        report.add(CheckResult(
            name="WhatsApp bridge token",
            passed=False,
            message="WHATSAPP_BRIDGE_TOKEN is not set (optional, depends on bridge config)",
            critical=False,
        ))


def check_telegram(report: HealthReport) -> None:
    """Verify Telegram bot token is present and has valid format."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        report.add(CheckResult(
            name="Telegram bot token",
            passed=False,
            message="TELEGRAM_BOT_TOKEN is not set",
            critical=False,
        ))
        return

    valid, reason = _validate_key_format("TELEGRAM_BOT_TOKEN", token)
    if valid:
        report.add(CheckResult(
            name="Telegram bot token",
            passed=True,
            message="TELEGRAM_BOT_TOKEN is set and format looks valid",
            critical=False,
        ))
    else:
        report.add(CheckResult(
            name="Telegram bot token",
            passed=False,
            message=f"TELEGRAM_BOT_TOKEN is set but {reason}",
            critical=False,
        ))


def check_multi_provider(report: HealthReport) -> None:
    """Verify that more than one provider API key is set (enables rotation)."""
    keys = ["OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]
    present = [k for k in keys if os.environ.get(k, "")]
    count = len(present)

    if count >= 2:
        report.add(CheckResult(
            name="Multi-provider rotation",
            passed=True,
            message=f"{count} provider keys set ({', '.join(present)}) -- rotation enabled",
            critical=False,
        ))
    elif count == 1:
        report.add(CheckResult(
            name="Multi-provider rotation",
            passed=False,
            message=f"Only 1 provider key set ({present[0]}) -- add more for failover rotation",
            critical=False,
        ))
    else:
        report.add(CheckResult(
            name="Multi-provider rotation",
            passed=False,
            message="No provider keys set -- cannot enable rotation",
            critical=False,
        ))


def check_langsmith(report: HealthReport) -> None:
    """Verify LangSmith tracing configuration."""
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        report.add(CheckResult(
            name="LangSmith tracing",
            passed=False,
            message="LANGCHAIN_API_KEY is not set -- tracing disabled",
            critical=False,
        ))
        return

    valid, reason = _validate_key_format("LANGCHAIN_API_KEY", api_key)
    if valid:
        project = os.environ.get("LANGCHAIN_PROJECT", "graphbot")
        report.add(CheckResult(
            name="LangSmith tracing",
            passed=True,
            message=f"LANGCHAIN_API_KEY is set, project='{project}'",
            critical=False,
        ))
    else:
        report.add(CheckResult(
            name="LangSmith tracing",
            passed=False,
            message=f"LANGCHAIN_API_KEY is set but {reason}",
            critical=False,
        ))


# -- Env loading -------------------------------------------------------------

def load_env_file(path: Path | None = None) -> None:
    """Load .env.local from project root (same logic as channel files)."""
    env_file = path or (PROJECT_ROOT / ".env.local")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


# -- Report printer ----------------------------------------------------------

def print_report(report: HealthReport) -> None:
    """Print a human-readable healthcheck report to stdout."""
    print()
    print("=" * 60)
    print("  GraphBot Healthcheck")
    print("=" * 60)
    print()

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        critical_tag = "" if result.critical else " (optional)"
        print(f"  [{status}] {result.name}{critical_tag}")
        print(f"         {result.message}")
        print()

    # Summary
    total = len(report.results)
    passed = sum(1 for r in report.results if r.passed)
    failed = total - passed
    critical_failed = sum(1 for r in report.results if not r.passed and r.critical)

    print("-" * 60)
    print(f"  Total: {total}  Passed: {passed}  Failed: {failed}")
    if critical_failed:
        print(f"  CRITICAL FAILURES: {critical_failed}")
        print("  Fix critical failures before running GraphBot.")
    else:
        print("  All critical checks passed.")
    print("-" * 60)
    print()


# -- Main entry point --------------------------------------------------------

def run_healthcheck(load_env: bool = True) -> HealthReport:
    """Execute all healthchecks and return the report."""
    if load_env:
        load_env_file()

    report = HealthReport()
    check_api_keys(report)
    check_graph_db(report)
    check_graph_persistence(report)
    check_whatsapp(report)
    check_telegram(report)
    check_multi_provider(report)
    check_langsmith(report)
    return report


def main() -> None:
    """CLI entry point."""
    report = run_healthcheck()
    print_report(report)
    sys.exit(0 if report.all_critical_passed else 1)


if __name__ == "__main__":
    main()
