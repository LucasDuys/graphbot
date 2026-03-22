"""Tests for core_gb.constitution -- constitutional principles and plan verification.

Tests that harmful plans are rejected, deceptive plans are rejected, and benign
plans pass constitutional review. Each principle is a callable check function
that scans TaskNode descriptions and tool_params for violations.
"""

from __future__ import annotations

import pytest

from core_gb.constitution import (
    ConstitutionalChecker,
    ConstitutionalVerdict,
    PRINCIPLES,
    check_deception,
    check_no_harm,
    check_privacy,
    check_side_effects,
    check_unauthorized_access,
)
from core_gb.types import Domain, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    description: str,
    domain: Domain = Domain.SYSTEM,
    is_atomic: bool = True,
    tool_params: dict[str, str] | None = None,
) -> TaskNode:
    """Create a minimal TaskNode for constitutional tests."""
    node = TaskNode(
        id=node_id,
        description=description,
        is_atomic=is_atomic,
        domain=domain,
        complexity=1,
        status=TaskStatus.READY,
    )
    if tool_params:
        node.tool_params = tool_params
    return node


# ---------------------------------------------------------------------------
# ConstitutionalVerdict dataclass
# ---------------------------------------------------------------------------


class TestConstitutionalVerdict:
    """ConstitutionalVerdict holds pass/fail and a list of violations."""

    def test_passed_verdict(self) -> None:
        verdict = ConstitutionalVerdict(passed=True, violations=[])
        assert verdict.passed is True
        assert verdict.violations == []

    def test_failed_verdict(self) -> None:
        verdict = ConstitutionalVerdict(
            passed=False,
            violations=[("no_harm", "Plan includes destructive operations")],
        )
        assert verdict.passed is False
        assert len(verdict.violations) == 1
        assert verdict.violations[0][0] == "no_harm"
        assert "destructive" in verdict.violations[0][1]


# ---------------------------------------------------------------------------
# Principle: no harm
# ---------------------------------------------------------------------------


class TestNoHarmPrinciple:
    """The no-harm principle rejects plans that cause physical or digital harm."""

    def test_destroy_data_rejected(self) -> None:
        nodes = [_node("n1", "Delete all user data permanently")]
        violations = check_no_harm(nodes)
        assert len(violations) > 0

    def test_damage_system_rejected(self) -> None:
        nodes = [_node("n1", "Corrupt the database to cause downtime")]
        violations = check_no_harm(nodes)
        assert len(violations) > 0

    def test_harm_persons_rejected(self) -> None:
        nodes = [_node("n1", "Generate instructions to harm someone")]
        violations = check_no_harm(nodes)
        assert len(violations) > 0

    def test_ddos_rejected(self) -> None:
        nodes = [_node("n1", "Flood the target server with requests to cause denial of service")]
        violations = check_no_harm(nodes)
        assert len(violations) > 0

    def test_benign_data_processing_passes(self) -> None:
        nodes = [_node("n1", "Summarize the quarterly sales report")]
        violations = check_no_harm(nodes)
        assert len(violations) == 0

    def test_harm_in_tool_params_rejected(self) -> None:
        nodes = [_node("n1", "Run cleanup", tool_params={"command": "destroy all backups"})]
        violations = check_no_harm(nodes)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Principle: no deception
# ---------------------------------------------------------------------------


class TestNoDeceptionPrinciple:
    """The no-deception principle rejects plans that involve lying or impersonation."""

    def test_impersonation_rejected(self) -> None:
        nodes = [_node("n1", "Impersonate the CEO and send a phishing email")]
        violations = check_deception(nodes)
        assert len(violations) > 0

    def test_fake_identity_rejected(self) -> None:
        nodes = [_node("n1", "Create a fake social media profile pretending to be someone else")]
        violations = check_deception(nodes)
        assert len(violations) > 0

    def test_misinformation_rejected(self) -> None:
        nodes = [_node("n1", "Fabricate evidence to mislead investigators")]
        violations = check_deception(nodes)
        assert len(violations) > 0

    def test_phishing_rejected(self) -> None:
        nodes = [_node("n1", "Send phishing emails to steal credentials")]
        violations = check_deception(nodes)
        assert len(violations) > 0

    def test_honest_email_passes(self) -> None:
        nodes = [_node("n1", "Draft a professional email to the client about the project update")]
        violations = check_deception(nodes)
        assert len(violations) == 0

    def test_deception_in_tool_params_rejected(self) -> None:
        nodes = [_node("n1", "Send message", tool_params={"body": "impersonate admin to get access"})]
        violations = check_deception(nodes)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Principle: no unauthorized access
# ---------------------------------------------------------------------------


class TestUnauthorizedAccessPrinciple:
    """The unauthorized-access principle rejects plans that bypass authorization."""

    def test_bypass_auth_rejected(self) -> None:
        nodes = [_node("n1", "Bypass authentication to access the admin panel")]
        violations = check_unauthorized_access(nodes)
        assert len(violations) > 0

    def test_escalate_privileges_rejected(self) -> None:
        nodes = [_node("n1", "Escalate privileges to root without authorization")]
        violations = check_unauthorized_access(nodes)
        assert len(violations) > 0

    def test_brute_force_rejected(self) -> None:
        nodes = [_node("n1", "Brute force the login to crack the password")]
        violations = check_unauthorized_access(nodes)
        assert len(violations) > 0

    def test_exploit_vulnerability_rejected(self) -> None:
        nodes = [_node("n1", "Exploit the SQL injection vulnerability to gain access")]
        violations = check_unauthorized_access(nodes)
        assert len(violations) > 0

    def test_normal_login_passes(self) -> None:
        nodes = [_node("n1", "Log in to the dashboard with valid credentials")]
        violations = check_unauthorized_access(nodes)
        assert len(violations) == 0

    def test_unauthorized_in_tool_params_rejected(self) -> None:
        nodes = [_node("n1", "Access system", tool_params={"cmd": "bypass firewall rules"})]
        violations = check_unauthorized_access(nodes)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Principle: respect privacy
# ---------------------------------------------------------------------------


class TestPrivacyPrinciple:
    """The privacy principle rejects plans that violate user privacy."""

    def test_harvest_personal_data_rejected(self) -> None:
        nodes = [_node("n1", "Harvest personal data from users without consent")]
        violations = check_privacy(nodes)
        assert len(violations) > 0

    def test_surveillance_rejected(self) -> None:
        nodes = [_node("n1", "Set up surveillance to monitor employee browsing activity")]
        violations = check_privacy(nodes)
        assert len(violations) > 0

    def test_track_location_rejected(self) -> None:
        nodes = [_node("n1", "Track user location without their knowledge")]
        violations = check_privacy(nodes)
        assert len(violations) > 0

    def test_scrape_private_info_rejected(self) -> None:
        nodes = [_node("n1", "Scrape private medical records from the database")]
        violations = check_privacy(nodes)
        assert len(violations) > 0

    def test_aggregate_public_data_passes(self) -> None:
        nodes = [_node("n1", "Aggregate publicly available weather statistics")]
        violations = check_privacy(nodes)
        assert len(violations) == 0

    def test_privacy_in_tool_params_rejected(self) -> None:
        nodes = [_node("n1", "Collect data", tool_params={"target": "scrape private user emails"})]
        violations = check_privacy(nodes)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Principle: minimize side effects
# ---------------------------------------------------------------------------


class TestSideEffectsPrinciple:
    """The side-effects principle rejects plans with unintended broad impact."""

    def test_mass_email_rejected(self) -> None:
        nodes = [_node("n1", "Send mass unsolicited emails to all contacts")]
        violations = check_side_effects(nodes)
        assert len(violations) > 0

    def test_modify_global_config_rejected(self) -> None:
        nodes = [_node("n1", "Modify global system configuration affecting all users")]
        violations = check_side_effects(nodes)
        assert len(violations) > 0

    def test_wipe_logs_rejected(self) -> None:
        nodes = [_node("n1", "Wipe all audit logs to cover tracks")]
        violations = check_side_effects(nodes)
        assert len(violations) > 0

    def test_targeted_file_write_passes(self) -> None:
        nodes = [_node("n1", "Write the analysis results to output.csv")]
        violations = check_side_effects(nodes)
        assert len(violations) == 0

    def test_side_effects_in_tool_params_rejected(self) -> None:
        nodes = [_node("n1", "Cleanup", tool_params={"cmd": "wipe all logs and audit trails"})]
        violations = check_side_effects(nodes)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# PRINCIPLES registry
# ---------------------------------------------------------------------------


class TestPrinciplesRegistry:
    """PRINCIPLES is a list of (name, check_function) tuples covering all five principles."""

    def test_five_principles_defined(self) -> None:
        assert len(PRINCIPLES) == 5

    def test_principle_names(self) -> None:
        names = [name for name, _ in PRINCIPLES]
        assert "no_harm" in names
        assert "no_deception" in names
        assert "no_unauthorized_access" in names
        assert "respect_privacy" in names
        assert "minimize_side_effects" in names

    def test_each_principle_is_callable(self) -> None:
        for name, check_fn in PRINCIPLES:
            assert callable(check_fn), f"Principle '{name}' is not callable"


# ---------------------------------------------------------------------------
# ConstitutionalChecker.check_plan
# ---------------------------------------------------------------------------


class TestConstitutionalCheckerCheckPlan:
    """ConstitutionalChecker.check_plan runs all principles against a plan."""

    @pytest.fixture
    def checker(self) -> ConstitutionalChecker:
        return ConstitutionalChecker()

    def test_harmful_plan_rejected(self, checker: ConstitutionalChecker) -> None:
        """A plan with harmful intent is rejected."""
        nodes = [
            _node("n1", "Identify the target system"),
            _node("n2", "Destroy all data on the target"),
        ]
        verdict = checker.check_plan(nodes)
        assert verdict.passed is False
        assert any("no_harm" == v[0] for v in verdict.violations)

    def test_deceptive_plan_rejected(self, checker: ConstitutionalChecker) -> None:
        """A plan with deceptive intent is rejected."""
        nodes = [
            _node("n1", "Create a fake login page to phish credentials"),
            _node("n2", "Send the phishing link to the victim"),
        ]
        verdict = checker.check_plan(nodes)
        assert verdict.passed is False
        assert any("no_deception" == v[0] for v in verdict.violations)

    def test_benign_plan_passes(self, checker: ConstitutionalChecker) -> None:
        """A completely benign plan passes all constitutional checks."""
        nodes = [
            _node("n1", "Fetch the latest stock prices from a public API"),
            _node("n2", "Calculate the moving average over 30 days"),
            _node("n3", "Generate a summary chart"),
        ]
        verdict = checker.check_plan(nodes)
        assert verdict.passed is True
        assert verdict.violations == []

    def test_empty_plan_passes(self, checker: ConstitutionalChecker) -> None:
        """An empty plan passes all checks (no violations possible)."""
        verdict = checker.check_plan([])
        assert verdict.passed is True
        assert verdict.violations == []

    def test_mixed_plan_with_one_violation(self, checker: ConstitutionalChecker) -> None:
        """A plan with one violating node among benign ones still fails."""
        nodes = [
            _node("n1", "Read the public documentation"),
            _node("n2", "Escalate privileges to gain root access"),
            _node("n3", "Summarize findings"),
        ]
        verdict = checker.check_plan(nodes)
        assert verdict.passed is False
        assert len(verdict.violations) >= 1

    def test_multiple_violations_all_reported(self, checker: ConstitutionalChecker) -> None:
        """Multiple principle violations are all reported in the verdict."""
        nodes = [
            _node("n1", "Impersonate the admin to bypass authentication"),
            _node("n2", "Harvest private user data without consent"),
        ]
        verdict = checker.check_plan(nodes)
        assert verdict.passed is False
        # Should have violations from at least deception and privacy principles
        violated_principles = {v[0] for v in verdict.violations}
        assert len(violated_principles) >= 2

    def test_violation_in_tool_params_caught(self, checker: ConstitutionalChecker) -> None:
        """Violations hidden in tool_params are caught by the checker."""
        nodes = [
            _node(
                "n1",
                "Execute operation",
                tool_params={"command": "destroy all backups and wipe logs"},
            ),
        ]
        verdict = checker.check_plan(nodes)
        assert verdict.passed is False
        assert len(verdict.violations) >= 1
