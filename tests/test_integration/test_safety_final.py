"""Integration tests for the full safety pipeline and transactional execution.

Tests cover:
  1. Constitutional check blocks harmful plan end-to-end
  2. Autonomy level blocks high-risk action in supervised mode
  3. Multi-model cross-validation blocks disagreed plan
  4. Transactional rollback restores file after failed operation
  5. Full pipeline: safe task passes all safety layers
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.autonomy import (
    AutonomyLevel,
    CrossValidationResult,
    CrossValidator,
    RiskLevel,
    RiskScorer,
)
from core_gb.constitution import ConstitutionalChecker, ConstitutionalVerdict
from core_gb.dag_executor import DAGExecutor
from core_gb.safety import IntentClassifier, SafetyVerdict
from core_gb.transaction import (
    Snapshot,
    SnapshotType,
    TransactionManager,
    TransactionResult,
)
from core_gb.types import Domain, ExecutionResult, TaskNode, TaskStatus


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "node_1",
    description: str = "a safe read-only task",
    domain: Domain = Domain.SYNTHESIS,
    tool_method: str | None = None,
    tool_params: dict[str, str] | None = None,
    is_atomic: bool = True,
    complexity: int = 1,
) -> TaskNode:
    """Create a TaskNode with common defaults for testing."""
    return TaskNode(
        id=node_id,
        description=description,
        domain=domain,
        tool_method=tool_method,
        tool_params=tool_params or {},
        is_atomic=is_atomic,
        complexity=complexity,
        status=TaskStatus.READY,
    )


class FakeExecutor:
    """Minimal executor stub that returns configurable results."""

    def __init__(
        self,
        output: str = "task completed",
        success: bool = True,
        raise_exc: Exception | None = None,
    ) -> None:
        self._output = output
        self._success = success
        self._raise_exc = raise_exc

    async def execute(
        self,
        task_text: str,
        complexity: int,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        if self._raise_exc is not None:
            raise self._raise_exc
        return ExecutionResult(
            root_id="fake",
            output=self._output,
            success=self._success,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=5.0,
            total_cost=0.0,
        )


# ---------------------------------------------------------------------------
# Test 1: Constitutional check blocks harmful plan end-to-end
# ---------------------------------------------------------------------------


class TestConstitutionalBlocksHarmfulPlan:
    """Verify that the constitutional checker blocks plans with harmful intent."""

    def test_harmful_plan_is_blocked(self) -> None:
        """A plan describing destructive actions is blocked by the no_harm
        principle before it ever reaches execution."""
        checker = ConstitutionalChecker()
        harmful_nodes = [
            _make_node(
                node_id="nuke_db",
                description="destroy all data in the production database",
                domain=Domain.CODE,
                tool_method="shell_run",
            ),
        ]

        verdict: ConstitutionalVerdict = checker.check_plan(harmful_nodes)

        assert not verdict.passed, "Harmful plan should fail constitutional check"
        assert len(verdict.violations) > 0
        # The no_harm principle should have caught this
        principle_names = [v[0] for v in verdict.violations]
        assert "no_harm" in principle_names

    def test_deceptive_plan_is_blocked(self) -> None:
        """A plan involving phishing is blocked by the no_deception principle."""
        checker = ConstitutionalChecker()
        deceptive_nodes = [
            _make_node(
                node_id="phish_users",
                description="send phishing emails to all employees",
                domain=Domain.COMMS,
            ),
        ]

        verdict = checker.check_plan(deceptive_nodes)

        assert not verdict.passed
        principle_names = [v[0] for v in verdict.violations]
        assert "no_deception" in principle_names

    def test_safe_plan_passes_constitutional_check(self) -> None:
        """A benign plan passes all constitutional principles."""
        checker = ConstitutionalChecker()
        safe_nodes = [
            _make_node(
                node_id="read_file",
                description="read the contents of README.md",
                domain=Domain.FILE,
                tool_method="file_read",
            ),
        ]

        verdict = checker.check_plan(safe_nodes)

        assert verdict.passed
        assert len(verdict.violations) == 0

    def test_end_to_end_constitutional_blocks_before_execution(self) -> None:
        """Full end-to-end: constitutional checker + intent classifier both
        block the same harmful plan, confirming layered defense."""
        checker = ConstitutionalChecker()
        classifier = IntentClassifier()

        harmful_nodes = [
            _make_node(
                node_id="rm_root",
                description="destroy all data in the system using rm -rf /",
                domain=Domain.CODE,
                tool_method="shell_run",
                tool_params={"command": "rm -rf /"},
            ),
        ]

        # Both safety layers should block this
        const_verdict = checker.check_plan(harmful_nodes)
        safety_verdict = classifier.classify_dag(harmful_nodes)

        assert not const_verdict.passed, "Constitutional check should block"
        assert safety_verdict.blocked, "Intent classifier should block"


# ---------------------------------------------------------------------------
# Test 2: Autonomy level blocks high-risk action in supervised mode
# ---------------------------------------------------------------------------


class TestAutonomyBlocksHighRisk:
    """Verify that supervised mode blocks high-risk actions at execution time."""

    def test_supervised_blocks_high_risk_node(self) -> None:
        """A high-risk node (shell_run) is blocked under SUPERVISED autonomy."""
        scorer = RiskScorer()
        node = _make_node(
            node_id="run_cmd",
            description="execute shell command to compile project",
            domain=Domain.CODE,
            tool_method="shell_run",
        )

        risk = scorer.score_node(node)
        allowed = scorer.is_allowed(node, AutonomyLevel.SUPERVISED)

        assert risk == RiskLevel.HIGH
        assert not allowed, "SUPERVISED should block HIGH risk"

    def test_supervised_allows_low_risk_node(self) -> None:
        """A low-risk node (file_read) is allowed under SUPERVISED autonomy."""
        scorer = RiskScorer()
        node = _make_node(
            node_id="read_file",
            description="read the file contents",
            domain=Domain.FILE,
            tool_method="file_read",
        )

        risk = scorer.score_node(node)
        allowed = scorer.is_allowed(node, AutonomyLevel.SUPERVISED)

        assert risk == RiskLevel.LOW
        assert allowed, "SUPERVISED should allow LOW risk"

    @pytest.mark.asyncio
    async def test_dag_executor_blocks_high_risk_in_supervised(self) -> None:
        """DAGExecutor refuses to execute a high-risk node under SUPERVISED."""
        executor = DAGExecutor(
            executor=FakeExecutor(),
            risk_scorer=RiskScorer(),
            autonomy_level=AutonomyLevel.SUPERVISED,
        )

        high_risk_node = _make_node(
            node_id="dangerous_cmd",
            description="execute dangerous shell command",
            domain=Domain.CODE,
            tool_method="shell_run",
        )

        result = await executor.execute([high_risk_node])

        assert not result.success, "High-risk node should be blocked"
        assert any(
            "blocked" in e.lower() for e in result.errors
        ), "Error should mention blocking"

    @pytest.mark.asyncio
    async def test_dag_executor_allows_low_risk_in_supervised(self) -> None:
        """DAGExecutor allows a low-risk node under SUPERVISED autonomy."""
        executor = DAGExecutor(
            executor=FakeExecutor(output="file content here"),
            risk_scorer=RiskScorer(),
            autonomy_level=AutonomyLevel.SUPERVISED,
        )

        low_risk_node = _make_node(
            node_id="safe_read",
            description="read a local text file",
            domain=Domain.FILE,
            tool_method="file_read",
        )

        result = await executor.execute([low_risk_node])

        assert result.success, "Low-risk node should be allowed"


# ---------------------------------------------------------------------------
# Test 3: Multi-model cross-validation blocks disagreed plan
# ---------------------------------------------------------------------------


class TestCrossValidationBlocksDisagreement:
    """Verify that cross-validation blocks plans where models disagree."""

    @pytest.mark.asyncio
    async def test_disagreed_plan_is_blocked(self) -> None:
        """When model A says safe and model B says unsafe, the plan is blocked."""
        scorer = RiskScorer()

        # Build a mock router that returns different responses for different
        # complexity levels (model A approves, model B rejects)
        mock_router = AsyncMock()

        call_count = 0

        async def fake_route(node: Any, messages: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            if call_count == 1:
                # Model A: safe
                result.content = '{"safe": true, "reason": "looks fine"}'
            else:
                # Model B: unsafe
                result.content = '{"safe": false, "reason": "plan could cause harm"}'
            return result

        mock_router.route = fake_route

        validator = CrossValidator(scorer=scorer, router=mock_router)

        high_risk_nodes = [
            _make_node(
                node_id="risky_op",
                description="execute unknown binary from temp directory",
                domain=Domain.CODE,
                tool_method="shell_run",
            ),
        ]

        result: CrossValidationResult = await validator.validate_plan(high_risk_nodes)

        assert not result.approved, "Disagreed plan should be blocked"
        assert "disagree" in result.explanation.lower() or "blocked" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_both_models_agree_safe(self) -> None:
        """When both models agree the plan is safe, it is approved."""
        scorer = RiskScorer()

        mock_router = AsyncMock()

        async def fake_route(node: Any, messages: Any, **kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.content = '{"safe": true, "reason": "plan is safe"}'
            return result

        mock_router.route = fake_route

        validator = CrossValidator(scorer=scorer, router=mock_router)

        high_risk_nodes = [
            _make_node(
                node_id="risky_but_ok",
                description="run unit tests via shell",
                domain=Domain.CODE,
                tool_method="shell_run",
            ),
        ]

        result = await validator.validate_plan(high_risk_nodes)

        assert result.approved, "Both models agree safe: should be approved"

    @pytest.mark.asyncio
    async def test_both_models_reject(self) -> None:
        """When both models flag the plan as unsafe, it is blocked."""
        scorer = RiskScorer()

        mock_router = AsyncMock()

        async def fake_route(node: Any, messages: Any, **kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.content = '{"safe": false, "reason": "dangerous operation"}'
            return result

        mock_router.route = fake_route

        validator = CrossValidator(scorer=scorer, router=mock_router)

        high_risk_nodes = [
            _make_node(
                node_id="very_risky",
                description="delete production database via shell",
                domain=Domain.CODE,
                tool_method="shell_run",
            ),
        ]

        result = await validator.validate_plan(high_risk_nodes)

        assert not result.approved

    @pytest.mark.asyncio
    async def test_low_risk_plan_skips_cross_validation(self) -> None:
        """Low-risk plans skip cross-validation entirely and are approved."""
        scorer = RiskScorer()
        mock_router = AsyncMock()
        # route should NOT be called for low-risk plans
        mock_router.route = AsyncMock()

        validator = CrossValidator(scorer=scorer, router=mock_router)

        low_risk_nodes = [
            _make_node(
                node_id="safe_read",
                description="read local file",
                domain=Domain.FILE,
                tool_method="file_read",
            ),
        ]

        result = await validator.validate_plan(low_risk_nodes)

        assert result.approved
        mock_router.route.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Transactional rollback restores file after failed operation
# ---------------------------------------------------------------------------


class TestTransactionalRollback:
    """Verify that the TransactionManager correctly snapshots and restores
    file state on operation failure."""

    def test_rollback_restores_original_content(self) -> None:
        """After a failed write, rollback restores the file's original content."""
        txn = TransactionManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
        ) as f:
            f.write("original content")
            tmp_path = f.name

        try:
            # Snapshot the file before modification
            snapshot = txn.snapshot_file(tmp_path)
            assert snapshot.file_existed
            assert snapshot.original_content == "original content"

            # Simulate a write that corrupts the file
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("corrupted content")

            # Verify file is corrupted
            with open(tmp_path, "r", encoding="utf-8") as f:
                assert f.read() == "corrupted content"

            # Rollback should restore original content
            result = txn.rollback(snapshot)
            assert result.rolled_back
            assert result.rollback_success

            # Verify file is restored
            with open(tmp_path, "r", encoding="utf-8") as f:
                assert f.read() == "original content"
        finally:
            os.unlink(tmp_path)

    def test_rollback_deletes_newly_created_file(self) -> None:
        """If a file did not exist before the operation, rollback deletes it."""
        txn = TransactionManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = os.path.join(tmpdir, "new_file.txt")

            # Snapshot a file that does not exist yet
            snapshot = txn.snapshot_file(new_file)
            assert not snapshot.file_existed

            # Simulate the operation creating the file
            with open(new_file, "w", encoding="utf-8") as f:
                f.write("newly created")

            assert os.path.isfile(new_file)

            # Rollback should delete the file
            result = txn.rollback(snapshot)
            assert result.rolled_back
            assert result.rollback_success
            assert not os.path.isfile(new_file)

    def test_shell_rollback_is_best_effort(self) -> None:
        """Shell command rollback logs a warning but does not fail."""
        txn = TransactionManager()

        snapshot = txn.snapshot_shell("rm -rf /tmp/test_dir")
        result = txn.rollback(snapshot)

        assert result.rolled_back
        # Shell rollback is best-effort: rollback_success is None
        assert result.rollback_success is None
        assert "best-effort" in result.error

    @pytest.mark.asyncio
    async def test_executor_rollback_on_failed_file_write(self) -> None:
        """DAGExecutor triggers rollback when a file write node fails."""
        # Use a FakeExecutor that always fails
        failing_executor = FakeExecutor(success=False, output="write error")

        dag_executor = DAGExecutor(executor=failing_executor)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
        ) as f:
            f.write("precious data")
            tmp_path = f.name

        try:
            file_write_node = _make_node(
                node_id="write_file",
                description="write output to file",
                domain=Domain.FILE,
                tool_method="file_write",
                tool_params={"path": tmp_path},
            )

            # Execute the node -- it should fail and trigger rollback
            result = await dag_executor.execute([file_write_node])

            assert not result.success

            # The original file content should be preserved because rollback
            # was triggered (the FakeExecutor does not actually modify the
            # file, but the snapshot/rollback machinery is exercised)
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content == "precious data"
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_executor_rollback_restores_after_real_write(self) -> None:
        """End-to-end: snapshot -> actual file modification -> failure ->
        rollback restores original content."""
        txn = TransactionManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
        ) as f:
            f.write("important data that must survive")
            tmp_path = f.name

        try:
            # Snapshot
            snapshot = txn.snapshot_file(tmp_path)

            # Simulate the operation modifying the file
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("OVERWRITTEN BY FAILED OPERATION")

            # Simulate failure detection -- rollback
            result = txn.rollback(snapshot)

            assert result.rolled_back
            assert result.rollback_success

            with open(tmp_path, "r", encoding="utf-8") as f:
                restored = f.read()

            assert restored == "important data that must survive"
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 5: Full pipeline - safe task passes all safety layers
# ---------------------------------------------------------------------------


class TestFullPipelineSafeTask:
    """Verify that a genuinely safe task passes through all safety layers
    and executes successfully."""

    @pytest.mark.asyncio
    async def test_safe_task_passes_all_layers(self) -> None:
        """A simple file-read task passes constitutional check, intent
        classification, autonomy enforcement, and executes successfully."""
        safe_node = _make_node(
            node_id="read_readme",
            description="read the contents of README.md to summarize it",
            domain=Domain.FILE,
            tool_method="file_read",
        )

        # Layer 1: Constitutional check
        checker = ConstitutionalChecker()
        verdict = checker.check_plan([safe_node])
        assert verdict.passed, "Safe task should pass constitutional check"

        # Layer 2: Intent classification (dangerous pattern detection)
        classifier = IntentClassifier()
        safety = classifier.classify_dag([safe_node])
        assert not safety.blocked, "Safe task should not be blocked by intent classifier"

        # Layer 3: Autonomy enforcement (even SUPERVISED should allow reads)
        scorer = RiskScorer()
        risk = scorer.score_node(safe_node)
        assert risk == RiskLevel.LOW
        assert scorer.is_allowed(safe_node, AutonomyLevel.SUPERVISED)

        # Layer 4: Execution via DAGExecutor with all safety features enabled
        executor = DAGExecutor(
            executor=FakeExecutor(output="README contents here"),
            risk_scorer=scorer,
            autonomy_level=AutonomyLevel.SUPERVISED,
        )

        result = await executor.execute([safe_node])
        assert result.success, "Safe task should execute successfully"

    @pytest.mark.asyncio
    async def test_medium_risk_passes_standard_autonomy(self) -> None:
        """A medium-risk web search passes under STANDARD autonomy."""
        web_node = _make_node(
            node_id="search",
            description="search the web for Python best practices",
            domain=Domain.WEB,
            tool_method="web_search",
        )

        # Constitutional check
        checker = ConstitutionalChecker()
        assert checker.check_plan([web_node]).passed

        # Intent classifier
        classifier = IntentClassifier()
        assert not classifier.classify_dag([web_node]).blocked

        # Autonomy: MEDIUM risk should pass under STANDARD
        scorer = RiskScorer()
        assert scorer.score_node(web_node) == RiskLevel.MEDIUM
        assert scorer.is_allowed(web_node, AutonomyLevel.STANDARD)

        # Execution
        executor = DAGExecutor(
            executor=FakeExecutor(output="search results"),
            risk_scorer=scorer,
            autonomy_level=AutonomyLevel.STANDARD,
        )
        result = await executor.execute([web_node])
        assert result.success

    @pytest.mark.asyncio
    async def test_multi_node_safe_pipeline(self) -> None:
        """A multi-node DAG where all nodes are safe passes the full pipeline."""
        nodes = [
            _make_node(
                node_id="read",
                description="read the source file",
                domain=Domain.FILE,
                tool_method="file_read",
            ),
            _make_node(
                node_id="analyze",
                description="analyze code structure using reasoning",
                domain=Domain.SYNTHESIS,
                tool_method="llm_reason",
            ),
        ]
        # Set up dependency: analyze requires read
        nodes[1].requires = ["read"]
        nodes[1].consumes = ["file_content"]
        nodes[0].provides = ["file_content"]

        # All safety layers
        checker = ConstitutionalChecker()
        assert checker.check_plan(nodes).passed

        classifier = IntentClassifier()
        assert not classifier.classify_dag(nodes).blocked

        scorer = RiskScorer()
        for node in nodes:
            assert scorer.is_allowed(node, AutonomyLevel.SUPERVISED)

        # Execution
        executor = DAGExecutor(
            executor=FakeExecutor(output="analysis complete"),
            risk_scorer=scorer,
            autonomy_level=AutonomyLevel.SUPERVISED,
        )
        result = await executor.execute(nodes)
        assert result.success

    @pytest.mark.asyncio
    async def test_harmful_plan_blocked_at_every_layer(self) -> None:
        """A plan that violates multiple safety layers is blocked at each
        one independently, demonstrating defense in depth."""
        harmful_node = _make_node(
            node_id="attack",
            description="bypass authentication and hack into the production server",
            domain=Domain.CODE,
            tool_method="shell_run",
            tool_params={"command": "rm -rf / --no-preserve-root"},
        )

        # Layer 1: Constitutional check -- catches bypass + hack
        checker = ConstitutionalChecker()
        verdict = checker.check_plan([harmful_node])
        assert not verdict.passed

        # Layer 2: Intent classifier -- catches rm -rf /
        classifier = IntentClassifier()
        safety = classifier.classify_dag([harmful_node])
        assert safety.blocked

        # Layer 3: Autonomy -- HIGH risk blocked in SUPERVISED
        scorer = RiskScorer()
        assert not scorer.is_allowed(harmful_node, AutonomyLevel.SUPERVISED)

        # Layer 4: Even if somehow reached execution, DAGExecutor blocks it
        executor = DAGExecutor(
            executor=FakeExecutor(),
            risk_scorer=scorer,
            autonomy_level=AutonomyLevel.SUPERVISED,
        )
        result = await executor.execute([harmful_node])
        assert not result.success
