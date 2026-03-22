"""Tests for verification result logging and ExecutionResult tracking.

Covers:
- ExecutionResult.verification_results field exists and is a tuple
- VerificationResult objects appended during verification pipeline
- INFO logging on pass, WARNING logging on failure + retry
- Aggregate stats: total verifications, pass rate, retry count, per-layer breakdown
- Integration with DAGExecutor._apply_verification()
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.types import (
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)
from core_gb.verification import (
    VerificationConfig,
    VerificationLayer1,
    VerificationResult,
    aggregate_verification_stats,
)


# ---------------------------------------------------------------------------
# ExecutionResult.verification_results field
# ---------------------------------------------------------------------------


class TestExecutionResultVerificationField:
    def test_verification_results_field_exists(self) -> None:
        """ExecutionResult has a verification_results field (tuple of VerificationResult)."""
        result = ExecutionResult(
            root_id="r1",
            output="test",
            success=True,
        )
        assert hasattr(result, "verification_results")
        assert result.verification_results == ()

    def test_verification_results_is_tuple(self) -> None:
        """verification_results defaults to empty tuple and accepts VerificationResult items."""
        vr = VerificationResult(passed=True, issues=[], layer=1, retry_count=0)
        result = ExecutionResult(
            root_id="r1",
            output="test",
            success=True,
            verification_results=(vr,),
        )
        assert isinstance(result.verification_results, tuple)
        assert len(result.verification_results) == 1
        assert result.verification_results[0].passed is True

    def test_verification_results_frozen(self) -> None:
        """verification_results field is immutable on a frozen ExecutionResult."""
        result = ExecutionResult(
            root_id="r1",
            output="test",
            success=True,
        )
        with pytest.raises(AttributeError):
            result.verification_results = ()  # type: ignore[misc]

    def test_multiple_verification_results(self) -> None:
        """Multiple VerificationResult objects can be stored."""
        vr1 = VerificationResult(passed=True, issues=[], layer=1, retry_count=0)
        vr2 = VerificationResult(passed=False, issues=["too short"], layer=1, retry_count=1)
        vr3 = VerificationResult(passed=True, issues=[], layer=2, retry_count=0)
        result = ExecutionResult(
            root_id="r1",
            output="test",
            success=True,
            verification_results=(vr1, vr2, vr3),
        )
        assert len(result.verification_results) == 3


# ---------------------------------------------------------------------------
# Aggregate verification stats
# ---------------------------------------------------------------------------


class TestAggregateVerificationStats:
    def test_empty_results(self) -> None:
        """Empty verification_results produces zero stats."""
        stats = aggregate_verification_stats(())
        assert stats["total_verifications"] == 0
        assert stats["pass_rate"] == 1.0
        assert stats["total_retries"] == 0
        assert stats["per_layer"] == {}

    def test_all_passed(self) -> None:
        """All passing results yield pass_rate 1.0."""
        vrs = (
            VerificationResult(passed=True, issues=[], layer=1, retry_count=0),
            VerificationResult(passed=True, issues=[], layer=2, retry_count=0),
        )
        stats = aggregate_verification_stats(vrs)
        assert stats["total_verifications"] == 2
        assert stats["pass_rate"] == 1.0
        assert stats["total_retries"] == 0
        assert stats["per_layer"][1]["total"] == 1
        assert stats["per_layer"][1]["passed"] == 1
        assert stats["per_layer"][2]["total"] == 1
        assert stats["per_layer"][2]["passed"] == 1

    def test_mixed_results(self) -> None:
        """Mixed pass/fail results yield correct pass_rate and retry count."""
        vrs = (
            VerificationResult(passed=True, issues=[], layer=1, retry_count=0),
            VerificationResult(passed=False, issues=["empty"], layer=1, retry_count=1),
            VerificationResult(passed=True, issues=[], layer=2, retry_count=0),
            VerificationResult(passed=False, issues=["low confidence"], layer=2, retry_count=0),
        )
        stats = aggregate_verification_stats(vrs)
        assert stats["total_verifications"] == 4
        assert stats["pass_rate"] == pytest.approx(0.5)
        assert stats["total_retries"] == 1
        # Layer 1: 2 total, 1 passed, 1 retry
        assert stats["per_layer"][1]["total"] == 2
        assert stats["per_layer"][1]["passed"] == 1
        assert stats["per_layer"][1]["retries"] == 1
        # Layer 2: 2 total, 1 passed, 0 retries
        assert stats["per_layer"][2]["total"] == 2
        assert stats["per_layer"][2]["passed"] == 1
        assert stats["per_layer"][2]["retries"] == 0

    def test_all_failed(self) -> None:
        """All failing results yield pass_rate 0.0."""
        vrs = (
            VerificationResult(passed=False, issues=["empty"], layer=1, retry_count=1),
            VerificationResult(passed=False, issues=["refusal"], layer=1, retry_count=1),
        )
        stats = aggregate_verification_stats(vrs)
        assert stats["total_verifications"] == 2
        assert stats["pass_rate"] == 0.0
        assert stats["total_retries"] == 2


# ---------------------------------------------------------------------------
# Logging behavior
# ---------------------------------------------------------------------------


class TestVerificationLogging:
    async def test_passed_verification_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """A passing verification logs at INFO level."""
        from core_gb.dag_executor import DAGExecutor

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="A detailed valid response with plenty of words here.",
                success=True,
                total_nodes=1,
                total_tokens=10,
                total_latency_ms=50.0,
                total_cost=0.001,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        dag = DAGExecutor(executor=executor, max_concurrency=10)

        node = TaskNode(
            id="log_test",
            description="A valid task",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
        )

        with caplog.at_level(logging.INFO, logger="core_gb.dag_executor"):
            result = await dag.execute([node])

        assert result.success is True
        # Should log verification passed at INFO
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("verification passed" in msg.lower() or "layer 1" in msg.lower()
                    for msg in info_messages)

    async def test_failed_verification_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing verification that triggers retry logs at WARNING level."""
        from core_gb.dag_executor import DAGExecutor

        call_count = 0

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ExecutionResult(
                    root_id=str(uuid.uuid4()),
                    output="I cannot help with that request.",
                    success=True,
                    total_nodes=1,
                    total_tokens=10,
                    total_latency_ms=50.0,
                    total_cost=0.001,
                )
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="Here is a proper detailed response to the task with many words.",
                success=True,
                total_nodes=1,
                total_tokens=15,
                total_latency_ms=60.0,
                total_cost=0.002,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        dag = DAGExecutor(executor=executor, max_concurrency=10)

        node = TaskNode(
            id="warn_test",
            description="Explain something complex in detail for this test",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=2,
            status=TaskStatus.READY,
        )

        with caplog.at_level(logging.DEBUG, logger="core_gb.dag_executor"):
            result = await dag.execute([node])

        assert result.success is True
        # Should have WARNING about failed verification triggering retry
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("verification failed" in msg.lower() or "retry" in msg.lower()
                    for msg in warning_messages)


# ---------------------------------------------------------------------------
# DAGExecutor._apply_verification collects VerificationResult
# ---------------------------------------------------------------------------


class TestDAGExecutorVerificationTracking:
    async def test_verification_results_collected_on_pass(self) -> None:
        """When verification passes, result contains the VerificationResult."""
        from core_gb.dag_executor import DAGExecutor

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="A detailed valid response with plenty of words here.",
                success=True,
                total_nodes=1,
                total_tokens=10,
                total_latency_ms=50.0,
                total_cost=0.001,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        dag = DAGExecutor(executor=executor, max_concurrency=10)

        node = TaskNode(
            id="collect_pass",
            description="A valid task",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
        )

        result = await dag.execute([node])
        assert result.success is True
        assert len(result.verification_results) >= 1
        vr = result.verification_results[0]
        assert vr.layer == 1
        assert vr.passed is True
        assert vr.retry_count == 0

    async def test_verification_results_collected_on_retry(self) -> None:
        """When verification fails and retry succeeds, both results are tracked."""
        from core_gb.dag_executor import DAGExecutor

        call_count = 0

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ExecutionResult(
                    root_id=str(uuid.uuid4()),
                    output="I cannot help with that request.",
                    success=True,
                    total_nodes=1,
                    total_tokens=10,
                    total_latency_ms=50.0,
                    total_cost=0.001,
                )
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="Here is a proper detailed response to the task.",
                success=True,
                total_nodes=1,
                total_tokens=15,
                total_latency_ms=60.0,
                total_cost=0.002,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        dag = DAGExecutor(executor=executor, max_concurrency=10)

        node = TaskNode(
            id="collect_retry",
            description="Explain something complex",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=2,
            status=TaskStatus.READY,
        )

        result = await dag.execute([node])
        assert result.success is True
        # Should have at least one verification result with retry_count=1
        assert len(result.verification_results) >= 1
        vr = result.verification_results[0]
        assert vr.layer == 1
        assert vr.retry_count == 1

    async def test_verification_results_not_collected_on_failed_node(self) -> None:
        """Failed nodes skip verification, so no verification_results."""
        from core_gb.dag_executor import DAGExecutor

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="",
                success=False,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=50.0,
                total_cost=0.0,
                errors=("Provider error",),
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        dag = DAGExecutor(executor=executor, max_concurrency=10)

        node = TaskNode(
            id="no_verify",
            description="Task that fails",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
        )

        result = await dag.execute([node])
        assert result.success is False
        assert result.verification_results == ()

    async def test_multiple_nodes_aggregate_verification_results(self) -> None:
        """Multiple nodes each produce verification results, aggregated in final result."""
        from core_gb.dag_executor import DAGExecutor

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output=f"Valid response for: {task[:20]}... with enough words here.",
                success=True,
                total_nodes=1,
                total_tokens=10,
                total_latency_ms=50.0,
                total_cost=0.001,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        dag = DAGExecutor(executor=executor, max_concurrency=10)

        nodes = [
            TaskNode(
                id="a",
                description="Task A is a valid task",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.READY,
            ),
            TaskNode(
                id="b",
                description="Task B is a valid task",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.READY,
            ),
            TaskNode(
                id="c",
                description="Task C is a valid task",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.READY,
            ),
        ]

        result = await dag.execute(nodes)
        assert result.success is True
        # Each node should produce exactly one verification result
        assert len(result.verification_results) == 3
        for vr in result.verification_results:
            assert vr.layer == 1
            assert vr.passed is True

    async def test_verification_disabled_no_results(self) -> None:
        """When verification is disabled, no verification results are collected."""
        from core_gb.dag_executor import DAGExecutor

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="A valid response with many words here for testing.",
                success=True,
                total_nodes=1,
                total_tokens=10,
                total_latency_ms=50.0,
                total_cost=0.001,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)
        config = VerificationConfig(layer1_enabled=False)
        dag = DAGExecutor(
            executor=executor, max_concurrency=10, verification_config=config,
        )

        node = TaskNode(
            id="no_l1",
            description="Task without verification",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
        )

        result = await dag.execute([node])
        assert result.success is True
        assert result.verification_results == ()
