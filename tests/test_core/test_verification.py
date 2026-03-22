"""Tests for the Layer 1 format/type verification engine.

Covers:
- VerificationResult dataclass structure
- Rule-based checks: non-empty output, valid JSON, refusal detection, length
- Single retry with modified prompt on failure
- Integration as post-execution hook in DAGExecutor._execute_node()
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_gb.types import (
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)
from core_gb.verification import VerificationLayer1, VerificationResult


# ---------------------------------------------------------------------------
# VerificationResult dataclass
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_fields_exist(self) -> None:
        """VerificationResult has passed, issues, layer, retry_count."""
        result = VerificationResult(
            passed=True, issues=[], layer=1, retry_count=0
        )
        assert result.passed is True
        assert result.issues == []
        assert result.layer == 1
        assert result.retry_count == 0

    def test_failed_result_with_issues(self) -> None:
        """A failed result carries a list of issue strings."""
        result = VerificationResult(
            passed=False,
            issues=["Output is empty", "Expected JSON"],
            layer=1,
            retry_count=1,
        )
        assert result.passed is False
        assert len(result.issues) == 2
        assert result.retry_count == 1


# ---------------------------------------------------------------------------
# Rule-based checks
# ---------------------------------------------------------------------------


class TestNonEmptyCheck:
    def test_empty_string_fails(self) -> None:
        """Empty output should fail verification."""
        v = VerificationLayer1()
        result = v.verify("", expects_json=False, complexity=1)
        assert result.passed is False
        assert any("empty" in issue.lower() for issue in result.issues)

    def test_whitespace_only_fails(self) -> None:
        """Whitespace-only output should fail verification."""
        v = VerificationLayer1()
        result = v.verify("   \n\t  ", expects_json=False, complexity=1)
        assert result.passed is False
        assert any("empty" in issue.lower() for issue in result.issues)

    def test_non_empty_passes(self) -> None:
        """Non-empty output passes the emptiness check."""
        v = VerificationLayer1()
        result = v.verify("Hello world, this is a valid response.", expects_json=False, complexity=1)
        assert result.passed is True


class TestValidJsonCheck:
    def test_valid_json_passes(self) -> None:
        """Valid JSON passes when expects_json is True."""
        v = VerificationLayer1()
        result = v.verify('{"key": "value"}', expects_json=True, complexity=1)
        assert result.passed is True

    def test_invalid_json_fails(self) -> None:
        """Invalid JSON fails when expects_json is True."""
        v = VerificationLayer1()
        result = v.verify("not json at all", expects_json=True, complexity=1)
        assert result.passed is False
        assert any("json" in issue.lower() for issue in result.issues)

    def test_json_not_required_skips_check(self) -> None:
        """Non-JSON output passes when expects_json is False."""
        v = VerificationLayer1()
        result = v.verify("plain text answer", expects_json=False, complexity=1)
        assert result.passed is True

    def test_json_array_passes(self) -> None:
        """JSON array is valid JSON."""
        v = VerificationLayer1()
        result = v.verify('[1, 2, 3]', expects_json=True, complexity=1)
        assert result.passed is True


class TestRefusalPhraseDetection:
    @pytest.mark.parametrize(
        "phrase",
        [
            "I cannot help with that request.",
            "I'm not able to provide that information.",
            "As an AI language model, I don't have access to that.",
            "As an AI, I cannot browse the internet.",
            "I'm sorry, but I cannot assist with that.",
        ],
    )
    def test_refusal_phrases_detected(self, phrase: str) -> None:
        """Known refusal phrases trigger verification failure."""
        v = VerificationLayer1()
        result = v.verify(phrase, expects_json=False, complexity=1)
        assert result.passed is False
        assert any("refusal" in issue.lower() for issue in result.issues)

    def test_legitimate_use_of_cannot_passes(self) -> None:
        """Text containing 'cannot' in legitimate context should pass.

        Only standalone refusal patterns at the start of a response
        trigger failure, not arbitrary uses of the word 'cannot' in
        longer legitimate content.
        """
        v = VerificationLayer1()
        long_text = (
            "The system cannot process requests that exceed the rate limit. "
            "To avoid this, implement exponential backoff in your client code. "
            "Here is an example implementation with proper error handling."
        )
        result = v.verify(long_text, expects_json=False, complexity=1)
        assert result.passed is True


class TestReasonableLengthCheck:
    def test_short_output_for_complex_task_fails(self) -> None:
        """One or two words for a complex task (complexity >= 2) should fail."""
        v = VerificationLayer1()
        result = v.verify("Yes", expects_json=False, complexity=3)
        assert result.passed is False
        assert any("length" in issue.lower() or "short" in issue.lower() for issue in result.issues)

    def test_short_output_for_simple_task_passes(self) -> None:
        """Short output is acceptable for simple tasks (complexity 1)."""
        v = VerificationLayer1()
        result = v.verify("42", expects_json=False, complexity=1)
        assert result.passed is True

    def test_adequate_output_for_complex_task_passes(self) -> None:
        """Longer output for complex tasks passes the length check."""
        v = VerificationLayer1()
        long_answer = "The answer involves several steps. First we compute X, then Y, finally Z."
        result = v.verify(long_answer, expects_json=False, complexity=3)
        assert result.passed is True


class TestMultipleIssues:
    def test_multiple_issues_collected(self) -> None:
        """When multiple rules fail, all issues are collected."""
        v = VerificationLayer1()
        # Empty output fails both emptiness and (if expects_json) JSON check
        result = v.verify("", expects_json=True, complexity=3)
        assert result.passed is False
        assert len(result.issues) >= 1  # At least empty check


# ---------------------------------------------------------------------------
# Retry with modified prompt
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    async def test_retry_on_failure_appends_hints(self) -> None:
        """On verification failure, retry is triggered with issues appended as hints.

        verify_and_retry receives the initial output directly (no executor call).
        It only calls executor.execute once for the retry.
        """
        v = VerificationLayer1()

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            # Retry returns valid output
            return ExecutionResult(
                root_id="r1",
                output="Here is the valid answer with enough detail.",
                success=True,
                total_nodes=1,
                total_tokens=20,
                total_latency_ms=50.0,
                total_cost=0.001,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)

        node = TaskNode(
            id="test_node",
            description="Explain quantum computing",
            is_atomic=True,
            complexity=2,
        )

        result, vr = await v.verify_and_retry(
            output="I cannot help with that.",
            node=node,
            executor=executor,
            expects_json=False,
        )

        # Should have retried exactly once
        assert executor.execute.call_count == 1
        assert vr.retry_count == 1
        assert result == "Here is the valid answer with enough detail."
        # The retry call should include hints about the issues
        retry_call_args = executor.execute.call_args_list[0]
        retry_task_text = retry_call_args[0][0]
        assert "VERIFICATION HINT" in retry_task_text
        assert "refusal" in retry_task_text.lower()

    async def test_no_retry_on_success(self) -> None:
        """When verification passes, no retry is triggered."""
        v = VerificationLayer1()

        executor = MagicMock()
        executor.execute = AsyncMock()

        node = TaskNode(
            id="test_node",
            description="What is 2+2?",
            is_atomic=True,
            complexity=1,
        )

        result, vr = await v.verify_and_retry(
            output="4",
            node=node,
            executor=executor,
            expects_json=False,
        )

        # No retry should have been made
        executor.execute.assert_not_called()
        assert vr.passed is True
        assert vr.retry_count == 0
        assert result == "4"

    async def test_retry_returns_original_on_second_failure(self) -> None:
        """If retry also fails verification, return the retry output anyway."""
        v = VerificationLayer1()

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            # Always returns a refusal
            return ExecutionResult(
                root_id="r1",
                output="I cannot do that either.",
                success=True,
                total_nodes=1,
                total_tokens=10,
                total_latency_ms=50.0,
                total_cost=0.001,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)

        node = TaskNode(
            id="test_node",
            description="Explain something complex in detail",
            is_atomic=True,
            complexity=2,
        )

        result, vr = await v.verify_and_retry(
            output="I cannot help with that.",
            node=node,
            executor=executor,
            expects_json=False,
        )

        # Single retry attempted
        assert vr.retry_count == 1
        # Returns the retry output even though it also failed
        assert result == "I cannot do that either."
        assert vr.passed is False


# ---------------------------------------------------------------------------
# Integration with DAGExecutor._execute_node
# ---------------------------------------------------------------------------


class TestDAGExecutorIntegration:
    async def test_verification_runs_on_successful_node(self) -> None:
        """Verification layer runs on every successful DAG node output."""
        from core_gb.dag_executor import DAGExecutor

        call_count = 0

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            return ExecutionResult(
                root_id=str(uuid.uuid4()),
                output="A complete and valid response to the task.",
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
                description="Task A",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.READY,
            ),
        ]

        result = await dag.execute(nodes)
        assert result.success is True
        # Only one call needed (no retry since output is valid)
        assert call_count == 1

    async def test_verification_triggers_retry_on_refusal(self) -> None:
        """When a node outputs a refusal, verification triggers a retry."""
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

        nodes = [
            TaskNode(
                id="a",
                description="Explain how neural networks work",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=2,
                status=TaskStatus.READY,
            ),
        ]

        result = await dag.execute(nodes)
        assert result.success is True
        # Should have made 2 calls: original + retry
        assert call_count == 2
        assert "proper detailed response" in result.output

    async def test_verification_skipped_on_failed_node(self) -> None:
        """Verification does not run on nodes that already failed execution."""
        from core_gb.dag_executor import DAGExecutor

        call_count = 0

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
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

        nodes = [
            TaskNode(
                id="a",
                description="Task A",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                complexity=1,
                status=TaskStatus.READY,
            ),
        ]

        result = await dag.execute(nodes)
        assert result.success is False
        # Only one call -- no retry since the execution itself failed
        assert call_count == 1

    async def test_verification_retry_on_invalid_json(self) -> None:
        """When expects_json is True and output is not JSON, retry is triggered."""
        v = VerificationLayer1()

        call_count = 0

        async def mock_execute(
            task: str, complexity: int = 1, provides_keys: list[str] | None = None,
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            return ExecutionResult(
                root_id="r1",
                output='{"data_a": "proper result"}',
                success=True,
                total_nodes=1,
                total_tokens=15,
                total_latency_ms=60.0,
                total_cost=0.002,
            )

        executor = MagicMock()
        executor.execute = AsyncMock(side_effect=mock_execute)

        node = TaskNode(
            id="a",
            description="Task A",
            is_atomic=True,
            domain=Domain.SYNTHESIS,
            complexity=1,
            status=TaskStatus.READY,
            provides=["data_a"],
        )

        result, vr = await v.verify_and_retry(
            output="This is not valid JSON at all.",
            node=node,
            executor=executor,
            expects_json=True,
        )

        # Should have retried due to invalid JSON
        assert call_count == 1
        assert vr.retry_count == 1
        assert result == '{"data_a": "proper result"}'
