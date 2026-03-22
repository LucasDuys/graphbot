"""Tests for VerificationLayer2 -- self-consistency via 3-way sampling.

Covers:
- SamplingResult dataclass structure
- Complexity threshold: only run 3-way sampling on nodes with complexity >= threshold
- 3-way parallel fan-out via asyncio.gather
- CISC (Confidence-weighted Intersample Consistency): pairwise string similarity
- Majority selection: 2-of-3 agreement selects majority answer
- Fallback: all 3 disagree -> return first with low_confidence=True
- Cost bound: exactly 3 calls, no retries within sampling
- kwargs forwarding to router.route
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from core_gb.verification import SamplingResult, VerificationLayer2
from models.base import ModelProvider
from models.router import ModelRouter


class FakeProvider(ModelProvider):
    """Minimal mock provider for verification tests."""

    def __init__(self) -> None:
        self._mock_complete = AsyncMock()

    @property
    def name(self) -> str:
        return "fake"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


def _make_task(complexity: int = 3, task_id: str = "t1") -> TaskNode:
    return TaskNode(
        id=task_id,
        description="Explain quantum entanglement",
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=complexity,
        status=TaskStatus.READY,
    )


def _make_result(
    content: str = "hello",
    model: str = "some-model",
    tokens_in: int = 10,
    tokens_out: int = 5,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=42.0,
        cost=0.001,
    )


# ---------------------------------------------------------------------------
# SamplingResult dataclass
# ---------------------------------------------------------------------------


class TestSamplingResult:
    def test_sampling_result_fields(self) -> None:
        """SamplingResult exposes content, sample_count, agreement_score, low_confidence."""
        result = SamplingResult(
            content="answer",
            model="test-model",
            tokens_in=30,
            tokens_out=15,
            latency_ms=100.0,
            cost=0.003,
            sample_count=3,
            agreement_score=0.95,
            low_confidence=False,
        )
        assert result.content == "answer"
        assert result.sample_count == 3
        assert result.agreement_score == 0.95
        assert result.low_confidence is False

    def test_sampling_result_low_confidence_flag(self) -> None:
        """low_confidence=True when all 3 samples disagree."""
        result = SamplingResult(
            content="first",
            model="test-model",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50.0,
            cost=0.001,
            sample_count=3,
            agreement_score=0.2,
            low_confidence=True,
        )
        assert result.low_confidence is True

    def test_sampling_result_is_frozen(self) -> None:
        """SamplingResult inherits frozen from CompletionResult."""
        result = SamplingResult(
            content="answer",
            model="test-model",
            tokens_in=10,
            tokens_out=5,
            latency_ms=50.0,
            cost=0.001,
            sample_count=3,
            agreement_score=0.9,
            low_confidence=False,
        )
        with pytest.raises(AttributeError):
            result.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Complexity threshold
# ---------------------------------------------------------------------------


class TestComplexityThreshold:
    async def test_skips_verification_below_threshold(self) -> None:
        """Nodes with complexity < threshold bypass 3-way sampling."""
        provider = FakeProvider()
        single_result = _make_result(content="simple answer", tokens_out=10)
        provider._mock_complete.return_value = single_result

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=2)
        messages = [{"role": "user", "content": "simple question"}]

        result = await layer.verify(task, messages)

        # Only 1 call (no 3-way sampling)
        assert provider._mock_complete.await_count == 1
        assert result.content == "simple answer"
        assert result.sample_count == 1
        assert result.low_confidence is False

    async def test_runs_verification_at_threshold(self) -> None:
        """Nodes with complexity == threshold get 3-way sampling."""
        provider = FakeProvider()
        same_result = _make_result(
            content="consistent answer about quantum physics",
            tokens_out=20,
        )
        provider._mock_complete.return_value = same_result

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "explain quantum entanglement"}]

        result = await layer.verify(task, messages)

        # 3 calls (3-way sampling)
        assert provider._mock_complete.await_count == 3
        assert result.sample_count == 3

    async def test_runs_verification_above_threshold(self) -> None:
        """Nodes with complexity > threshold also get 3-way sampling."""
        provider = FakeProvider()
        same_result = _make_result(
            content="complex answer about advanced topic",
            tokens_out=20,
        )
        provider._mock_complete.return_value = same_result

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=5)
        messages = [{"role": "user", "content": "explain general relativity"}]

        result = await layer.verify(task, messages)

        assert provider._mock_complete.await_count == 3
        assert result.sample_count == 3

    async def test_configurable_threshold(self) -> None:
        """Complexity threshold is configurable via constructor."""
        provider = FakeProvider()
        same_result = _make_result(content="answer", tokens_out=10)
        provider._mock_complete.return_value = same_result

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=1)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hello"}]

        result = await layer.verify(task, messages)

        # threshold=1 means all tasks get 3-way sampling
        assert provider._mock_complete.await_count == 3


# ---------------------------------------------------------------------------
# 3-way sampling and CISC agreement
# ---------------------------------------------------------------------------


class TestThreeWaySampling:
    async def test_all_three_agree_returns_content(self) -> None:
        """When all 3 outputs are identical, return it with high agreement."""
        provider = FakeProvider()
        same = _make_result(content="The answer is 42.", tokens_out=10)
        provider._mock_complete.return_value = same

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "What is the answer?"}]

        result = await layer.verify(task, messages)

        assert result.content == "The answer is 42."
        assert result.agreement_score == 1.0
        assert result.low_confidence is False
        assert result.sample_count == 3

    async def test_two_agree_selects_majority(self) -> None:
        """When 2 of 3 outputs agree, select the majority answer."""
        provider = FakeProvider()
        agree_result = _make_result(
            content="Quantum entanglement is a physical phenomenon.",
            tokens_out=15,
        )
        disagree_result = _make_result(
            content="I do not know the answer to that question.",
            tokens_out=15,
        )
        provider._mock_complete.side_effect = [
            agree_result, disagree_result, agree_result,
        ]

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "Explain entanglement"}]

        result = await layer.verify(task, messages)

        assert result.content == "Quantum entanglement is a physical phenomenon."
        assert result.low_confidence is False
        assert result.agreement_score > 0.5

    async def test_all_three_disagree_returns_first_with_low_confidence(self) -> None:
        """When all 3 outputs disagree, return the first with low_confidence=True."""
        provider = FakeProvider()
        r1 = _make_result(content="Alpha bravo charlie delta echo", tokens_out=10)
        r2 = _make_result(content="Foxtrot golf hotel india juliet", tokens_out=10)
        r3 = _make_result(content="Kilo lima mike november oscar", tokens_out=10)
        provider._mock_complete.side_effect = [r1, r2, r3]

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "random question"}]

        result = await layer.verify(task, messages)

        # Fallback: return first sample with low_confidence
        assert result.content == "Alpha bravo charlie delta echo"
        assert result.low_confidence is True
        assert result.sample_count == 3


# ---------------------------------------------------------------------------
# Cost bound: max 3x overhead (exactly 3 calls, no retries)
# ---------------------------------------------------------------------------


class TestCostBound:
    async def test_exactly_three_calls_no_retries(self) -> None:
        """Verification makes exactly 3 calls, never more (max 3x overhead)."""
        provider = FakeProvider()
        same = _make_result(content="consistent", tokens_out=10)
        provider._mock_complete.return_value = same

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=4)
        messages = [{"role": "user", "content": "test"}]

        await layer.verify(task, messages)

        assert provider._mock_complete.await_count == 3

    async def test_cost_aggregation(self) -> None:
        """Total cost and tokens are summed from all 3 samples."""
        provider = FakeProvider()
        r = _make_result(content="answer", tokens_in=10, tokens_out=5)
        provider._mock_complete.return_value = r

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "test"}]

        result = await layer.verify(task, messages)

        # 3 calls * (10 + 5) = 45 total tokens
        assert result.tokens_in == 30  # 3 * 10
        assert result.tokens_out == 15  # 3 * 5
        assert abs(result.cost - 0.003) < 1e-9  # 3 * 0.001


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    async def test_samples_run_in_parallel(self) -> None:
        """All 3 samples should be launched concurrently via asyncio.gather."""
        call_times: list[float] = []

        async def slow_complete(
            messages: list[dict], model: str, **kwargs: object
        ) -> CompletionResult:
            loop = asyncio.get_event_loop()
            call_times.append(loop.time())
            await asyncio.sleep(0.05)
            return _make_result(content="answer", tokens_out=10)

        provider = FakeProvider()
        provider._mock_complete.side_effect = slow_complete

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "test"}]

        await layer.verify(task, messages)

        # All 3 calls should have started at nearly the same time
        assert len(call_times) == 3
        max_spread = max(call_times) - min(call_times)
        # If sequential, spread would be >= 0.05 * 2 = 0.1
        # If parallel, spread should be < 0.02
        assert max_spread < 0.02, f"Calls not parallel: spread={max_spread:.3f}s"


# ---------------------------------------------------------------------------
# CISC (pairwise string similarity)
# ---------------------------------------------------------------------------


class TestCISC:
    def test_pairwise_similarity_identical(self) -> None:
        """Identical strings have similarity 1.0."""
        layer = VerificationLayer2.__new__(VerificationLayer2)
        score = layer._pairwise_similarity("hello world", "hello world")
        assert score == 1.0

    def test_pairwise_similarity_different(self) -> None:
        """Completely different strings have low similarity."""
        layer = VerificationLayer2.__new__(VerificationLayer2)
        score = layer._pairwise_similarity(
            "alpha bravo charlie",
            "xray yankee zulu",
        )
        assert score < 0.3

    def test_pairwise_similarity_similar(self) -> None:
        """Similar strings have high similarity."""
        layer = VerificationLayer2.__new__(VerificationLayer2)
        score = layer._pairwise_similarity(
            "The answer to the question is 42.",
            "The answer to that question is 42.",
        )
        assert score > 0.8

    def test_select_best_by_agreement_majority(self) -> None:
        """select_best returns the output with highest total pairwise agreement."""
        layer = VerificationLayer2.__new__(VerificationLayer2)
        outputs = [
            "The speed of light is approximately 3e8 m/s.",
            "The speed of light is approximately 3e8 m/s.",
            "Bananas are a good source of potassium.",
        ]
        best_idx, score = layer._select_best(outputs)
        # The first two agree, so best should be index 0 or 1
        assert best_idx in (0, 1)
        assert score > 0.5

    def test_select_best_all_identical(self) -> None:
        """When all identical, agreement score is 1.0."""
        layer = VerificationLayer2.__new__(VerificationLayer2)
        outputs = ["same", "same", "same"]
        best_idx, score = layer._select_best(outputs)
        assert score == 1.0

    def test_select_best_all_different(self) -> None:
        """When all different, agreement score is low."""
        layer = VerificationLayer2.__new__(VerificationLayer2)
        outputs = [
            "Alpha bravo charlie delta echo",
            "Foxtrot golf hotel india juliet",
            "Kilo lima mike november oscar",
        ]
        _, score = layer._select_best(outputs)
        assert score < 0.5


# ---------------------------------------------------------------------------
# Default complexity threshold
# ---------------------------------------------------------------------------


class TestDefaultThreshold:
    def test_default_threshold_is_three(self) -> None:
        """Default complexity threshold is 3."""
        provider = FakeProvider()
        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router)
        assert layer.complexity_threshold == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_output_handled(self) -> None:
        """Empty outputs from all samples do not crash."""
        provider = FakeProvider()
        empty = _make_result(content="", tokens_out=0)
        provider._mock_complete.return_value = empty

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "test"}]

        result = await layer.verify(task, messages)

        assert result.content == ""
        assert result.sample_count == 3

    async def test_kwargs_forwarded_to_router(self) -> None:
        """Extra kwargs are forwarded to the router's route call."""
        provider = FakeProvider()
        r = _make_result(content="answer", tokens_out=10)
        provider._mock_complete.return_value = r

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "test"}]

        await layer.verify(task, messages, response_format={"type": "json_object"})

        # All 3 calls should have received the extra kwarg
        for call in provider._mock_complete.call_args_list:
            assert call.kwargs.get("response_format") == {"type": "json_object"}

    async def test_single_call_returns_sampling_result(self) -> None:
        """Below-threshold single call still returns a SamplingResult."""
        provider = FakeProvider()
        r = _make_result(content="simple", tokens_out=5)
        provider._mock_complete.return_value = r

        router = ModelRouter(provider=provider)
        layer = VerificationLayer2(router=router, complexity_threshold=3)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hi"}]

        result = await layer.verify(task, messages)

        assert isinstance(result, SamplingResult)
        assert result.sample_count == 1
        assert result.agreement_score == 1.0
