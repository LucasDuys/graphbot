"""Tests for ConfidenceEstimator and token budget directives."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from core_gb.confidence import ConfidenceEstimator
from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.base import ModelProvider
from models.router import CascadeConfig, CascadeResult, ModelRouter


def _make_task(complexity: int = 1, description: str = "test task") -> TaskNode:
    return TaskNode(
        id="t1",
        description=description,
        is_atomic=True,
        domain=Domain.SYSTEM,
        complexity=complexity,
        status=TaskStatus.READY,
    )


def _make_result(
    content: str = "hello",
    model: str = "some-model",
    tokens_in: int = 10,
    tokens_out: int = 5,
    logprobs: list[float] | None = None,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=42.0,
        cost=0.0,
        logprobs=logprobs,
    )


class FakeProvider(ModelProvider):
    """Minimal mock provider for confidence tests."""

    def __init__(self) -> None:
        self._mock_complete = AsyncMock()

    @property
    def name(self) -> str:
        return "fake"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


# ---------------------------------------------------------------------------
# ConfidenceEstimator.estimate() -- core scoring
# ---------------------------------------------------------------------------

class TestConfidenceEstimatorBasicScoring:
    def test_empty_content_returns_zero(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(content="", tokens_out=0)
        task = _make_task()
        assert estimator.estimate(result, task) == 0.0

    def test_whitespace_only_content_returns_zero(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(content="   \n\t  ", tokens_out=0)
        task = _make_task()
        assert estimator.estimate(result, task) == 0.0

    def test_short_content_low_tokens_scores_low(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(content="ok", tokens_out=1)
        task = _make_task()
        score = estimator.estimate(result, task)
        assert 0.0 < score < 0.5

    def test_long_content_many_tokens_scores_high(self) -> None:
        estimator = ConfidenceEstimator()
        content = "A comprehensive and detailed answer. " * 20
        result = _make_result(content=content, tokens_out=80)
        task = _make_task()
        score = estimator.estimate(result, task)
        assert score >= 0.7

    def test_score_always_between_zero_and_one(self) -> None:
        estimator = ConfidenceEstimator()
        task = _make_task()
        for length in [0, 5, 50, 200, 1000]:
            content = "x" * length
            tokens_out = length // 4
            result = _make_result(content=content, tokens_out=tokens_out)
            score = estimator.estimate(result, task)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ConfidenceEstimator -- refusal detection
# ---------------------------------------------------------------------------

class TestRefusalDetection:
    def test_refusal_phrase_i_cannot(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="I cannot help with that request.",
            tokens_out=10,
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        # Refusal should significantly reduce confidence
        assert score < 0.5

    def test_refusal_phrase_im_not_able(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="I'm not able to assist with that.",
            tokens_out=10,
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        assert score < 0.5

    def test_refusal_phrase_i_apologize(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="I apologize, but I can't do that.",
            tokens_out=10,
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        assert score < 0.5

    def test_refusal_phrase_as_an_ai(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="As an AI language model, I cannot provide that information.",
            tokens_out=15,
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        assert score < 0.5

    def test_non_refusal_content_not_penalized(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="The answer to your question is 42. Here is a detailed explanation.",
            tokens_out=25,
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        # Non-refusal content should not be penalized
        assert score >= 0.5


# ---------------------------------------------------------------------------
# ConfidenceEstimator -- structured output validity
# ---------------------------------------------------------------------------

class TestStructuredOutputValidity:
    def test_valid_json_when_provides_keys_present(self) -> None:
        estimator = ConfidenceEstimator()
        data = {
            "weather": "sunny with clear skies throughout the day",
            "temperature": "22 degrees Celsius, feels like 24",
        }
        result = _make_result(
            content=json.dumps(data),
            tokens_out=30,
        )
        task = _make_task()
        task.provides = ["weather", "temperature"]
        score = estimator.estimate(result, task)
        # Valid structured output with all keys should produce high confidence
        assert score >= 0.7

    def test_invalid_json_when_provides_keys_present(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="Not valid JSON at all, just plain text response here.",
            tokens_out=20,
        )
        task = _make_task()
        task.provides = ["weather", "temperature"]
        score_with_provides = estimator.estimate(result, task)

        # Same content without provides keys (no expectation of JSON)
        task_no_provides = _make_task()
        score_without_provides = estimator.estimate(result, task_no_provides)

        # Invalid JSON when structured output is expected should lower confidence
        assert score_with_provides < score_without_provides

    def test_json_missing_expected_keys_lowers_confidence(self) -> None:
        estimator = ConfidenceEstimator()
        data = {"weather": "sunny"}  # Missing "temperature"
        result = _make_result(
            content=json.dumps(data),
            tokens_out=20,
        )
        task = _make_task()
        task.provides = ["weather", "temperature"]
        score = estimator.estimate(result, task)

        # Compare with a result that has all keys
        full_data = {"weather": "sunny", "temperature": "22C"}
        full_result = _make_result(
            content=json.dumps(full_data),
            tokens_out=20,
        )
        full_score = estimator.estimate(full_result, task)

        assert score < full_score


# ---------------------------------------------------------------------------
# ConfidenceEstimator -- logprobs scoring
# ---------------------------------------------------------------------------

class TestLogprobsScoring:
    def test_high_logprobs_boost_confidence(self) -> None:
        estimator = ConfidenceEstimator()
        # High logprobs (close to 0.0 means high probability in log space)
        result = _make_result(
            content="The answer is 42. Here is a detailed explanation of the reasoning.",
            tokens_out=20,
            logprobs=[-0.1, -0.05, -0.2, -0.1, -0.15],
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        assert score >= 0.7

    def test_low_logprobs_reduce_confidence(self) -> None:
        estimator = ConfidenceEstimator()
        # Low logprobs (very negative means low probability)
        result = _make_result(
            content="The answer is 42. Here is a detailed explanation of the reasoning.",
            tokens_out=20,
            logprobs=[-5.0, -6.0, -4.5, -7.0, -5.5],
        )
        task = _make_task()
        score_low = estimator.estimate(result, task)

        # Same content but with high logprobs
        result_high = _make_result(
            content="The answer is 42. Here is a detailed explanation of the reasoning.",
            tokens_out=20,
            logprobs=[-0.1, -0.05, -0.2, -0.1, -0.15],
        )
        score_high = estimator.estimate(result_high, task)

        assert score_low < score_high

    def test_no_logprobs_uses_heuristic_fallback(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="The answer is 42. Here is a detailed explanation of the reasoning.",
            tokens_out=20,
            logprobs=None,
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        # Should still produce a valid score via heuristics alone
        assert 0.0 <= score <= 1.0

    def test_empty_logprobs_list_uses_fallback(self) -> None:
        estimator = ConfidenceEstimator()
        result = _make_result(
            content="A reasonable response with some content here.",
            tokens_out=15,
            logprobs=[],
        )
        task = _make_task()
        score = estimator.estimate(result, task)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Token budget directives
# ---------------------------------------------------------------------------

class TestTokenBudgetConfig:
    def test_cascade_config_has_budget_fields(self) -> None:
        config = CascadeConfig()
        assert hasattr(config, "base_tokens")
        assert hasattr(config, "complexity_multipliers")

    def test_default_base_tokens(self) -> None:
        config = CascadeConfig()
        assert config.base_tokens == 256

    def test_default_complexity_multipliers(self) -> None:
        config = CascadeConfig()
        assert config.complexity_multipliers is not None
        # Should have multipliers for complexity levels 1-5
        assert len(config.complexity_multipliers) == 5
        # Multipliers should increase with complexity
        values = list(config.complexity_multipliers.values())
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_custom_base_tokens(self) -> None:
        config = CascadeConfig(base_tokens=512)
        assert config.base_tokens == 512

    def test_custom_complexity_multipliers(self) -> None:
        custom = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0, 5: 4.0}
        config = CascadeConfig(complexity_multipliers=custom)
        assert config.complexity_multipliers == custom


class TestTokenBudgetCalculation:
    def test_budget_for_complexity_1(self) -> None:
        estimator = ConfidenceEstimator()
        config = CascadeConfig(base_tokens=256)
        budget = estimator.compute_token_budget(1, config)
        expected = int(256 * config.complexity_multipliers[1])
        assert budget == expected

    def test_budget_for_complexity_5(self) -> None:
        estimator = ConfidenceEstimator()
        config = CascadeConfig(base_tokens=256)
        budget = estimator.compute_token_budget(5, config)
        expected = int(256 * config.complexity_multipliers[5])
        assert budget == expected

    def test_budget_scales_with_base_tokens(self) -> None:
        estimator = ConfidenceEstimator()
        config_small = CascadeConfig(base_tokens=128)
        config_large = CascadeConfig(base_tokens=512)
        budget_small = estimator.compute_token_budget(3, config_small)
        budget_large = estimator.compute_token_budget(3, config_large)
        assert budget_large > budget_small

    def test_budget_clamped_for_out_of_range_complexity(self) -> None:
        estimator = ConfidenceEstimator()
        config = CascadeConfig(base_tokens=256)
        # Complexity 0 should clamp to 1
        budget_zero = estimator.compute_token_budget(0, config)
        budget_one = estimator.compute_token_budget(1, config)
        assert budget_zero == budget_one
        # Complexity 10 should clamp to 5
        budget_ten = estimator.compute_token_budget(10, config)
        budget_five = estimator.compute_token_budget(5, config)
        assert budget_ten == budget_five


# ---------------------------------------------------------------------------
# Integration: route_cascade uses ConfidenceEstimator
# ---------------------------------------------------------------------------

class TestCascadeUsesConfidenceEstimator:
    async def test_cascade_uses_estimator_for_confidence(self) -> None:
        """route_cascade should delegate to ConfidenceEstimator.estimate()."""
        provider = FakeProvider()
        good_result = _make_result(
            content="A comprehensive and detailed answer to the question. " * 5,
            model="cheap",
            tokens_out=50,
        )
        provider._mock_complete.return_value = good_result

        config = CascadeConfig(
            chain=["cheap", "expensive"],
            confidence_threshold=0.5,
        )
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=1)
        messages = [{"role": "user", "content": "hello"}]

        result = await router.route_cascade(task, messages)

        assert isinstance(result, CascadeResult)
        assert result.confidence > 0.0
        # Should accept the cheap model's result
        assert result.model == "cheap"
        assert provider._mock_complete.await_count == 1

    async def test_refusal_triggers_escalation(self) -> None:
        """A refusal response from the cheap model should escalate."""
        provider = FakeProvider()

        refusal = _make_result(
            content="I cannot help with that request. As an AI, I have limitations.",
            model="cheap",
            tokens_out=15,
        )
        good = _make_result(
            content="Here is a comprehensive answer to your question with details.",
            model="expensive",
            tokens_out=30,
        )
        provider._mock_complete.side_effect = [refusal, good]

        config = CascadeConfig(
            chain=["cheap", "expensive"],
            confidence_threshold=0.6,
        )
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=2)
        messages = [{"role": "user", "content": "complex question"}]

        result = await router.route_cascade(task, messages)

        assert result.escalated is True
        assert result.model == "expensive"
        assert provider._mock_complete.await_count == 2

    async def test_cascade_passes_max_tokens_kwarg(self) -> None:
        """route_cascade should pass max_tokens based on token budget."""
        provider = FakeProvider()
        good_result = _make_result(
            content="A comprehensive and detailed answer to the question. " * 5,
            model="cheap",
            tokens_out=50,
        )
        provider._mock_complete.return_value = good_result

        config = CascadeConfig(
            chain=["cheap", "expensive"],
            confidence_threshold=0.5,
            base_tokens=256,
        )
        router = ModelRouter(provider=provider, cascade_config=config)

        task = _make_task(complexity=3)
        messages = [{"role": "user", "content": "hello"}]

        await router.route_cascade(task, messages)

        call_kwargs = provider._mock_complete.call_args
        # max_tokens should be passed as a kwarg
        assert "max_tokens" in call_kwargs.kwargs or (
            len(call_kwargs.args) > 2 and "max_tokens" in str(call_kwargs)
        )


# ---------------------------------------------------------------------------
# Integration: _estimate_confidence delegates to ConfidenceEstimator
# ---------------------------------------------------------------------------

class TestEstimateConfidenceDelegation:
    def test_static_method_still_works(self) -> None:
        """The _estimate_confidence static method should still return a float."""
        result = _make_result(
            content="A valid response.",
            tokens_out=10,
        )
        score = ModelRouter._estimate_confidence(result)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
