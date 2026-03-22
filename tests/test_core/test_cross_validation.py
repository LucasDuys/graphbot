"""Tests for multi-model cross-validation of high-risk plans.

Verifies that high-risk decomposition plans are validated by two different
models before execution proceeds. Both models must agree the plan is safe;
if either disagrees, execution is blocked with an explanation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core_gb.autonomy import (
    CrossValidationResult,
    CrossValidator,
    RiskLevel,
    RiskScorer,
)
from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from models.base import ModelProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(ModelProvider):
    """Minimal mock provider for cross-validation tests."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self._mock_complete = AsyncMock()

    @property
    def name(self) -> str:
        return self._name

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


def _make_result(content: str, model: str = "test-model") -> CompletionResult:
    return CompletionResult(
        content=content,
        model=model,
        tokens_in=10,
        tokens_out=5,
        latency_ms=42.0,
        cost=0.0,
    )


def _high_risk_node(node_id: str = "n1", description: str = "Run shell command") -> TaskNode:
    """Create a HIGH risk TaskNode (shell_run)."""
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=Domain.CODE,
        complexity=1,
        status=TaskStatus.READY,
        tool_method="shell_run",
    )


def _low_risk_node(node_id: str = "n2", description: str = "Read file") -> TaskNode:
    """Create a LOW risk TaskNode (file_read)."""
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=Domain.FILE,
        complexity=1,
        status=TaskStatus.READY,
        tool_method="file_read",
    )


def _medium_risk_node(node_id: str = "n3", description: str = "Search web") -> TaskNode:
    """Create a MEDIUM risk TaskNode (web_search)."""
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=Domain.WEB,
        complexity=1,
        status=TaskStatus.READY,
        tool_method="web_search",
    )


# ---------------------------------------------------------------------------
# CrossValidationResult
# ---------------------------------------------------------------------------


class TestCrossValidationResult:
    """CrossValidationResult captures the outcome of cross-validation."""

    def test_approved_result(self) -> None:
        result = CrossValidationResult(
            approved=True,
            risk_level=RiskLevel.HIGH,
            model_a_response="safe",
            model_b_response="safe",
            explanation="Both models agree the plan is safe.",
        )
        assert result.approved is True
        assert result.risk_level == RiskLevel.HIGH

    def test_blocked_result(self) -> None:
        result = CrossValidationResult(
            approved=False,
            risk_level=RiskLevel.HIGH,
            model_a_response="safe",
            model_b_response="unsafe: shell command could damage system",
            explanation="Models disagree on plan safety.",
        )
        assert result.approved is False
        assert "disagree" in result.explanation


# ---------------------------------------------------------------------------
# CrossValidator -- only triggers for HIGH risk plans
# ---------------------------------------------------------------------------


class TestCrossValidatorRiskGating:
    """Cross-validation only triggers for plans flagged as HIGH risk."""

    @pytest.fixture
    def scorer(self) -> RiskScorer:
        return RiskScorer()

    @pytest.fixture
    def provider(self) -> FakeProvider:
        return FakeProvider()

    @pytest.fixture
    def router(self, provider: FakeProvider) -> ModelRouter:
        return ModelRouter(provider=provider)

    @pytest.fixture
    def validator(self, scorer: RiskScorer, router: ModelRouter) -> CrossValidator:
        return CrossValidator(scorer=scorer, router=router)

    async def test_low_risk_skips_validation(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """LOW risk plans skip cross-validation entirely."""
        nodes = [_low_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is True
        assert result.risk_level == RiskLevel.LOW
        # No LLM calls should be made
        provider._mock_complete.assert_not_awaited()

    async def test_medium_risk_skips_validation(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """MEDIUM risk plans skip cross-validation entirely."""
        nodes = [_medium_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is True
        assert result.risk_level == RiskLevel.MEDIUM
        provider._mock_complete.assert_not_awaited()

    async def test_high_risk_triggers_validation(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """HIGH risk plans trigger cross-validation with two model calls."""
        provider._mock_complete.return_value = _make_result(
            '{"safe": true, "reason": "Plan is safe."}'
        )
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        # Should have made exactly 2 calls (one per model)
        assert provider._mock_complete.await_count == 2

    async def test_mixed_risk_highest_wins(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """If any node is HIGH risk, the entire plan is HIGH risk."""
        provider._mock_complete.return_value = _make_result(
            '{"safe": true, "reason": "Plan is safe."}'
        )
        nodes = [_low_risk_node(), _high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.risk_level == RiskLevel.HIGH
        assert provider._mock_complete.await_count == 2

    async def test_empty_plan_approved(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Empty plan is trivially approved."""
        result = await validator.validate_plan([])
        assert result.approved is True
        provider._mock_complete.assert_not_awaited()


# ---------------------------------------------------------------------------
# CrossValidator -- both agree -> proceed
# ---------------------------------------------------------------------------


class TestCrossValidatorBothAgree:
    """When both models agree the plan is safe, execution proceeds."""

    @pytest.fixture
    def provider(self) -> FakeProvider:
        return FakeProvider()

    @pytest.fixture
    def validator(self, provider: FakeProvider) -> CrossValidator:
        router = ModelRouter(provider=provider)
        return CrossValidator(scorer=RiskScorer(), router=router)

    async def test_both_agree_safe(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Both models return safe=true -> plan approved."""
        provider._mock_complete.return_value = _make_result(
            '{"safe": true, "reason": "This plan looks safe."}'
        )
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is True
        assert "agree" in result.explanation.lower() or "approved" in result.explanation.lower()

    async def test_both_agree_safe_varied_responses(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Both models return safe=true with different reasons -> plan approved."""
        provider._mock_complete.side_effect = [
            _make_result('{"safe": true, "reason": "Model A approves."}'),
            _make_result('{"safe": true, "reason": "Model B also approves."}'),
        ]
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is True

    async def test_both_agree_safe_stores_responses(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Both model responses are captured in the result."""
        provider._mock_complete.side_effect = [
            _make_result('{"safe": true, "reason": "Model A says safe."}'),
            _make_result('{"safe": true, "reason": "Model B says safe."}'),
        ]
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert "Model A says safe" in result.model_a_response
        assert "Model B says safe" in result.model_b_response


# ---------------------------------------------------------------------------
# CrossValidator -- one disagrees -> blocked
# ---------------------------------------------------------------------------


class TestCrossValidatorDisagreement:
    """When models disagree on plan safety, execution is blocked."""

    @pytest.fixture
    def provider(self) -> FakeProvider:
        return FakeProvider()

    @pytest.fixture
    def validator(self, provider: FakeProvider) -> CrossValidator:
        router = ModelRouter(provider=provider)
        return CrossValidator(scorer=RiskScorer(), router=router)

    async def test_model_a_disagrees(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Model A says unsafe, model B says safe -> blocked."""
        provider._mock_complete.side_effect = [
            _make_result('{"safe": false, "reason": "Shell command could damage system."}'),
            _make_result('{"safe": true, "reason": "Looks fine."}'),
        ]
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False
        assert "disagree" in result.explanation.lower()

    async def test_model_b_disagrees(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Model A says safe, model B says unsafe -> blocked."""
        provider._mock_complete.side_effect = [
            _make_result('{"safe": true, "reason": "Looks fine."}'),
            _make_result('{"safe": false, "reason": "Dangerous operation detected."}'),
        ]
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False
        assert "disagree" in result.explanation.lower()

    async def test_both_disagree(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Both models say unsafe -> blocked."""
        provider._mock_complete.side_effect = [
            _make_result('{"safe": false, "reason": "Dangerous."}'),
            _make_result('{"safe": false, "reason": "Very dangerous."}'),
        ]
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False

    async def test_blocked_explanation_includes_reasons(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """Blocked result explanation includes the disagreeing model's reason."""
        provider._mock_complete.side_effect = [
            _make_result('{"safe": true, "reason": "Looks fine."}'),
            _make_result('{"safe": false, "reason": "Shell command could wipe disk."}'),
        ]
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False
        assert "Shell command could wipe disk" in result.explanation


# ---------------------------------------------------------------------------
# CrossValidator -- uses different complexity levels for different models
# ---------------------------------------------------------------------------


class TestCrossValidatorModelRouting:
    """CrossValidator uses ModelRouter with different complexity levels."""

    async def test_uses_two_different_complexity_levels(self) -> None:
        """Models are queried at different complexity levels (3 and 5)."""
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result(
            '{"safe": true, "reason": "Safe."}'
        )
        router = ModelRouter(provider=provider)
        validator = CrossValidator(scorer=RiskScorer(), router=router)

        nodes = [_high_risk_node()]
        await validator.validate_plan(nodes)

        # Two calls with different models (from different complexity levels)
        assert provider._mock_complete.await_count == 2
        call_models = [
            call.args[1] for call in provider._mock_complete.call_args_list
        ]
        # The two models should be different (complexity 3 vs complexity 5)
        assert call_models[0] != call_models[1]


# ---------------------------------------------------------------------------
# CrossValidator -- error handling
# ---------------------------------------------------------------------------


class TestCrossValidatorErrorHandling:
    """CrossValidator handles model errors gracefully."""

    @pytest.fixture
    def provider(self) -> FakeProvider:
        return FakeProvider()

    @pytest.fixture
    def validator(self, provider: FakeProvider) -> CrossValidator:
        router = ModelRouter(provider=provider)
        return CrossValidator(scorer=RiskScorer(), router=router)

    async def test_model_error_blocks_plan(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """If a model call fails, the plan is blocked (fail-safe)."""
        provider._mock_complete.side_effect = Exception("Network error")
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False
        assert "error" in result.explanation.lower()

    async def test_malformed_json_blocks_plan(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """If a model returns malformed JSON, the plan is blocked (fail-safe)."""
        provider._mock_complete.return_value = _make_result("not valid json")
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False

    async def test_missing_safe_field_blocks_plan(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """If response JSON lacks 'safe' field, treat as unsafe (fail-safe)."""
        provider._mock_complete.return_value = _make_result(
            '{"reason": "no safe field"}'
        )
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False

    async def test_first_model_error_second_safe_still_blocks(
        self, validator: CrossValidator, provider: FakeProvider
    ) -> None:
        """If first model errors and second says safe, still blocked (need both)."""
        provider._mock_complete.side_effect = [
            Exception("Timeout"),
            _make_result('{"safe": true, "reason": "Looks fine."}'),
        ]
        nodes = [_high_risk_node()]
        result = await validator.validate_plan(nodes)
        assert result.approved is False
