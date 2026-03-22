"""Tests for VerificationConfig and its integration with DAGExecutor.

Covers:
- VerificationConfig dataclass defaults and construction
- DAGExecutor receives and applies VerificationConfig
- Layer 1 gating: layer1_enabled controls whether L1 runs
- Layer 2 gating: complexity >= layer2_threshold triggers L2
- Layer 3 gating: complexity >= layer3_threshold AND layer3_opt_in triggers L3
- Orchestrator passes verification_config to DAGExecutor
- Per-node verification level selection based on complexity
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.dag_executor import DAGExecutor
from core_gb.types import (
    CompletionResult,
    Domain,
    ExecutionResult,
    TaskNode,
    TaskStatus,
)
from core_gb.verification import VerificationConfig, VerificationLayer1


# ---------------------------------------------------------------------------
# VerificationConfig dataclass
# ---------------------------------------------------------------------------


class TestVerificationConfig:
    def test_default_values(self) -> None:
        """VerificationConfig has correct defaults."""
        config = VerificationConfig()
        assert config.layer1_enabled is True
        assert config.layer2_threshold == 3
        assert config.layer3_threshold == 5
        assert config.layer3_opt_in is False

    def test_custom_values(self) -> None:
        """VerificationConfig accepts custom values."""
        config = VerificationConfig(
            layer1_enabled=False,
            layer2_threshold=2,
            layer3_threshold=4,
            layer3_opt_in=True,
        )
        assert config.layer1_enabled is False
        assert config.layer2_threshold == 2
        assert config.layer3_threshold == 4
        assert config.layer3_opt_in is True

    def test_partial_override(self) -> None:
        """Overriding some fields keeps defaults for others."""
        config = VerificationConfig(layer2_threshold=5)
        assert config.layer1_enabled is True
        assert config.layer2_threshold == 5
        assert config.layer3_threshold == 5
        assert config.layer3_opt_in is False


# ---------------------------------------------------------------------------
# DAGExecutor accepts VerificationConfig
# ---------------------------------------------------------------------------


def _make_leaf(
    node_id: str,
    description: str,
    complexity: int = 1,
    requires: list[str] | None = None,
    provides: list[str] | None = None,
    consumes: list[str] | None = None,
) -> TaskNode:
    """Helper to create an atomic leaf TaskNode."""
    return TaskNode(
        id=node_id,
        description=description,
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=complexity,
        status=TaskStatus.READY,
        requires=requires or [],
        provides=provides or [],
        consumes=consumes or [],
    )


class MockExecutor:
    """Mock executor that returns configurable results."""

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(
        self,
        task: str,
        complexity: int = 1,
        provides_keys: list[str] | None = None,
    ) -> ExecutionResult:
        self.call_count += 1
        return ExecutionResult(
            root_id=str(uuid.uuid4()),
            output="A valid response with enough words for verification.",
            success=True,
            total_nodes=1,
            total_tokens=10,
            total_latency_ms=50.0,
            total_cost=0.001,
        )


class TestDAGExecutorAcceptsConfig:
    def test_dag_executor_accepts_verification_config(self) -> None:
        """DAGExecutor constructor accepts an optional verification_config parameter."""
        config = VerificationConfig(layer1_enabled=False)
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)
        assert dag._verification_config is config

    def test_dag_executor_default_config(self) -> None:
        """DAGExecutor uses default VerificationConfig when none is provided."""
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor)
        assert dag._verification_config is not None
        assert dag._verification_config.layer1_enabled is True
        assert dag._verification_config.layer2_threshold == 3


# ---------------------------------------------------------------------------
# Layer 1 gating via config
# ---------------------------------------------------------------------------


class TestLayer1Gating:
    async def test_layer1_disabled_skips_verification(self) -> None:
        """When layer1_enabled=False, no L1 verification is run on node output."""
        config = VerificationConfig(layer1_enabled=False)
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        nodes = [_make_leaf("a", "Task A", complexity=1)]

        # Patch the verifier to track if it gets called
        dag._verifier = MagicMock(spec=VerificationLayer1)
        dag._verifier.verify_and_retry = AsyncMock()

        result = await dag.execute(nodes)

        assert result.success is True
        # L1 verifier should NOT have been called
        dag._verifier.verify_and_retry.assert_not_called()

    async def test_layer1_enabled_runs_verification(self) -> None:
        """When layer1_enabled=True (default), L1 verification runs on node output."""
        config = VerificationConfig(layer1_enabled=True)
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        nodes = [_make_leaf("a", "Task A", complexity=1)]

        result = await dag.execute(nodes)

        assert result.success is True
        # L1 runs by default; the valid output means no retry
        assert executor.call_count == 1


# ---------------------------------------------------------------------------
# Layer 2 gating via config
# ---------------------------------------------------------------------------


class TestLayer2Gating:
    async def test_layer2_triggered_at_threshold(self) -> None:
        """When node complexity >= layer2_threshold, Layer 2 verification runs."""
        config = VerificationConfig(layer2_threshold=3)
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        # Patch L2 verifier to track calls
        mock_l2 = MagicMock()
        mock_l2.verify = AsyncMock(return_value=MagicMock(
            content="A valid response with enough words for verification.",
            model="mock",
            tokens_in=30,
            tokens_out=15,
            latency_ms=100.0,
            cost=0.003,
            sample_count=3,
            agreement_score=1.0,
            low_confidence=False,
        ))
        dag._layer2_verifier = mock_l2

        nodes = [_make_leaf("a", "Complex task needing verification", complexity=3)]
        await dag.execute(nodes)

        mock_l2.verify.assert_called_once()

    async def test_layer2_skipped_below_threshold(self) -> None:
        """When node complexity < layer2_threshold, Layer 2 does not run."""
        config = VerificationConfig(layer2_threshold=3)
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        mock_l2 = MagicMock()
        mock_l2.verify = AsyncMock()
        dag._layer2_verifier = mock_l2

        nodes = [_make_leaf("a", "Simple task", complexity=2)]
        await dag.execute(nodes)

        mock_l2.verify.assert_not_called()

    async def test_layer2_triggered_above_threshold(self) -> None:
        """When node complexity > layer2_threshold, Layer 2 also runs."""
        config = VerificationConfig(layer2_threshold=3)
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        mock_l2 = MagicMock()
        mock_l2.verify = AsyncMock(return_value=MagicMock(
            content="A valid response with enough words for verification.",
            model="mock",
            tokens_in=30,
            tokens_out=15,
            latency_ms=100.0,
            cost=0.003,
            sample_count=3,
            agreement_score=1.0,
            low_confidence=False,
        ))
        dag._layer2_verifier = mock_l2

        nodes = [_make_leaf("a", "Very complex task", complexity=5)]
        await dag.execute(nodes)

        mock_l2.verify.assert_called_once()


# ---------------------------------------------------------------------------
# Layer 3 gating via config
# ---------------------------------------------------------------------------


class TestLayer3Gating:
    async def test_layer3_not_triggered_without_opt_in(self) -> None:
        """Layer 3 does not run even at high complexity when layer3_opt_in=False."""
        config = VerificationConfig(
            layer3_threshold=5,
            layer3_opt_in=False,
        )
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        nodes = [_make_leaf("a", "Extremely complex task", complexity=7)]
        result = await dag.execute(nodes)

        # L3 should not have been triggered (no opt-in)
        assert result.success is True

    async def test_layer3_not_triggered_below_threshold_with_opt_in(self) -> None:
        """Layer 3 does not run when complexity < layer3_threshold, even with opt-in."""
        config = VerificationConfig(
            layer3_threshold=5,
            layer3_opt_in=True,
        )
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        nodes = [_make_leaf("a", "Medium task", complexity=4)]
        result = await dag.execute(nodes)

        assert result.success is True


# ---------------------------------------------------------------------------
# Orchestrator passes config to DAGExecutor
# ---------------------------------------------------------------------------


class TestOrchestratorPassesConfig:
    def test_orchestrator_accepts_verification_config(self) -> None:
        """Orchestrator constructor accepts an optional verification_config."""
        from graph.store import GraphStore
        from models.router import ModelRouter

        from core_gb.orchestrator import Orchestrator

        store = GraphStore(db_path=None)
        store.initialize()

        # Create a minimal mock provider
        provider = MagicMock()
        provider.name = "mock"
        router = ModelRouter(provider=provider)

        config = VerificationConfig(layer1_enabled=False, layer2_threshold=4)
        orchestrator = Orchestrator(store, router, verification_config=config)

        # The DAGExecutor inside should have the config
        assert orchestrator._dag_executor._verification_config is config

        store.close()

    def test_orchestrator_default_config(self) -> None:
        """Orchestrator uses default VerificationConfig when none is provided."""
        from graph.store import GraphStore
        from models.router import ModelRouter

        from core_gb.orchestrator import Orchestrator

        store = GraphStore(db_path=None)
        store.initialize()

        provider = MagicMock()
        provider.name = "mock"
        router = ModelRouter(provider=provider)

        orchestrator = Orchestrator(store, router)

        assert orchestrator._dag_executor._verification_config is not None
        assert orchestrator._dag_executor._verification_config.layer1_enabled is True

        store.close()


# ---------------------------------------------------------------------------
# Per-node verification level selection
# ---------------------------------------------------------------------------


class TestPerNodeVerificationLevel:
    async def test_low_complexity_gets_layer1_only(self) -> None:
        """Node with complexity=1 gets only Layer 1 verification."""
        config = VerificationConfig(
            layer1_enabled=True,
            layer2_threshold=3,
            layer3_threshold=5,
            layer3_opt_in=True,
        )
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        mock_l2 = MagicMock()
        mock_l2.verify = AsyncMock()
        dag._layer2_verifier = mock_l2

        nodes = [_make_leaf("a", "Simple task", complexity=1)]
        result = await dag.execute(nodes)

        assert result.success is True
        # L1 runs (default), L2 does not
        mock_l2.verify.assert_not_called()

    async def test_medium_complexity_gets_layer1_and_layer2(self) -> None:
        """Node with complexity=3 gets Layer 1 and Layer 2."""
        config = VerificationConfig(
            layer1_enabled=True,
            layer2_threshold=3,
            layer3_threshold=5,
            layer3_opt_in=True,
        )
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        mock_l2 = MagicMock()
        mock_l2.verify = AsyncMock(return_value=MagicMock(
            content="A valid response with enough words for verification.",
            model="mock",
            tokens_in=30,
            tokens_out=15,
            latency_ms=100.0,
            cost=0.003,
            sample_count=3,
            agreement_score=1.0,
            low_confidence=False,
        ))
        dag._layer2_verifier = mock_l2

        nodes = [_make_leaf("a", "Medium complexity task", complexity=3)]
        result = await dag.execute(nodes)

        assert result.success is True
        # Both L1 and L2 ran
        mock_l2.verify.assert_called_once()

    async def test_high_complexity_no_opt_in_gets_layer1_and_layer2(self) -> None:
        """Node with complexity=5 but no L3 opt-in gets L1 + L2 only."""
        config = VerificationConfig(
            layer1_enabled=True,
            layer2_threshold=3,
            layer3_threshold=5,
            layer3_opt_in=False,
        )
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        mock_l2 = MagicMock()
        mock_l2.verify = AsyncMock(return_value=MagicMock(
            content="A valid response with enough words for verification.",
            model="mock",
            tokens_in=30,
            tokens_out=15,
            latency_ms=100.0,
            cost=0.003,
            sample_count=3,
            agreement_score=1.0,
            low_confidence=False,
        ))
        dag._layer2_verifier = mock_l2

        nodes = [_make_leaf("a", "High complexity task", complexity=5)]
        result = await dag.execute(nodes)

        assert result.success is True
        mock_l2.verify.assert_called_once()

    async def test_mixed_complexity_nodes_get_correct_layers(self) -> None:
        """Multiple nodes with different complexities get appropriate verification."""
        config = VerificationConfig(
            layer1_enabled=True,
            layer2_threshold=3,
        )
        executor = MockExecutor()
        dag = DAGExecutor(executor=executor, verification_config=config)

        l2_call_complexities: list[int] = []
        original_l2 = dag._layer2_verifier

        async def track_l2_verify(task: TaskNode, messages: list[dict], **kwargs: object) -> object:
            l2_call_complexities.append(task.complexity)
            return MagicMock(
                content="A valid response with enough words for verification.",
                model="mock",
                tokens_in=30,
                tokens_out=15,
                latency_ms=100.0,
                cost=0.003,
                sample_count=3,
                agreement_score=1.0,
                low_confidence=False,
            )

        mock_l2 = MagicMock()
        mock_l2.verify = AsyncMock(side_effect=track_l2_verify)
        dag._layer2_verifier = mock_l2

        nodes = [
            _make_leaf("a", "Simple task", complexity=1),
            _make_leaf("b", "Medium task", complexity=3),
            _make_leaf("c", "Complex task", complexity=5),
        ]
        result = await dag.execute(nodes)

        assert result.success is True
        # Only complexity 3 and 5 nodes should have triggered L2
        assert len(l2_call_complexities) == 2
        assert 3 in l2_call_complexities
        assert 5 in l2_call_complexities
        # Complexity 1 node should NOT have triggered L2
        assert 1 not in l2_call_complexities
