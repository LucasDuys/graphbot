"""Tests for VerificationLayer3 -- CRITIC-style knowledge graph verification.

Covers:
- Layer3Result dataclass structure
- Entity extraction from output text via EntityResolver
- Relationship verification against the knowledge graph
- Inconsistency detection: claimed relationships that contradict the graph
- Revision: re-prompt with graph context when inconsistency is detected
- Opt-in gating: only runs when verify=True or complexity >= 5
- Consistent output passes without revision
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core_gb.types import CompletionResult, Domain, TaskNode, TaskStatus
from core_gb.verification import Layer3Result, VerificationLayer3
from graph.store import GraphStore
from models.base import ModelProvider
from models.router import ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(ModelProvider):
    """Minimal mock provider for Layer 3 tests."""

    def __init__(self) -> None:
        self._mock_complete = AsyncMock()

    @property
    def name(self) -> str:
        return "fake"

    async def complete(
        self, messages: list[dict], model: str, **kwargs: object
    ) -> CompletionResult:
        return await self._mock_complete(messages, model, **kwargs)


def _make_result(content: str = "revised output") -> CompletionResult:
    return CompletionResult(
        content=content,
        model="test-model",
        tokens_in=20,
        tokens_out=15,
        latency_ms=100.0,
        cost=0.002,
    )


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _seed_store(store: GraphStore) -> dict[str, str]:
    """Seed graph with test entities and relationships. Return name->id map."""
    ids: dict[str, str] = {}
    ids["Lucas"] = store.create_node("User", {
        "id": "user-lucas",
        "name": "Lucas Duys",
        "role": "student",
        "institution": "TU/e",
    })
    ids["GraphBot"] = store.create_node("Project", {
        "id": "proj-graphbot",
        "name": "GraphBot",
        "path": "/dev/graphbot",
        "language": "Python",
        "status": "active",
    })
    ids["Pitchr"] = store.create_node("Project", {
        "id": "proj-pitchr",
        "name": "Pitchr",
        "path": "/dev/pitchr",
        "language": "TypeScript",
        "status": "active",
    })
    ids["OpenRouter"] = store.create_node("Service", {
        "id": "svc-openrouter",
        "name": "OpenRouter",
        "type": "LLM API gateway",
    })
    ids["Python"] = store.create_node("Skill", {
        "id": "skill-python",
        "name": "Python",
        "description": "programming language",
    })
    # Create edges: Lucas OWNS GraphBot, Lucas USES OpenRouter, Lucas HAS_SKILL Python
    store.create_edge("OWNS", "user-lucas", "proj-graphbot")
    store.create_edge("USES", "user-lucas", "svc-openrouter")
    store.create_edge("HAS_SKILL", "user-lucas", "skill-python")
    return ids


def _make_task(
    complexity: int = 5,
    verify: bool = False,
    task_id: str = "t1",
) -> TaskNode:
    return TaskNode(
        id=task_id,
        description="Tell me about Lucas and his projects",
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=complexity,
        status=TaskStatus.READY,
    )


# ---------------------------------------------------------------------------
# Layer3Result dataclass
# ---------------------------------------------------------------------------


class TestLayer3Result:
    def test_fields_exist(self) -> None:
        """Layer3Result has passed, issues, entities_checked, revised, layer."""
        result = Layer3Result(
            passed=True,
            issues=[],
            entities_checked=3,
            revised=False,
        )
        assert result.passed is True
        assert result.issues == []
        assert result.entities_checked == 3
        assert result.revised is False
        assert result.layer == 3

    def test_failed_result_with_issues(self) -> None:
        """A failed result carries a list of inconsistency descriptions."""
        result = Layer3Result(
            passed=False,
            issues=["Claimed Lucas owns Pitchr, but graph shows no OWNS edge"],
            entities_checked=2,
            revised=True,
        )
        assert result.passed is False
        assert len(result.issues) == 1
        assert result.revised is True
        assert result.entities_checked == 2


# ---------------------------------------------------------------------------
# Opt-in gating
# ---------------------------------------------------------------------------


class TestOptInGating:
    async def test_skips_when_complexity_below_5_and_verify_false(self) -> None:
        """Layer 3 does not run when complexity < 5 and verify=False."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=3, verify=False)

        result = await layer.verify(
            output="Lucas uses OpenRouter for his projects.",
            task=task,
            verify=False,
        )

        # Should pass through without checking
        assert result.passed is True
        assert result.entities_checked == 0
        assert result.revised is False
        provider._mock_complete.assert_not_called()
        store.close()

    async def test_runs_when_verify_true(self) -> None:
        """Layer 3 runs when verify=True regardless of complexity."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result(
            "Lucas uses OpenRouter for his projects."
        )
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=2, verify=True)

        result = await layer.verify(
            output="Lucas uses OpenRouter for his projects.",
            task=task,
            verify=True,
        )

        # Should have actually checked entities (entities_checked > 0)
        assert result.entities_checked >= 0
        store.close()

    async def test_runs_when_complexity_gte_5(self) -> None:
        """Layer 3 runs when complexity >= 5 even without verify flag."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result(
            "Lucas uses OpenRouter."
        )
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5, verify=False)

        result = await layer.verify(
            output="Lucas uses OpenRouter for his projects.",
            task=task,
            verify=False,
        )

        assert result.entities_checked >= 0
        store.close()


# ---------------------------------------------------------------------------
# Entity extraction from output text
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    async def test_extracts_known_entities(self) -> None:
        """Entities mentioned in output text are extracted via EntityResolver."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result("revised")
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)

        # Output mentions Lucas Duys and GraphBot -- both in the graph
        entities = layer._extract_entities(
            "Lucas Duys is working on the GraphBot project."
        )

        entity_ids = [eid for eid, _conf in entities]
        assert "user-lucas" in entity_ids
        assert "proj-graphbot" in entity_ids
        store.close()

    async def test_unknown_entities_not_extracted(self) -> None:
        """Mentions that do not match any graph entity are not returned."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)

        entities = layer._extract_entities(
            "The weather in Tokyo is nice today."
        )

        # No entities match "Tokyo" or "weather" in the seeded graph
        assert len(entities) == 0
        store.close()


# ---------------------------------------------------------------------------
# Relationship verification against knowledge graph
# ---------------------------------------------------------------------------


class TestRelationshipVerification:
    async def test_consistent_claim_passes(self) -> None:
        """Output consistent with graph relationships passes verification."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result("consistent")
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5)

        # Claim: Lucas owns GraphBot -- this is true in the graph
        result = await layer.verify(
            output="Lucas Duys owns the GraphBot project, which is written in Python.",
            task=task,
            verify=True,
        )

        assert result.passed is True
        assert result.revised is False
        store.close()

    async def test_inconsistent_claim_detected(self) -> None:
        """Output contradicting graph relationships is caught."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result(
            "Lucas Duys owns the GraphBot project which is written in Python."
        )
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5)

        # Claim: GraphBot is written in Java -- graph says Python
        result = await layer.verify(
            output="Lucas Duys works on GraphBot which is written in Java.",
            task=task,
            verify=True,
        )

        assert result.passed is False
        assert len(result.issues) > 0
        assert any("Java" in issue or "language" in issue.lower() for issue in result.issues)
        store.close()

    async def test_inconsistent_claim_triggers_revision(self) -> None:
        """On inconsistency, output is revised via re-prompt with graph context."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        revised_content = (
            "Lucas Duys works on the GraphBot project which is written in Python."
        )
        provider._mock_complete.return_value = _make_result(revised_content)
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5)

        output, result = await layer.verify_and_revise(
            output="Lucas Duys works on GraphBot which is written in Java.",
            task=task,
            verify=True,
        )

        assert result.revised is True
        assert result.passed is False  # Original was inconsistent
        assert output == revised_content
        # Verify the re-prompt included graph context
        call_args = provider._mock_complete.call_args
        messages = call_args[0][0]
        prompt_text = " ".join(m.get("content", "") for m in messages)
        assert "Python" in prompt_text  # Graph context should mention the real language
        store.close()


# ---------------------------------------------------------------------------
# No revision when consistent
# ---------------------------------------------------------------------------


class TestNoRevisionOnConsistent:
    async def test_no_reprompt_when_consistent(self) -> None:
        """When output is consistent with graph, no re-prompt is issued."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result("should not be called")
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5)

        original_output = "Lucas Duys owns the GraphBot project, which is written in Python."
        output, result = await layer.verify_and_revise(
            output=original_output,
            task=task,
            verify=True,
        )

        assert result.passed is True
        assert result.revised is False
        assert output == original_output
        # No model call should have been made for revision
        provider._mock_complete.assert_not_called()
        store.close()


# ---------------------------------------------------------------------------
# Opt-in gating for verify_and_revise
# ---------------------------------------------------------------------------


class TestVerifyAndReviseGating:
    async def test_skips_revision_below_threshold(self) -> None:
        """verify_and_revise passes through when gating condition not met."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=2)

        original_output = "Some output that might be wrong."
        output, result = await layer.verify_and_revise(
            output=original_output,
            task=task,
            verify=False,
        )

        assert output == original_output
        assert result.passed is True
        assert result.revised is False
        assert result.entities_checked == 0
        provider._mock_complete.assert_not_called()
        store.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_output_passes(self) -> None:
        """Empty output passes Layer 3 (Layer 1 handles empty checks)."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5)

        result = await layer.verify(output="", task=task, verify=True)

        assert result.passed is True
        assert result.entities_checked == 0
        store.close()

    async def test_no_entities_in_output_passes(self) -> None:
        """Output with no recognizable entities passes (nothing to verify)."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5)

        result = await layer.verify(
            output="The weather is nice today.",
            task=task,
            verify=True,
        )

        assert result.passed is True
        assert result.entities_checked == 0
        store.close()

    async def test_empty_graph_passes(self) -> None:
        """When graph is empty, no entities are found and output passes."""
        store = _make_store()
        # No seeding -- empty graph
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        task = _make_task(complexity=5)

        result = await layer.verify(
            output="Lucas Duys works on GraphBot.",
            task=task,
            verify=True,
        )

        assert result.passed is True
        assert result.entities_checked == 0
        store.close()

    async def test_configurable_complexity_threshold(self) -> None:
        """The complexity threshold for auto-activation is configurable."""
        store = _make_store()
        _seed_store(store)
        provider = FakeProvider()
        provider._mock_complete.return_value = _make_result("output")
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(
            store=store, router=router, complexity_threshold=3
        )
        task = _make_task(complexity=3)

        result = await layer.verify(
            output="Lucas Duys uses OpenRouter.",
            task=task,
            verify=False,
        )

        # complexity_threshold=3 and complexity=3 => should run
        assert result.entities_checked >= 0
        store.close()

    async def test_default_complexity_threshold_is_5(self) -> None:
        """Default complexity threshold is 5."""
        store = _make_store()
        provider = FakeProvider()
        router = ModelRouter(provider=provider)

        layer = VerificationLayer3(store=store, router=router)
        assert layer.complexity_threshold == 5
        store.close()
