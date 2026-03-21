"""Tests for pattern storage and retrieval via the knowledge graph."""

from __future__ import annotations

from core_gb.patterns import PatternStore
from core_gb.types import Pattern
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _make_pattern(
    pid: str = "pat-001",
    trigger: str = "Weather in {slot_0} and {slot_1}",
    description: str = "Multi-city weather lookup",
    variable_slots: tuple[str, ...] = ("slot_0", "slot_1"),
    success_count: int = 5,
    avg_tokens: float = 320.0,
    avg_latency_ms: float = 850.0,
) -> Pattern:
    return Pattern(
        id=pid,
        trigger=trigger,
        description=description,
        variable_slots=variable_slots,
        success_count=success_count,
        avg_tokens=avg_tokens,
        avg_latency_ms=avg_latency_ms,
    )


class TestPatternStore:
    def test_save_and_load(self) -> None:
        store = _make_store()
        ps = PatternStore(store)
        original = _make_pattern()

        ps.save(original)
        loaded = ps.load_all()

        assert len(loaded) == 1
        p = loaded[0]
        assert p.trigger == original.trigger
        assert p.description == original.description
        assert p.variable_slots == original.variable_slots
        assert p.success_count == original.success_count

    def test_save_returns_id(self) -> None:
        store = _make_store()
        ps = PatternStore(store)
        pattern = _make_pattern(pid="pat-return-id")

        result_id = ps.save(pattern)

        assert result_id == "pat-return-id"

    def test_load_empty(self) -> None:
        store = _make_store()
        ps = PatternStore(store)

        loaded = ps.load_all()

        assert loaded == []

    def test_increment_usage(self) -> None:
        store = _make_store()
        ps = PatternStore(store)
        pattern = _make_pattern(pid="pat-inc", success_count=1)
        ps.save(pattern)

        ps.increment_usage("pat-inc")

        node = store.get_node("PatternNode", "pat-inc")
        assert node is not None
        assert int(node["success_count"]) == 2

    def test_increment_updates_last_used(self) -> None:
        store = _make_store()
        ps = PatternStore(store)
        pattern = _make_pattern(pid="pat-ts")
        ps.save(pattern)

        ps.increment_usage("pat-ts")

        node = store.get_node("PatternNode", "pat-ts")
        assert node is not None
        assert node["last_used"] is not None

    def test_multiple_patterns(self) -> None:
        store = _make_store()
        ps = PatternStore(store)

        ps.save(_make_pattern(pid="pat-a", trigger="Do A"))
        ps.save(_make_pattern(pid="pat-b", trigger="Do B"))
        ps.save(_make_pattern(pid="pat-c", trigger="Do C"))

        loaded = ps.load_all()

        assert len(loaded) == 3
        triggers = {p.trigger for p in loaded}
        assert triggers == {"Do A", "Do B", "Do C"}
