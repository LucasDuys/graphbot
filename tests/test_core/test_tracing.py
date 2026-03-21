"""Tests for pipeline stage tracing decorator."""

import asyncio
import time

import pytest

from core_gb.tracing import TraceCollector, TraceEntry, traced, get_collector


class TestTraceEntry:
    def test_trace_entry_fields(self) -> None:
        entry = TraceEntry(stage_name="test", duration_ms=42.0)
        assert entry.stage_name == "test"
        assert entry.duration_ms == 42.0
        assert isinstance(entry.metadata, dict)
        assert entry.timestamp > 0

    def test_trace_entry_frozen(self) -> None:
        entry = TraceEntry(stage_name="test", duration_ms=1.0)
        with pytest.raises(AttributeError):
            entry.stage_name = "other"  # type: ignore[misc]


class TestTraceCollector:
    def test_collector_record_and_entries(self) -> None:
        collector = TraceCollector()
        entry = TraceEntry(stage_name="a", duration_ms=10.0)
        collector.record(entry)
        assert len(collector.entries) == 1
        assert collector.entries[0].stage_name == "a"

    def test_collector_entries_returns_copy(self) -> None:
        collector = TraceCollector()
        collector.record(TraceEntry(stage_name="a", duration_ms=1.0))
        entries = collector.entries
        entries.clear()
        assert len(collector.entries) == 1

    def test_collector_clear(self) -> None:
        collector = TraceCollector()
        collector.record(TraceEntry(stage_name="a", duration_ms=1.0))
        collector.record(TraceEntry(stage_name="b", duration_ms=2.0))
        assert len(collector.entries) == 2
        collector.clear()
        assert len(collector.entries) == 0

    def test_collector_summary(self) -> None:
        collector = TraceCollector()
        collector.record(TraceEntry(stage_name="parse", duration_ms=10.0))
        collector.record(TraceEntry(stage_name="execute", duration_ms=50.0))
        collector.record(TraceEntry(stage_name="parse", duration_ms=15.0))
        summary = collector.summary()
        assert summary == {"parse": 25.0, "execute": 50.0}


class TestTracedDecorator:
    def setup_method(self) -> None:
        get_collector().clear()

    def test_sync_function_traced(self) -> None:
        @traced("sync_stage")
        def do_work() -> str:
            return "done"

        do_work()
        entries = get_collector().entries
        assert len(entries) == 1
        assert entries[0].stage_name == "sync_stage"

    def test_async_function_traced(self) -> None:
        @traced("async_stage")
        async def do_work() -> str:
            return "done"

        asyncio.run(do_work())
        entries = get_collector().entries
        assert len(entries) == 1
        assert entries[0].stage_name == "async_stage"

    def test_trace_entry_has_timing(self) -> None:
        @traced("timed_stage")
        def slow_work() -> None:
            time.sleep(0.01)

        slow_work()
        entries = get_collector().entries
        assert len(entries) == 1
        assert entries[0].duration_ms > 0

    def test_trace_entry_stage_name(self) -> None:
        @traced("my_custom_stage")
        def work() -> None:
            pass

        work()
        assert get_collector().entries[0].stage_name == "my_custom_stage"

    def test_traced_preserves_return_value(self) -> None:
        @traced("return_stage")
        def compute() -> int:
            return 42

        assert compute() == 42

    def test_traced_preserves_return_value_async(self) -> None:
        @traced("return_stage_async")
        async def compute() -> int:
            return 99

        assert asyncio.run(compute()) == 99

    def test_traced_records_on_exception(self) -> None:
        @traced("failing_stage")
        def broken() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            broken()

        entries = get_collector().entries
        assert len(entries) == 1
        assert entries[0].stage_name == "failing_stage"
        assert entries[0].duration_ms > 0

    def test_traced_records_on_exception_async(self) -> None:
        @traced("failing_async_stage")
        async def broken() -> None:
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(broken())

        entries = get_collector().entries
        assert len(entries) == 1
        assert entries[0].stage_name == "failing_async_stage"

    def test_traced_preserves_function_name(self) -> None:
        @traced("stage")
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"
