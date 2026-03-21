"""Pipeline stage tracing and timing infrastructure."""

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceEntry:
    """A single recorded trace from a pipeline stage execution."""

    stage_name: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class TraceCollector:
    """Collects trace entries for pipeline execution analysis."""

    def __init__(self) -> None:
        self._entries: list[TraceEntry] = []

    def record(self, entry: TraceEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[TraceEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def summary(self) -> dict[str, float]:
        """Return stage_name -> total_duration_ms mapping."""
        totals: dict[str, float] = {}
        for e in self._entries:
            totals[e.stage_name] = totals.get(e.stage_name, 0.0) + e.duration_ms
        return totals


# Global collector (can be replaced per-request in Phase 5)
_collector = TraceCollector()


def get_collector() -> TraceCollector:
    """Return the global trace collector instance."""
    return _collector


def traced(stage_name: str) -> Callable:
    """Decorator that records execution timing for a function.

    Works with both sync and async functions.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                entry = TraceEntry(stage_name=stage_name, duration_ms=duration_ms)
                _collector.record(entry)
                logger.debug("%s completed in %.1fms", stage_name, duration_ms)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                entry = TraceEntry(stage_name=stage_name, duration_ms=duration_ms)
                _collector.record(entry)
                logger.debug("%s completed in %.1fms", stage_name, duration_ms)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
