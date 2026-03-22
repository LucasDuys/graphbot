"""Tests for PatternMatcher -- matching incoming tasks against cached patterns."""

from __future__ import annotations

from core_gb.patterns import PatternMatcher
from core_gb.types import Pattern


def _weather_pattern() -> Pattern:
    return Pattern(
        id="p1",
        trigger="Compare the weather in {slot_0}, {slot_1}, and {slot_2}",
        description="Weather comparison",
        variable_slots=("slot_0", "slot_1", "slot_2"),
        tree_template="[]",
        success_count=5,
        avg_tokens=100.0,
        avg_latency_ms=500.0,
    )


def _math_pattern() -> Pattern:
    return Pattern(
        id="p2",
        trigger="Calculate {slot_0} times {slot_1}",
        description="Multiplication",
        variable_slots=("slot_0", "slot_1"),
        tree_template="[]",
        success_count=10,
        avg_tokens=50.0,
        avg_latency_ms=200.0,
    )


class TestPatternMatcher:
    def test_exact_structural_match(self) -> None:
        matcher = PatternMatcher()
        pattern = _weather_pattern()
        result = matcher.match(
            "Compare the weather in Paris, Tokyo, and Sydney", [pattern]
        )
        assert result is not None
        matched_pattern, bindings = result
        assert matched_pattern.id == "p1"
        assert bindings["slot_0"] == "Paris"
        assert bindings["slot_1"] == "Tokyo"
        assert bindings["slot_2"] == "Sydney"

    def test_no_match_different_task(self) -> None:
        matcher = PatternMatcher()
        pattern = _weather_pattern()
        result = matcher.match("What is 2+2?", [pattern])
        assert result is None

    def test_best_match_selected(self) -> None:
        matcher = PatternMatcher()
        weather = _weather_pattern()
        math = _math_pattern()
        result = matcher.match(
            "Calculate 7 times 8", [weather, math]
        )
        assert result is not None
        matched_pattern, bindings = result
        assert matched_pattern.id == "p2"
        assert bindings["slot_0"] == "7"
        assert bindings["slot_1"] == "8"

    def test_threshold_filtering(self) -> None:
        matcher = PatternMatcher()
        pattern = _weather_pattern()
        # A somewhat similar but not matching task, with a very high threshold
        result = matcher.match(
            "Show the weather in Paris", [pattern], threshold=0.99
        )
        assert result is None

    def test_variable_bindings_extracted(self) -> None:
        matcher = PatternMatcher()
        pattern = _math_pattern()
        result = matcher.match("Calculate 123 times 456", [pattern])
        assert result is not None
        _, bindings = result
        assert isinstance(bindings, dict)
        assert bindings == {"slot_0": "123", "slot_1": "456"}

    def test_case_insensitive(self) -> None:
        matcher = PatternMatcher()
        pattern = _weather_pattern()
        result = matcher.match(
            "COMPARE THE WEATHER IN Paris, Tokyo, AND Sydney", [pattern]
        )
        assert result is not None
        matched_pattern, _ = result
        assert matched_pattern.id == "p1"

    def test_empty_patterns_list(self) -> None:
        matcher = PatternMatcher()
        result = matcher.match("anything at all", [])
        assert result is None

    def test_success_rate_weights_scoring(self) -> None:
        """A pattern with high failure_count is penalized in match scoring."""
        matcher = PatternMatcher()
        # Two patterns with identical triggers but different success rates
        reliable = Pattern(
            id="reliable",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Reliable multiplication",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=9,
            failure_count=1,
        )
        unreliable = Pattern(
            id="unreliable",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Unreliable multiplication",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=1,
            failure_count=9,
        )
        # Unreliable listed first, but reliable should win due to weighting
        result = matcher.match("Calculate 7 times 8", [unreliable, reliable])
        assert result is not None
        matched, _ = result
        assert matched.id == "reliable"

    def test_all_failures_drops_below_threshold(self) -> None:
        """A pattern with 100% failure rate drops below default threshold."""
        matcher = PatternMatcher()
        all_fail = Pattern(
            id="all-fail",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Always fails",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=0,
            failure_count=10,
        )
        # Raw score would be 1.0 (exact match), but weighted score should
        # drop below the default 0.7 threshold
        result = matcher.match("Calculate 7 times 8", [all_fail])
        assert result is None

    def test_zero_usage_pattern_not_penalized(self) -> None:
        """A brand-new pattern (0 success, 0 failure) is not penalized."""
        matcher = PatternMatcher()
        fresh = Pattern(
            id="fresh",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Brand new pattern",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=0,
            failure_count=0,
        )
        result = matcher.match("Calculate 7 times 8", [fresh])
        assert result is not None
        matched, _ = result
        assert matched.id == "fresh"
