"""Tests for PatternMatcher -- matching incoming tasks against cached patterns."""

from __future__ import annotations

import logging

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


class TestFailureDeprioritization:
    """Tests for T132 -- failure deprioritization in PatternMatcher.

    Patterns with success_rate < 20% are deprioritized:
    - Skipped when alternatives exist
    - Returned with warning when no alternatives
    - 0% success rate (all failures) forces decomposition (always skipped)
    - 0 executions treated as neutral
    """

    def test_zero_percent_success_rate_skipped_as_only_pattern(self) -> None:
        """A pattern with 0% success rate (all failures) is always skipped,
        even when it is the only matching pattern. This forces decomposition."""
        matcher = PatternMatcher()
        all_fail = Pattern(
            id="all-fail",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Always fails",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=0,
            failure_count=5,
        )
        result = matcher.match("Calculate 7 times 8", [all_fail])
        assert result is None

    def test_hundred_percent_success_rate_preferred(self) -> None:
        """A pattern with 100% success rate is always preferred over others."""
        matcher = PatternMatcher()
        perfect = Pattern(
            id="perfect",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Always succeeds",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=20,
            failure_count=0,
        )
        mediocre = Pattern(
            id="mediocre",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Sometimes fails",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
            failure_count=5,
        )
        result = matcher.match("Calculate 7 times 8", [mediocre, perfect])
        assert result is not None
        matched, _ = result
        assert matched.id == "perfect"

    def test_no_executions_treated_as_neutral(self) -> None:
        """A pattern with 0 success and 0 failure is not penalized."""
        matcher = PatternMatcher()
        new_pattern = Pattern(
            id="new",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Never used",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=0,
            failure_count=0,
        )
        result = matcher.match("Calculate 7 times 8", [new_pattern])
        assert result is not None
        matched, _ = result
        assert matched.id == "new"

    def test_low_success_rate_skipped_when_alternative_exists(self) -> None:
        """A pattern with <20% success rate is skipped when a better
        alternative is available."""
        matcher = PatternMatcher()
        low_rate = Pattern(
            id="low-rate",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Rarely succeeds",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=1,
            failure_count=9,  # 10% success rate
        )
        good_rate = Pattern(
            id="good-rate",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Usually succeeds",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=8,
            failure_count=2,  # 80% success rate
        )
        result = matcher.match("Calculate 7 times 8", [low_rate, good_rate])
        assert result is not None
        matched, _ = result
        assert matched.id == "good-rate"

    def test_low_success_rate_returned_when_only_option(self) -> None:
        """A pattern with <20% (but >0%) success rate is still returned
        when no alternatives exist, but a warning is logged."""
        matcher = PatternMatcher()
        low_rate = Pattern(
            id="low-rate",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Rarely succeeds",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=1,
            failure_count=9,  # 10% success rate
        )
        # With a low threshold so the weighted score can still pass
        result = matcher.match("Calculate 7 times 8", [low_rate], threshold=0.05)
        assert result is not None
        matched, _ = result
        assert matched.id == "low-rate"

    def test_low_success_rate_logs_warning_when_only_option(
        self, caplog: "logging.LogRecord"
    ) -> None:
        """When a low-success-rate pattern is the only option, a warning
        is logged."""
        matcher = PatternMatcher()
        low_rate = Pattern(
            id="low-rate",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Rarely succeeds",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=1,
            failure_count=9,  # 10% success rate
        )
        with caplog.at_level(logging.WARNING, logger="core_gb.patterns"):
            matcher.match("Calculate 7 times 8", [low_rate], threshold=0.05)
        assert any(
            "low success rate" in record.message.lower()
            for record in caplog.records
        )

    def test_zero_percent_skipped_even_with_low_threshold(self) -> None:
        """A 0% pattern is skipped regardless of threshold setting."""
        matcher = PatternMatcher()
        all_fail = Pattern(
            id="all-fail",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Always fails",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=0,
            failure_count=3,
        )
        result = matcher.match("Calculate 7 times 8", [all_fail], threshold=0.0)
        assert result is None

    def test_multiple_low_rate_patterns_all_skipped_for_good_one(self) -> None:
        """When multiple low-rate patterns exist alongside a good one,
        only the good one is returned."""
        matcher = PatternMatcher()
        bad_1 = Pattern(
            id="bad-1",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Bad pattern 1",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=1,
            failure_count=19,  # 5% success rate
        )
        bad_2 = Pattern(
            id="bad-2",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Bad pattern 2",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=1,
            failure_count=8,  # ~11% success rate
        )
        good = Pattern(
            id="good",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Good pattern",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=7,
            failure_count=3,  # 70% success rate
        )
        result = matcher.match("Calculate 7 times 8", [bad_1, bad_2, good])
        assert result is not None
        matched, _ = result
        assert matched.id == "good"

    def test_boundary_exactly_twenty_percent_not_skipped(self) -> None:
        """A pattern with exactly 20% success rate is NOT deprioritized
        (the threshold is strictly less than 20%)."""
        matcher = PatternMatcher()
        boundary = Pattern(
            id="boundary",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Right at boundary",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=2,
            failure_count=8,  # exactly 20% success rate
        )
        # With threshold=0.15 so the weighted score (1.0 * 0.20 = 0.20) passes
        result = matcher.match("Calculate 7 times 8", [boundary], threshold=0.15)
        assert result is not None
        matched, _ = result
        assert matched.id == "boundary"
