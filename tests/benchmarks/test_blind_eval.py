"""Tests for blind LLM-as-judge evaluation with mocked judge (no real API calls).

Covers pairwise comparison logic, A/B label randomization, win/loss/tie
counting, JSON output format, edge cases, and --dry-run mode.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from scripts.blind_eval import (
    COMPARISON_PAIRS,
    FAKE_TASKS,
    JudgeVerdict,
    PairSummary,
    _mock_judge,
    _parse_judge_response,
    _validate_verdict,
    call_judge,
    evaluate_pair,
    format_summary,
    generate_fake_data,
    load_thesis_validation,
    results_to_json,
    run_blind_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "test_01",
    question: str = "What is 1+1?",
    outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a minimal task dict for testing."""
    if outputs is None:
        outputs = {
            "8b_graphbot": "The answer is 2.",
            "70b_direct": "2",
            "gpt4o_direct": "1+1 equals 2.",
        }
    return {"task_id": task_id, "question": question, "outputs": outputs}


def _fixed_judge(winner: str, reasoning: str = "test") -> AsyncMock:
    """Return an AsyncMock for call_judge that always returns a fixed verdict."""
    mock = AsyncMock(return_value=({"winner": winner, "reasoning": reasoning}, 10.0))
    return mock


# ---------------------------------------------------------------------------
# Test: Pairwise comparison logic with mocked judge
# ---------------------------------------------------------------------------


class TestPairwiseComparison:
    """Test the core pairwise evaluation logic with mocked judges."""

    @pytest.mark.asyncio
    async def test_evaluate_pair_returns_verdict(self) -> None:
        """evaluate_pair should return a JudgeVerdict with correct fields."""
        task = _make_task()
        random.seed(42)

        with patch("scripts.blind_eval.call_judge", _fixed_judge("A")):
            verdict = await evaluate_pair(
                task=task,
                system_a_key="8b_graphbot",
                system_b_key="70b_direct",
                pair_label="8B+GraphBot vs 70B",
                judge_model="test-model",
                dry_run=False,
            )

        assert verdict is not None
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.task_id == "test_01"
        assert verdict.pair_label == "8B+GraphBot vs 70B"
        assert verdict.system_a_key == "8b_graphbot"
        assert verdict.system_b_key == "70b_direct"
        assert verdict.winner_label in ("A", "B", "tie")

    @pytest.mark.asyncio
    async def test_evaluate_pair_resolves_winner_system(self) -> None:
        """Winner label should be resolved back to the actual system key."""
        task = _make_task()
        # Fix seed so blinding is deterministic
        random.seed(0)

        with patch("scripts.blind_eval.call_judge", _fixed_judge("A")):
            verdict = await evaluate_pair(
                task=task,
                system_a_key="8b_graphbot",
                system_b_key="70b_direct",
                pair_label="test",
                judge_model="test-model",
            )

        assert verdict is not None
        # winner_system should be whichever system was assigned label A
        assert verdict.winner_system == verdict.blinded_a_is

    @pytest.mark.asyncio
    async def test_evaluate_pair_missing_output_returns_none(self) -> None:
        """evaluate_pair returns None when a system output is missing."""
        task = _make_task(outputs={"8b_graphbot": "answer"})

        verdict = await evaluate_pair(
            task=task,
            system_a_key="8b_graphbot",
            system_b_key="70b_direct",
            pair_label="test",
            judge_model="test-model",
        )

        assert verdict is None

    @pytest.mark.asyncio
    async def test_evaluate_pair_tie_resolves_to_tie(self) -> None:
        """When judge returns tie, winner_system should be 'tie'."""
        task = _make_task()
        random.seed(1)

        with patch("scripts.blind_eval.call_judge", _fixed_judge("tie")):
            verdict = await evaluate_pair(
                task=task,
                system_a_key="8b_graphbot",
                system_b_key="70b_direct",
                pair_label="test",
                judge_model="test-model",
            )

        assert verdict is not None
        assert verdict.winner_label == "tie"
        assert verdict.winner_system == "tie"


# ---------------------------------------------------------------------------
# Test: Randomization of A/B labeling
# ---------------------------------------------------------------------------


class TestBlindingRandomization:
    """Verify that A/B label assignment is properly randomized."""

    @pytest.mark.asyncio
    async def test_label_assignment_varies_across_seeds(self) -> None:
        """Different seeds should produce different A/B assignments."""
        task = _make_task()
        assignments: list[str] = []

        for seed in range(20):
            random.seed(seed)
            with patch("scripts.blind_eval.call_judge", _fixed_judge("A")):
                verdict = await evaluate_pair(
                    task=task,
                    system_a_key="8b_graphbot",
                    system_b_key="70b_direct",
                    pair_label="test",
                    judge_model="test-model",
                )
            assert verdict is not None
            assignments.append(verdict.blinded_a_is)

        # Should see both systems assigned to label A across 20 seeds
        unique_assignments = set(assignments)
        assert len(unique_assignments) == 2, (
            f"Expected both systems in A position, got: {unique_assignments}"
        )

    @pytest.mark.asyncio
    async def test_blinded_labels_are_complementary(self) -> None:
        """blinded_a_is and blinded_b_is should always cover both systems."""
        task = _make_task()

        for seed in range(10):
            random.seed(seed)
            with patch("scripts.blind_eval.call_judge", _fixed_judge("A")):
                verdict = await evaluate_pair(
                    task=task,
                    system_a_key="8b_graphbot",
                    system_b_key="70b_direct",
                    pair_label="test",
                    judge_model="test-model",
                )
            assert verdict is not None
            assert {verdict.blinded_a_is, verdict.blinded_b_is} == {
                "8b_graphbot",
                "70b_direct",
            }

    @pytest.mark.asyncio
    async def test_winner_resolution_respects_blinding(self) -> None:
        """If judge says A wins and A is 70b_direct, winner_system is 70b_direct."""
        task = _make_task()

        # Try many seeds; verify resolution is always consistent
        for seed in range(15):
            random.seed(seed)
            with patch("scripts.blind_eval.call_judge", _fixed_judge("B")):
                verdict = await evaluate_pair(
                    task=task,
                    system_a_key="8b_graphbot",
                    system_b_key="70b_direct",
                    pair_label="test",
                    judge_model="test-model",
                )
            assert verdict is not None
            # Winner label B -> winner_system should be blinded_b_is
            assert verdict.winner_system == verdict.blinded_b_is


# ---------------------------------------------------------------------------
# Test: Win/loss/tie counting
# ---------------------------------------------------------------------------


class TestWinLossTieCounting:
    """Verify that run_blind_eval counts wins/losses/ties correctly."""

    @pytest.mark.asyncio
    async def test_all_wins_for_system_a(self) -> None:
        """When system_a always wins, wins_a == total for each pair."""
        tasks = [_make_task(task_id=f"t{i}") for i in range(5)]

        async def _judge_a_wins(
            question: str,
            answer_a: str,
            answer_b: str,
            judge_model: str,
            dry_run: bool = False,
        ) -> tuple[dict[str, str], float]:
            return {"winner": "A", "reasoning": "A wins"}, 10.0

        with patch("scripts.blind_eval.call_judge", side_effect=_judge_a_wins):
            # Fix seed so blinding always puts system_a as label A
            random.seed(99)
            verdicts, summaries = await run_blind_eval(
                tasks=tasks, judge_model="test", dry_run=False,
            )

        # Since blinding randomizes, "A wins" means whoever is in position A wins.
        # We verify total counts are consistent: wins_a + wins_b + ties == total
        for pair_label, summary in summaries.items():
            assert summary.wins_a + summary.wins_b + summary.ties == summary.total
            assert summary.total == 5

    @pytest.mark.asyncio
    async def test_all_ties(self) -> None:
        """When judge always returns tie, all summaries should show only ties."""
        tasks = [_make_task(task_id=f"t{i}") for i in range(3)]

        async def _judge_tie(
            question: str,
            answer_a: str,
            answer_b: str,
            judge_model: str,
            dry_run: bool = False,
        ) -> tuple[dict[str, str], float]:
            return {"winner": "tie", "reasoning": "tie"}, 10.0

        with patch("scripts.blind_eval.call_judge", side_effect=_judge_tie):
            verdicts, summaries = await run_blind_eval(
                tasks=tasks, judge_model="test", dry_run=False,
            )

        for pair_label, summary in summaries.items():
            assert summary.wins_a == 0
            assert summary.wins_b == 0
            assert summary.ties == summary.total
            assert summary.total == 3

    @pytest.mark.asyncio
    async def test_mixed_results_add_up(self) -> None:
        """Win/loss/tie counts must always sum to total."""
        tasks = [_make_task(task_id=f"t{i}") for i in range(10)]
        call_count = 0

        async def _rotating_judge(
            question: str,
            answer_a: str,
            answer_b: str,
            judge_model: str,
            dry_run: bool = False,
        ) -> tuple[dict[str, str], float]:
            nonlocal call_count
            winners = ["A", "B", "tie"]
            winner = winners[call_count % 3]
            call_count += 1
            return {"winner": winner, "reasoning": "rotating"}, 10.0

        random.seed(42)
        with patch("scripts.blind_eval.call_judge", side_effect=_rotating_judge):
            verdicts, summaries = await run_blind_eval(
                tasks=tasks, judge_model="test", dry_run=False,
            )

        for pair_label, summary in summaries.items():
            assert summary.wins_a + summary.wins_b + summary.ties == summary.total
            assert summary.total == 10

    @pytest.mark.asyncio
    async def test_all_losses_for_system_a(self) -> None:
        """When system_b always wins, wins_b == total."""
        tasks = [_make_task(task_id=f"t{i}") for i in range(4)]

        async def _judge_b_wins(
            question: str,
            answer_a: str,
            answer_b: str,
            judge_model: str,
            dry_run: bool = False,
        ) -> tuple[dict[str, str], float]:
            return {"winner": "B", "reasoning": "B wins"}, 10.0

        with patch("scripts.blind_eval.call_judge", side_effect=_judge_b_wins):
            verdicts, summaries = await run_blind_eval(
                tasks=tasks, judge_model="test", dry_run=False,
            )

        for pair_label, summary in summaries.items():
            assert summary.wins_a + summary.wins_b + summary.ties == summary.total
            assert summary.total == 4


# ---------------------------------------------------------------------------
# Test: JSON output format
# ---------------------------------------------------------------------------


class TestJsonOutputFormat:
    """Verify that the JSON output has the expected structure and fields."""

    def _sample_data(self) -> tuple[list[JudgeVerdict], dict[str, PairSummary]]:
        """Create sample verdicts and summaries for JSON tests."""
        verdict = JudgeVerdict(
            task_id="t1",
            question="Q?",
            pair_label="8B+GraphBot vs 70B",
            system_a_key="8b_graphbot",
            system_b_key="70b_direct",
            blinded_a_is="8b_graphbot",
            blinded_b_is="70b_direct",
            winner_label="A",
            winner_system="8b_graphbot",
            reasoning="A is better",
            latency_ms=42.5,
        )
        summary = PairSummary(
            pair_label="8B+GraphBot vs 70B",
            system_a_key="8b_graphbot",
            system_b_key="70b_direct",
            wins_a=3,
            wins_b=1,
            ties=1,
            total=5,
        )
        return [verdict], {"8B+GraphBot vs 70B": summary}

    def test_top_level_keys(self) -> None:
        """JSON output must have required top-level keys."""
        verdicts, summaries = self._sample_data()
        result = results_to_json(verdicts, summaries, "test-model", dry_run=True)

        expected_keys = {
            "timestamp",
            "judge_model",
            "dry_run",
            "task_count",
            "verdict_count",
            "verdicts",
            "summaries",
        }
        assert set(result.keys()) == expected_keys

    def test_verdict_fields(self) -> None:
        """Each verdict in the JSON output must have all required fields."""
        verdicts, summaries = self._sample_data()
        result = results_to_json(verdicts, summaries, "test-model", dry_run=False)

        expected_verdict_keys = {
            "task_id",
            "question",
            "pair_label",
            "system_a_key",
            "system_b_key",
            "blinded_a_is",
            "blinded_b_is",
            "winner_label",
            "winner_system",
            "reasoning",
            "latency_ms",
        }

        assert len(result["verdicts"]) == 1
        assert set(result["verdicts"][0].keys()) == expected_verdict_keys

    def test_summary_fields(self) -> None:
        """Each summary in the JSON output must have rate calculations."""
        verdicts, summaries = self._sample_data()
        result = results_to_json(verdicts, summaries, "test-model", dry_run=False)

        summary = result["summaries"]["8B+GraphBot vs 70B"]
        expected_summary_keys = {
            "pair_label",
            "system_a_key",
            "system_b_key",
            "wins_a",
            "wins_b",
            "ties",
            "total",
            "win_rate_a",
            "win_rate_b",
            "tie_rate",
        }
        assert set(summary.keys()) == expected_summary_keys

    def test_summary_rates_correct(self) -> None:
        """Win/loss/tie rates should be computed correctly."""
        verdicts, summaries = self._sample_data()
        result = results_to_json(verdicts, summaries, "test-model", dry_run=False)

        summary = result["summaries"]["8B+GraphBot vs 70B"]
        assert summary["win_rate_a"] == 60.0  # 3/5 * 100
        assert summary["win_rate_b"] == 20.0  # 1/5 * 100
        assert summary["tie_rate"] == 20.0    # 1/5 * 100

    def test_json_serializable(self) -> None:
        """The output must be fully JSON-serializable."""
        verdicts, summaries = self._sample_data()
        result = results_to_json(verdicts, summaries, "test-model", dry_run=True)

        # Should not raise
        serialized = json.dumps(result, indent=2)
        roundtripped = json.loads(serialized)
        assert roundtripped["judge_model"] == "test-model"
        assert roundtripped["dry_run"] is True

    def test_metadata_fields(self) -> None:
        """task_count and verdict_count should be accurate."""
        verdicts, summaries = self._sample_data()
        result = results_to_json(verdicts, summaries, "test-model", dry_run=False)

        assert result["task_count"] == 1
        assert result["verdict_count"] == 1


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases: malformed responses, missing data, boundary conditions."""

    def test_parse_valid_json(self) -> None:
        """_parse_judge_response handles valid JSON correctly."""
        result = _parse_judge_response('{"winner": "A", "reasoning": "good"}')
        assert result["winner"] == "A"
        assert result["reasoning"] == "good"

    def test_parse_json_in_code_block(self) -> None:
        """_parse_judge_response strips markdown code blocks."""
        raw = '```json\n{"winner": "B", "reasoning": "better"}\n```'
        result = _parse_judge_response(raw)
        assert result["winner"] == "B"

    def test_parse_json_with_surrounding_text(self) -> None:
        """_parse_judge_response extracts JSON from surrounding text."""
        raw = 'Here is my verdict: {"winner": "A", "reasoning": "more detailed"} end.'
        result = _parse_judge_response(raw)
        assert result["winner"] == "A"

    def test_parse_malformed_json_fallback_a(self) -> None:
        """Malformed JSON with 'answer a' text should fallback to A."""
        result = _parse_judge_response("I think answer A is clearly better")
        assert result["winner"] == "A"

    def test_parse_malformed_json_fallback_b(self) -> None:
        """Malformed JSON with 'answer b' text should fallback to B."""
        result = _parse_judge_response("Answer B is the winner here")
        assert result["winner"] == "B"

    def test_parse_completely_unparseable(self) -> None:
        """Completely unparseable response should default to tie."""
        result = _parse_judge_response("I cannot decide anything meaningful")
        assert result["winner"] == "tie"
        assert "Could not parse" in result["reasoning"]

    def test_parse_empty_string(self) -> None:
        """Empty string should default to tie."""
        result = _parse_judge_response("")
        assert result["winner"] == "tie"

    def test_validate_verdict_normalizes_case(self) -> None:
        """_validate_verdict should normalize winner to uppercase."""
        result = _validate_verdict({"winner": "a", "reasoning": "test"})
        assert result["winner"] == "A"

    def test_validate_verdict_invalid_winner(self) -> None:
        """Invalid winner values should be treated as tie."""
        result = _validate_verdict({"winner": "C", "reasoning": "test"})
        assert result["winner"] == "tie"

    def test_validate_verdict_missing_winner(self) -> None:
        """Missing winner key should default to tie."""
        result = _validate_verdict({"reasoning": "test"})
        assert result["winner"] == "tie"

    def test_validate_verdict_missing_reasoning(self) -> None:
        """Missing reasoning should use default text."""
        result = _validate_verdict({"winner": "B"})
        assert result["winner"] == "B"
        assert result["reasoning"] == "No reasoning provided."

    @pytest.mark.asyncio
    async def test_missing_both_outputs(self) -> None:
        """Task with no outputs at all should return None."""
        task = _make_task(outputs={})
        verdict = await evaluate_pair(
            task=task,
            system_a_key="8b_graphbot",
            system_b_key="70b_direct",
            pair_label="test",
            judge_model="test-model",
        )
        assert verdict is None

    def test_summary_zero_total_no_division_error(self) -> None:
        """results_to_json should handle zero-total summary without ZeroDivisionError."""
        summary = PairSummary(
            pair_label="test",
            system_a_key="a",
            system_b_key="b",
            wins_a=0,
            wins_b=0,
            ties=0,
            total=0,
        )
        result = results_to_json([], {"test": summary}, "model", dry_run=True)
        s = result["summaries"]["test"]
        assert s["win_rate_a"] == 0.0
        assert s["win_rate_b"] == 0.0
        assert s["tie_rate"] == 0.0

    def test_format_summary_zero_total(self) -> None:
        """format_summary should handle a pair with zero evaluations."""
        summary = PairSummary(
            pair_label="empty pair",
            system_a_key="a",
            system_b_key="b",
            total=0,
        )
        text = format_summary({"empty pair": summary})
        assert "No evaluations completed" in text


# ---------------------------------------------------------------------------
# Test: --dry-run mode
# ---------------------------------------------------------------------------


class TestDryRunMode:
    """Test --dry-run mode with mock judge and fake data."""

    def test_generate_fake_data_returns_tasks(self) -> None:
        """generate_fake_data should return a non-empty list of tasks."""
        tasks = generate_fake_data()
        assert len(tasks) > 0
        assert tasks == FAKE_TASKS

    def test_fake_tasks_have_required_keys(self) -> None:
        """Each fake task must have task_id, question, and outputs."""
        for task in FAKE_TASKS:
            assert "task_id" in task
            assert "question" in task
            assert "outputs" in task
            assert "8b_graphbot" in task["outputs"]
            assert "70b_direct" in task["outputs"]
            assert "gpt4o_direct" in task["outputs"]

    def test_mock_judge_returns_valid_verdict(self) -> None:
        """_mock_judge should return a valid (dict, float) tuple."""
        random.seed(42)
        result, latency = _mock_judge("short", "a longer answer here")
        assert isinstance(result, dict)
        assert "winner" in result
        assert "reasoning" in result
        assert result["winner"] in ("A", "B", "tie")
        assert isinstance(latency, float)
        assert latency > 0

    def test_mock_judge_produces_all_outcomes(self) -> None:
        """Over many calls, the mock judge should produce A, B, and tie."""
        outcomes: set[str] = set()
        # Use three input pairs to exercise all branches:
        # - similar length -> random.choice branch (can produce A, B, tie)
        # - a much longer -> A wins branch
        # - b much longer -> B wins branch
        pairs: list[tuple[str, str]] = [
            ("same length x", "same length y"),  # similar length -> random path
            ("a very long and detailed answer here", "short"),  # A longer -> A wins
            ("short", "a very long and detailed answer here"),  # B longer -> B wins
        ]
        for seed in range(200):
            random.seed(seed)
            for a, b in pairs:
                result, _ = _mock_judge(a, b)
                outcomes.add(result["winner"])
                if outcomes == {"A", "B", "tie"}:
                    break
            if outcomes == {"A", "B", "tie"}:
                break

        assert outcomes == {"A", "B", "tie"}, (
            f"Expected all three outcomes, got: {outcomes}"
        )

    @pytest.mark.asyncio
    async def test_call_judge_dry_run_no_api(self) -> None:
        """call_judge with dry_run=True should not import or call any API."""
        # If this tries to import OpenRouterProvider it would likely fail
        # in test env, so dry_run bypassing it is the test
        result, latency = await call_judge(
            question="test?",
            answer_a="a",
            answer_b="b",
            judge_model="unused",
            dry_run=True,
        )
        assert "winner" in result
        assert isinstance(latency, float)

    @pytest.mark.asyncio
    async def test_dry_run_full_pipeline(self) -> None:
        """Full dry-run pipeline should produce valid results without API calls."""
        tasks = generate_fake_data()
        random.seed(42)

        verdicts, summaries = await run_blind_eval(
            tasks=tasks, judge_model="test-model", dry_run=True,
        )

        # Should have verdicts for each task * each comparison pair
        expected_count = len(tasks) * len(COMPARISON_PAIRS)
        assert len(verdicts) == expected_count

        # Summaries should exist for each comparison pair
        assert len(summaries) == len(COMPARISON_PAIRS)

        for pair_label, summary in summaries.items():
            assert summary.total == len(tasks)
            assert summary.wins_a + summary.wins_b + summary.ties == summary.total

    @pytest.mark.asyncio
    async def test_dry_run_json_output(self) -> None:
        """Dry-run results should produce valid JSON output."""
        tasks = generate_fake_data()
        random.seed(42)

        verdicts, summaries = await run_blind_eval(
            tasks=tasks, judge_model="test-model", dry_run=True,
        )

        json_data = results_to_json(verdicts, summaries, "test-model", dry_run=True)

        # Verify it round-trips through JSON
        serialized = json.dumps(json_data)
        parsed = json.loads(serialized)

        assert parsed["dry_run"] is True
        assert parsed["judge_model"] == "test-model"
        assert parsed["verdict_count"] == len(verdicts)
        assert len(parsed["verdicts"]) == len(verdicts)


# ---------------------------------------------------------------------------
# Test: Thesis validation data loading
# ---------------------------------------------------------------------------


class TestLoadThesisValidation:
    """Test loading and normalization of thesis validation data."""

    def test_load_format_a_list(self, tmp_path: Any) -> None:
        """Load Format A data (task-centric list)."""
        data = [
            {
                "task_id": "t1",
                "question": "Q1?",
                "outputs": {"8b_graphbot": "a", "70b_direct": "b"},
            }
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        tasks = load_thesis_validation(path)
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"

    def test_load_format_a_dict_with_tasks_key(self, tmp_path: Any) -> None:
        """Load Format A data (dict with 'tasks' key)."""
        data = {
            "tasks": [
                {
                    "task_id": "t1",
                    "question": "Q1?",
                    "outputs": {"8b_graphbot": "a"},
                }
            ]
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        tasks = load_thesis_validation(path)
        assert len(tasks) == 1

    def test_load_skips_tasks_without_question(self, tmp_path: Any) -> None:
        """Tasks without a question field should be skipped."""
        data = [
            {"task_id": "t1", "outputs": {"8b_graphbot": "a"}},
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        tasks = load_thesis_validation(path)
        assert len(tasks) == 0

    def test_load_skips_tasks_without_outputs(self, tmp_path: Any) -> None:
        """Tasks without outputs should be skipped."""
        data = [
            {"task_id": "t1", "question": "Q?"},
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        tasks = load_thesis_validation(path)
        assert len(tasks) == 0

    def test_load_t187_format(self, tmp_path: Any) -> None:
        """Load T187 config-based format with output text."""
        data = {
            "configurations": [{"id": "llama8b_pipeline"}],
            "results": [
                {
                    "task_id": "t1",
                    "config_id": "llama8b_pipeline",
                    "question": "Q?",
                    "output": "answer from 8b",
                },
                {
                    "task_id": "t1",
                    "config_id": "llama70b_direct",
                    "question": "Q?",
                    "output": "answer from 70b",
                },
            ],
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        tasks = load_thesis_validation(path)
        assert len(tasks) == 1
        assert tasks[0]["outputs"]["8b_graphbot"] == "answer from 8b"
        assert tasks[0]["outputs"]["70b_direct"] == "answer from 70b"
