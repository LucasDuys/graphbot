"""Blind LLM-as-judge evaluation with pairwise comparison.

Reads outputs from benchmarks/thesis_validation.json (produced by the thesis
validation script), creates pairwise comparisons between 8B+GraphBot vs 70B
direct and 8B+GraphBot vs GPT-4o direct, randomizes label assignment for
blinding, and sends both outputs to a configurable judge model.

Computes win/loss/tie rates per comparison pair and saves results to
benchmarks/blind_eval.json.

Usage:
    python scripts/blind_eval.py                   # full evaluation
    python scripts/blind_eval.py --dry-run          # mock judge, fake data
    python scripts/blind_eval.py --judge-model X    # custom judge model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_env() -> None:
    """Load environment variables from .env.local if present."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


load_env()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL: str = "anthropic/claude-sonnet-4-20250514"

JUDGE_SYSTEM_PROMPT: str = (
    "You are an impartial judge evaluating two answers to a question. "
    "You will be given the original question and two candidate answers "
    "labeled A and B. Evaluate which answer is better based on accuracy, "
    "completeness, clarity, and usefulness. "
    "Respond ONLY with valid JSON: "
    '{\"winner\": \"A\" or \"B\" or \"tie\", \"reasoning\": \"...\"}'
)

JUDGE_USER_TEMPLATE: str = (
    "Question:\n{question}\n\n"
    "Answer A:\n{answer_a}\n\n"
    "Answer B:\n{answer_b}\n\n"
    "Which answer is better? Respond with JSON: "
    '{{"winner": "A" or "B" or "tie", "reasoning": "..."}}'
)

# Comparison pair definitions: (label, system_a_key, system_b_key)
COMPARISON_PAIRS: list[tuple[str, str, str]] = [
    ("8B+GraphBot vs 70B", "8b_graphbot", "70b_direct"),
    ("8B+GraphBot vs GPT-4o", "8b_graphbot", "gpt4o_direct"),
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    """Result from a single pairwise judge evaluation."""

    task_id: str
    question: str
    pair_label: str
    system_a_key: str
    system_b_key: str
    blinded_a_is: str  # which system was labeled A
    blinded_b_is: str  # which system was labeled B
    winner_label: str  # "A", "B", or "tie"
    winner_system: str  # resolved system key or "tie"
    reasoning: str
    latency_ms: float


@dataclass
class PairSummary:
    """Win/loss/tie summary for a comparison pair."""

    pair_label: str
    system_a_key: str
    system_b_key: str
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    total: int = 0


# ---------------------------------------------------------------------------
# Fake data for --dry-run mode
# ---------------------------------------------------------------------------

FAKE_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "math_01",
        "question": "What is 247 * 38?",
        "outputs": {
            "8b_graphbot": "247 * 38 = 9386. The multiplication proceeds by: 247*30=7410, 247*8=1976, 7410+1976=9386.",
            "70b_direct": "247 * 38 = 9,386",
            "gpt4o_direct": "247 multiplied by 38 equals 9,386.",
        },
    },
    {
        "task_id": "factual_01",
        "question": "What is the capital of France?",
        "outputs": {
            "8b_graphbot": "The capital of France is Paris. Paris is the largest city in France, located in the north-central part of the country along the Seine River.",
            "70b_direct": "The capital of France is Paris.",
            "gpt4o_direct": "Paris is the capital city of France.",
        },
    },
    {
        "task_id": "reasoning_01",
        "question": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "outputs": {
            "8b_graphbot": "No, we cannot conclude that some roses fade quickly. This is a formal logic error known as the fallacy of the undistributed middle. While all roses are flowers, the flowers that fade quickly may be an entirely separate subset that does not include any roses.",
            "70b_direct": "No, we cannot logically conclude that. The fact that some flowers fade quickly does not necessarily apply to roses specifically.",
            "gpt4o_direct": "We cannot conclude that some roses fade quickly. The premise only states that some flowers fade quickly, but those flowers might not include roses.",
        },
    },
    {
        "task_id": "coding_01",
        "question": "Write a Python function to check if a string is a palindrome.",
        "outputs": {
            "8b_graphbot": "def is_palindrome(s: str) -> bool:\n    cleaned = s.lower().replace(' ', '')\n    return cleaned == cleaned[::-1]\n\n# Example: is_palindrome('racecar') returns True",
            "70b_direct": "def is_palindrome(s):\n    return s == s[::-1]",
            "gpt4o_direct": "def is_palindrome(text: str) -> bool:\n    \"\"\"Check if a string is a palindrome.\"\"\"\n    cleaned = ''.join(c.lower() for c in text if c.isalnum())\n    return cleaned == cleaned[::-1]",
        },
    },
    {
        "task_id": "explanation_01",
        "question": "Explain how a binary search tree works.",
        "outputs": {
            "8b_graphbot": "A binary search tree (BST) is a data structure where each node has at most two children. The left subtree contains only nodes with values less than the parent, and the right subtree contains only nodes with values greater than the parent. This ordering property enables efficient searching, insertion, and deletion with O(log n) average time complexity.",
            "70b_direct": "A binary search tree is a tree data structure where each node has a left child with a smaller value and a right child with a larger value. You can search it in O(log n) time by comparing the target with each node and going left or right.",
            "gpt4o_direct": "A BST organizes data hierarchically. Each node stores a key, with left children being smaller and right children being larger. Operations like search, insert, and delete run in O(log n) on average, though worst case is O(n) for a degenerate tree.",
        },
    },
]


def generate_fake_data() -> list[dict[str, Any]]:
    """Generate fake thesis validation data for --dry-run mode."""
    return FAKE_TASKS


# ---------------------------------------------------------------------------
# Thesis validation data loading
# ---------------------------------------------------------------------------


def load_thesis_validation(path: Path) -> list[dict[str, Any]]:
    """Load and normalize thesis_validation.json into task list.

    Supports two input formats:

    Format A (task-centric, preferred for blind eval):
      A list/dict of task objects each containing:
        - task_id or id: string identifier
        - question or description: the original prompt
        - outputs: dict mapping system keys to output strings
      The outputs dict should use keys like "8b_graphbot", "70b_direct",
      "gpt4o_direct".

    Format B (T187 config-based results):
      A dict with "results" containing per-config/task entries with keys
      like config_id, task_id, task_category, plus an "output" field.
      Config IDs are mapped: llama8b_pipeline -> 8b_graphbot,
      llama70b_direct -> 70b_direct, gpt4o_direct -> gpt4o_direct.

    Returns normalized list of dicts with keys: task_id, question, outputs.
    """
    raw: Any = json.loads(path.read_text())

    # Detect format
    if isinstance(raw, dict) and "configurations" in raw:
        return _load_t187_format(raw)

    # Format A: task-centric
    if isinstance(raw, dict):
        tasks_raw = raw.get("tasks", raw.get("results", []))
    elif isinstance(raw, list):
        tasks_raw = raw
    else:
        raise ValueError(
            f"Unexpected format in {path}: expected list or dict, "
            f"got {type(raw).__name__}"
        )

    tasks: list[dict[str, Any]] = []
    for entry in tasks_raw:
        task_id: str = entry.get("task_id", entry.get("id", "unknown"))
        question: str = entry.get("question", entry.get("description", ""))
        outputs: dict[str, str] = entry.get("outputs", {})

        if not question:
            print(f"  Warning: skipping task {task_id} -- no question found")
            continue
        if not outputs:
            print(f"  Warning: skipping task {task_id} -- no outputs found")
            continue

        tasks.append({
            "task_id": task_id,
            "question": question,
            "outputs": outputs,
        })

    return tasks


# Config ID mapping from T187 format to blind eval system keys
_CONFIG_TO_SYSTEM: dict[str, str] = {
    "llama8b_pipeline": "8b_graphbot",
    "llama70b_direct": "70b_direct",
    "gpt4o_direct": "gpt4o_direct",
}


def _load_t187_format(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Load T187 config-based thesis_validation.json format.

    Groups results by task_id and collects outputs per system. Requires
    that results contain an 'output' field with the actual response text.
    """
    results: list[dict[str, Any]] = raw.get("results", [])

    # Check if results contain output text
    has_outputs = any("output" in r for r in results)
    if not has_outputs:
        print(
            "  Warning: thesis_validation.json uses T187 config format but "
            "results lack 'output' field with response text."
        )
        print(
            "  Blind evaluation requires actual outputs. Re-run the thesis "
            "validation script with output capture enabled, or use --dry-run."
        )
        return []

    # Group by task_id
    by_task: dict[str, dict[str, Any]] = {}
    for r in results:
        task_id: str = r.get("task_id", "unknown")
        config_id: str = r.get("config_id", "")
        output: str = r.get("output", "")
        question: str = r.get("question", r.get("description", ""))

        system_key = _CONFIG_TO_SYSTEM.get(config_id)
        if system_key is None:
            continue

        if task_id not in by_task:
            by_task[task_id] = {
                "task_id": task_id,
                "question": question,
                "outputs": {},
            }

        # Prefer question from any result that has it
        if question and not by_task[task_id]["question"]:
            by_task[task_id]["question"] = question

        by_task[task_id]["outputs"][system_key] = output

    tasks: list[dict[str, Any]] = []
    for task_id, task_data in by_task.items():
        if not task_data["question"]:
            print(f"  Warning: skipping task {task_id} -- no question found")
            continue
        if not task_data["outputs"]:
            print(f"  Warning: skipping task {task_id} -- no outputs found")
            continue
        tasks.append(task_data)

    return tasks


# ---------------------------------------------------------------------------
# Judge interaction
# ---------------------------------------------------------------------------


async def call_judge(
    question: str,
    answer_a: str,
    answer_b: str,
    judge_model: str,
    dry_run: bool = False,
) -> tuple[dict[str, str], float]:
    """Send a pairwise comparison to the judge model.

    Returns (parsed_response, latency_ms).
    The parsed response contains 'winner' and 'reasoning' keys.
    """
    if dry_run:
        return _mock_judge(answer_a, answer_b)

    from models.openrouter import OpenRouterProvider

    provider = OpenRouterProvider()

    user_prompt: str = JUDGE_USER_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    start = time.perf_counter()
    result = await provider.complete(messages, judge_model)
    latency = (time.perf_counter() - start) * 1000

    parsed = _parse_judge_response(result.content)
    return parsed, latency


def _mock_judge(answer_a: str, answer_b: str) -> tuple[dict[str, str], float]:
    """Return a mock judge verdict for --dry-run mode.

    Uses a simple length heuristic: longer answer wins, with some
    randomness to simulate realistic tie rates.
    """
    roll = random.random()
    if roll < 0.15:
        winner = "tie"
        reasoning = "Both answers are roughly equivalent in quality."
    elif len(answer_a) > len(answer_b) * 1.1:
        winner = "A"
        reasoning = "Answer A provides more detail and context."
    elif len(answer_b) > len(answer_a) * 1.1:
        winner = "B"
        reasoning = "Answer B provides more detail and context."
    else:
        winner = random.choice(["A", "B", "tie"])
        reasoning = "Answers are close in quality; marginal preference."

    return {"winner": winner, "reasoning": reasoning}, 50.0


def _parse_judge_response(content: str) -> dict[str, str]:
    """Parse the judge model's JSON response.

    Handles common formatting issues like markdown code blocks or
    extra text surrounding the JSON.
    """
    text = content.strip()

    # Strip markdown code block wrappers if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        json_lines: list[str] = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            if line.strip() == "```" and in_block:
                break
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines).strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        return _validate_verdict(parsed)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from surrounding text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            parsed = json.loads(text[brace_start:brace_end + 1])
            return _validate_verdict(parsed)
        except json.JSONDecodeError:
            pass

    # Fallback: try to detect winner from text
    text_lower = text.lower()
    if '"a"' in text_lower or "answer a" in text_lower:
        return {"winner": "A", "reasoning": f"Parsed from raw text: {text[:200]}"}
    if '"b"' in text_lower or "answer b" in text_lower:
        return {"winner": "B", "reasoning": f"Parsed from raw text: {text[:200]}"}

    return {"winner": "tie", "reasoning": f"Could not parse judge response: {text[:200]}"}


def _validate_verdict(parsed: dict[str, Any]) -> dict[str, str]:
    """Validate and normalize the parsed verdict dict."""
    winner = str(parsed.get("winner", "tie")).strip().upper()
    if winner not in ("A", "B", "TIE"):
        winner = "tie"
    else:
        winner = winner if winner in ("A", "B") else "tie"

    reasoning = str(parsed.get("reasoning", "No reasoning provided."))
    return {"winner": winner, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


async def evaluate_pair(
    task: dict[str, Any],
    system_a_key: str,
    system_b_key: str,
    pair_label: str,
    judge_model: str,
    dry_run: bool = False,
) -> JudgeVerdict | None:
    """Run a single blind pairwise evaluation for one task.

    Randomizes which system is labeled A vs B, calls the judge, and
    returns a JudgeVerdict with resolved winner information.

    Returns None if the task is missing outputs for either system.
    """
    outputs: dict[str, str] = task.get("outputs", {})
    task_id: str = task["task_id"]
    question: str = task["question"]

    output_a: str | None = outputs.get(system_a_key)
    output_b: str | None = outputs.get(system_b_key)

    if output_a is None or output_b is None:
        missing = []
        if output_a is None:
            missing.append(system_a_key)
        if output_b is None:
            missing.append(system_b_key)
        print(
            f"  Warning: task {task_id} missing outputs for "
            f"{', '.join(missing)} -- skipping pair {pair_label}"
        )
        return None

    # Randomize blinding: coin flip determines label assignment
    if random.random() < 0.5:
        label_a_system = system_a_key
        label_b_system = system_b_key
        answer_a = output_a
        answer_b = output_b
    else:
        label_a_system = system_b_key
        label_b_system = system_a_key
        answer_a = output_b
        answer_b = output_a

    verdict, latency = await call_judge(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        judge_model=judge_model,
        dry_run=dry_run,
    )

    winner_label: str = verdict["winner"]

    # Resolve blinded label back to system key
    if winner_label == "A":
        winner_system = label_a_system
    elif winner_label == "B":
        winner_system = label_b_system
    else:
        winner_system = "tie"

    return JudgeVerdict(
        task_id=task_id,
        question=question,
        pair_label=pair_label,
        system_a_key=system_a_key,
        system_b_key=system_b_key,
        blinded_a_is=label_a_system,
        blinded_b_is=label_b_system,
        winner_label=winner_label,
        winner_system=winner_system,
        reasoning=verdict["reasoning"],
        latency_ms=latency,
    )


async def run_blind_eval(
    tasks: list[dict[str, Any]],
    judge_model: str,
    dry_run: bool = False,
) -> tuple[list[JudgeVerdict], dict[str, PairSummary]]:
    """Run blind pairwise evaluation across all tasks and comparison pairs.

    Returns (verdicts, summaries) where summaries is keyed by pair_label.
    """
    verdicts: list[JudgeVerdict] = []
    summaries: dict[str, PairSummary] = {}

    for pair_label, sys_a, sys_b in COMPARISON_PAIRS:
        summaries[pair_label] = PairSummary(
            pair_label=pair_label,
            system_a_key=sys_a,
            system_b_key=sys_b,
        )

    for task in tasks:
        task_id = task["task_id"]
        for pair_label, sys_a, sys_b in COMPARISON_PAIRS:
            print(f"  [{task_id}] {pair_label}...", end=" ", flush=True)

            verdict = await evaluate_pair(
                task=task,
                system_a_key=sys_a,
                system_b_key=sys_b,
                pair_label=pair_label,
                judge_model=judge_model,
                dry_run=dry_run,
            )

            if verdict is None:
                print("SKIPPED")
                continue

            verdicts.append(verdict)
            summary = summaries[pair_label]
            summary.total += 1

            if verdict.winner_system == sys_a:
                summary.wins_a += 1
                print(f"-> {sys_a} wins")
            elif verdict.winner_system == sys_b:
                summary.wins_b += 1
                print(f"-> {sys_b} wins")
            else:
                summary.ties += 1
                print("-> tie")

    return verdicts, summaries


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_summary(summaries: dict[str, PairSummary]) -> str:
    """Format win/loss/tie summary for console output."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"Blind Evaluation Summary -- {now}")
    lines.append("=" * 60)

    for pair_label, summary in summaries.items():
        if summary.total == 0:
            lines.append(f"\n{pair_label}: No evaluations completed")
            continue

        lines.append(f"\n{pair_label}:")
        lines.append(f"  {summary.system_a_key}: {summary.wins_a} wins")
        lines.append(f"  {summary.system_b_key}: {summary.wins_b} wins")
        lines.append(f"  Ties: {summary.ties}")
        lines.append(f"  Total: {summary.total}")

        win_rate_a = summary.wins_a / summary.total * 100
        win_rate_b = summary.wins_b / summary.total * 100
        tie_rate = summary.ties / summary.total * 100
        lines.append(
            f"  Rates: {win_rate_a:.1f}% / {win_rate_b:.1f}% / {tie_rate:.1f}% "
            f"(win-a / win-b / tie)"
        )

    # Print the one-liner summaries the spec requests
    lines.append("")
    lines.append("-" * 60)
    for pair_label, summary in summaries.items():
        if summary.total == 0:
            continue
        # Use the first system (8b_graphbot) perspective
        lines.append(
            f"{pair_label}: "
            f"{summary.wins_a} wins, {summary.wins_b} losses, {summary.ties} ties"
        )

    lines.append("")
    return "\n".join(lines)


def results_to_json(
    verdicts: list[JudgeVerdict],
    summaries: dict[str, PairSummary],
    judge_model: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Convert evaluation results to a JSON-serializable dict."""
    return {
        "timestamp": datetime.now().isoformat(),
        "judge_model": judge_model,
        "dry_run": dry_run,
        "task_count": len({v.task_id for v in verdicts}),
        "verdict_count": len(verdicts),
        "verdicts": [
            {
                "task_id": v.task_id,
                "question": v.question,
                "pair_label": v.pair_label,
                "system_a_key": v.system_a_key,
                "system_b_key": v.system_b_key,
                "blinded_a_is": v.blinded_a_is,
                "blinded_b_is": v.blinded_b_is,
                "winner_label": v.winner_label,
                "winner_system": v.winner_system,
                "reasoning": v.reasoning,
                "latency_ms": round(v.latency_ms, 2),
            }
            for v in verdicts
        ],
        "summaries": {
            label: {
                "pair_label": s.pair_label,
                "system_a_key": s.system_a_key,
                "system_b_key": s.system_b_key,
                "wins_a": s.wins_a,
                "wins_b": s.wins_b,
                "ties": s.ties,
                "total": s.total,
                "win_rate_a": round(s.wins_a / s.total * 100, 1) if s.total > 0 else 0.0,
                "win_rate_b": round(s.wins_b / s.total * 100, 1) if s.total > 0 else 0.0,
                "tie_rate": round(s.ties / s.total * 100, 1) if s.total > 0 else 0.0,
            }
            for label, s in summaries.items()
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_evaluation(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    dry_run: bool = False,
    seed: int | None = None,
) -> None:
    """Run the full blind evaluation pipeline."""
    if seed is not None:
        random.seed(seed)

    # Load or generate task data
    validation_path = Path(__file__).parent.parent / "benchmarks" / "thesis_validation.json"

    tasks: list[dict[str, Any]] = []

    if validation_path.exists():
        print(f"Loading tasks from {validation_path}")
        tasks = load_thesis_validation(validation_path)

    if not tasks and dry_run:
        print("Dry-run mode: generating fake task data")
        tasks = generate_fake_data()
    elif not tasks and validation_path.exists():
        print("Error: no valid tasks found in thesis_validation.json")
        sys.exit(1)
    elif not tasks:
        print(
            f"Error: {validation_path} not found. "
            "Run the thesis validation script first, or use --dry-run."
        )
        sys.exit(1)

    print(f"Tasks loaded: {len(tasks)}")
    print(f"Judge model: {judge_model}")
    print(f"Dry run: {dry_run}")
    print(f"Comparison pairs: {len(COMPARISON_PAIRS)}")
    print()

    verdicts, summaries = await run_blind_eval(
        tasks=tasks,
        judge_model=judge_model,
        dry_run=dry_run,
    )

    # Print summary
    summary_text = format_summary(summaries)
    print("\n" + summary_text)

    # Save results
    out_dir = Path(__file__).parent.parent / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "blind_eval.json"

    json_data = results_to_json(verdicts, summaries, judge_model, dry_run)
    out_file.write_text(json.dumps(json_data, indent=2))

    print(f"Results saved to {out_file}")


def main() -> None:
    """Parse arguments and run the blind evaluation."""
    parser = argparse.ArgumentParser(
        description="Blind LLM-as-judge pairwise evaluation"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model for evaluation (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock judge and generate fake data if thesis_validation.json is missing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible blinding order",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            judge_model=args.judge_model,
            dry_run=args.dry_run,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
