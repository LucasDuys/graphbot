"""Convenience script: run thesis validation + blind evaluation in sequence.

Executes validate_thesis.py first to produce benchmarks/thesis_validation.json,
then runs blind_eval.py using those outputs. All shared flags (--dry-run) are
forwarded to both scripts. Blind-eval-specific flags (--judge-model, --seed)
are forwarded only to the second step.

Usage:
    python scripts/run_full_validation.py --dry-run
    python scripts/run_full_validation.py
    python scripts/run_full_validation.py --judge-model openai/gpt-4o --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _print_banner(text: str) -> None:
    """Print a section banner to stdout."""
    width: int = 70
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)
    print()


async def _run_thesis_validation(dry_run: bool, task: str | None) -> bool:
    """Run the thesis validation step.

    Returns True on success, False on failure.
    """
    _print_banner("STEP 1: Thesis Validation Benchmark")

    try:
        from scripts.validate_thesis import run_validation, TASKS
    except ImportError:
        # Fallback: add scripts dir to path
        sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
        from validate_thesis import run_validation, TASKS

    tasks_to_run: list[dict[str, Any]] = TASKS
    if task:
        matching = [t for t in TASKS if t["id"] == task]
        if not matching:
            print(f"Unknown task ID: {task}")
            return False
        tasks_to_run = matching

    try:
        await run_validation(dry_run=dry_run, tasks=tasks_to_run)
        return True
    except Exception as exc:
        print(f"Thesis validation failed: {exc}")
        return False


async def _run_blind_eval(
    dry_run: bool,
    judge_model: str | None,
    seed: int | None,
) -> bool:
    """Run the blind evaluation step.

    Returns True on success, False on failure.
    """
    _print_banner("STEP 2: Blind Pairwise Evaluation")

    try:
        from scripts.blind_eval import run_evaluation, DEFAULT_JUDGE_MODEL
    except ImportError:
        from blind_eval import run_evaluation, DEFAULT_JUDGE_MODEL

    model: str = judge_model if judge_model else DEFAULT_JUDGE_MODEL

    # The blind eval auto-detects benchmarks/thesis_validation.json
    try:
        await run_evaluation(
            judge_model=model,
            dry_run=dry_run,
            seed=seed,
        )
        return True
    except Exception as exc:
        print(f"Blind evaluation failed: {exc}")
        return False


async def run_full(
    dry_run: bool = False,
    task: str | None = None,
    judge_model: str | None = None,
    seed: int | None = None,
) -> int:
    """Run the complete validation pipeline.

    Returns 0 on success, 1 on failure.
    """
    overall_start: float = time.perf_counter()

    # Step 1: Thesis validation
    step1_ok: bool = await _run_thesis_validation(dry_run=dry_run, task=task)
    if not step1_ok:
        print("Aborting: thesis validation failed.")
        return 1

    # Verify output file exists before proceeding
    validation_output: Path = _PROJECT_ROOT / "benchmarks" / "thesis_validation.json"
    if not validation_output.exists():
        print(f"Expected output not found: {validation_output}")
        print("Aborting: cannot proceed to blind evaluation without validation data.")
        return 1

    # Step 2: Blind evaluation
    step2_ok: bool = await _run_blind_eval(
        dry_run=dry_run,
        judge_model=judge_model,
        seed=seed,
    )
    if not step2_ok:
        print("Blind evaluation failed.")
        return 1

    # Summary
    elapsed: float = time.perf_counter() - overall_start
    _print_banner("VALIDATION COMPLETE")
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("Output files:")
    print(f"  benchmarks/thesis_validation.json  (Step 1 -- benchmark data)")
    print(f"  benchmarks/blind_eval.json         (Step 2 -- judge verdicts)")
    print()

    return 0


def main() -> None:
    """Parse arguments and run the full validation pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full thesis validation pipeline: "
            "benchmark (validate_thesis.py) followed by "
            "blind evaluation (blind_eval.py)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock providers for both steps (no API calls, no cost)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run a single task by ID in the validation step (e.g. qa_01)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model for blind evaluation (default: Claude Sonnet)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible blinding order in blind eval",
    )
    args = parser.parse_args()

    exit_code: int = asyncio.run(
        run_full(
            dry_run=args.dry_run,
            task=args.task,
            judge_model=args.judge_model,
            seed=args.seed,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
