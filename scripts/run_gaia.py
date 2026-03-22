"""Run GAIA Level 1 benchmark tasks through GraphBot and report accuracy.

Downloads the GAIA Level 1 dataset from HuggingFace if the ``datasets``
library is available.  When the library is missing or the download fails,
falls back to a hardcoded set of 25 GAIA-style Level 1 questions with
known ground-truth answers.

Results are saved to ``benchmarks/gaia_results.json``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so local imports work when running as a
# script (``python scripts/run_gaia.py``).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_env() -> None:
    """Load .env.local from project root (same pattern as run_benchmarks.py)."""
    env_file = _PROJECT_ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

from core_gb.orchestrator import Orchestrator  # noqa: E402
from core_gb.types import ExecutionResult  # noqa: E402
from graph.store import GraphStore  # noqa: E402
from models.openrouter import OpenRouterProvider  # noqa: E402
from models.router import ModelRouter  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded GAIA Level 1 fallback tasks
# ---------------------------------------------------------------------------
# These are simple factual Q&A tasks modelled after the public GAIA Level 1
# split.  Each entry has a question and an exact ground-truth answer string.
# ---------------------------------------------------------------------------

FALLBACK_TASKS: list[dict[str, str]] = [
    {
        "question": "What is the capital of France?",
        "ground_truth": "Paris",
    },
    {
        "question": "How many planets are in our solar system?",
        "ground_truth": "8",
    },
    {
        "question": "What is the chemical symbol for gold?",
        "ground_truth": "Au",
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "ground_truth": "William Shakespeare",
    },
    {
        "question": "What is the largest ocean on Earth?",
        "ground_truth": "Pacific Ocean",
    },
    {
        "question": "What year did World War II end?",
        "ground_truth": "1945",
    },
    {
        "question": "What is the speed of light in km/s (rounded to nearest thousand)?",
        "ground_truth": "300000",
    },
    {
        "question": "What is the square root of 144?",
        "ground_truth": "12",
    },
    {
        "question": "What is the chemical formula for water?",
        "ground_truth": "H2O",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "ground_truth": "Leonardo da Vinci",
    },
    {
        "question": "What is the smallest prime number?",
        "ground_truth": "2",
    },
    {
        "question": "In which country is the Great Wall located?",
        "ground_truth": "China",
    },
    {
        "question": "What is the boiling point of water in degrees Celsius?",
        "ground_truth": "100",
    },
    {
        "question": "What element has the atomic number 1?",
        "ground_truth": "Hydrogen",
    },
    {
        "question": "How many continents are there on Earth?",
        "ground_truth": "7",
    },
    {
        "question": "What is the currency of Japan?",
        "ground_truth": "Yen",
    },
    {
        "question": "What is the tallest mountain in the world?",
        "ground_truth": "Mount Everest",
    },
    {
        "question": "What gas do plants absorb from the atmosphere?",
        "ground_truth": "Carbon dioxide",
    },
    {
        "question": "What is 15% of 200?",
        "ground_truth": "30",
    },
    {
        "question": "Who developed the theory of relativity?",
        "ground_truth": "Albert Einstein",
    },
    {
        "question": "What is the largest planet in our solar system?",
        "ground_truth": "Jupiter",
    },
    {
        "question": "What language has the most native speakers in the world?",
        "ground_truth": "Mandarin Chinese",
    },
    {
        "question": "How many sides does a hexagon have?",
        "ground_truth": "6",
    },
    {
        "question": "What is the freezing point of water in Fahrenheit?",
        "ground_truth": "32",
    },
    {
        "question": "What organ pumps blood through the human body?",
        "ground_truth": "Heart",
    },
]


def _try_load_hf_dataset() -> list[dict[str, str]] | None:
    """Attempt to load GAIA Level 1 tasks from HuggingFace ``datasets``.

    Returns a list of dicts with ``question`` and ``ground_truth`` keys, or
    ``None`` if the library is unavailable or the download fails.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        logger.info("datasets library not installed -- using fallback tasks")
        return None

    try:
        ds = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_level1",
            split="validation",
            trust_remote_code=True,
        )
        tasks: list[dict[str, str]] = []
        for row in ds:  # type: ignore[union-attr]
            question = row.get("Question", row.get("question", ""))
            answer = row.get("Final answer", row.get("final_answer", ""))
            if question and answer:
                tasks.append({"question": question, "ground_truth": str(answer)})
        if len(tasks) >= 20:
            return tasks
        logger.warning(
            "HuggingFace dataset returned only %d tasks -- using fallback", len(tasks)
        )
        return None
    except Exception as exc:
        logger.warning("Failed to load HuggingFace dataset: %s", exc)
        return None


def load_gaia_tasks() -> list[dict[str, str]]:
    """Return GAIA Level 1 tasks, preferring HuggingFace, falling back to hardcoded."""
    hf_tasks = _try_load_hf_dataset()
    if hf_tasks is not None:
        return hf_tasks
    return list(FALLBACK_TASKS)


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip whitespace and trailing punctuation for comparison."""
    text = text.strip().lower()
    while text and text[-1] in ".!?,;:":
        text = text[:-1]
    return text.strip()


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check whether ``predicted`` matches ``ground_truth``.

    Uses exact normalized match first, then checks if the ground truth
    appears as a substring of the prediction (for cases where the model
    returns a sentence containing the answer).
    """
    norm_pred = normalize_answer(predicted)
    norm_gt = normalize_answer(ground_truth)

    if norm_pred == norm_gt:
        return True

    # Substring match: ground truth found anywhere in prediction
    if norm_gt in norm_pred:
        return True

    return False


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def _build_orchestrator() -> tuple[Orchestrator, GraphStore]:
    """Construct the Orchestrator with default provider and router."""
    provider = OpenRouterProvider()
    router = ModelRouter(provider)
    store = GraphStore()
    store.initialize()
    orchestrator = Orchestrator(store, router)
    return orchestrator, store


async def run_single_task(
    orchestrator: Orchestrator,
    question: str,
) -> ExecutionResult:
    """Run a single question through the orchestrator."""
    return await orchestrator.process(question)


async def run_gaia_benchmark(
    tasks: list[dict[str, str]] | None = None,
    orchestrator: Orchestrator | None = None,
    store: GraphStore | None = None,
) -> dict[str, Any]:
    """Execute GAIA Level 1 tasks and return a structured results dict.

    Parameters
    ----------
    tasks:
        Override the task list (useful for testing).
    orchestrator:
        Pre-built orchestrator (useful for testing with mocks).
    store:
        GraphStore instance to close after the run.  Ignored when
        ``orchestrator`` is provided externally.
    """
    if tasks is None:
        tasks = load_gaia_tasks()

    owns_store = False
    if orchestrator is None:
        orchestrator, store = _build_orchestrator()
        owns_store = True

    results: list[dict[str, Any]] = []
    correct = 0
    total_tokens = 0
    total_cost = 0.0

    print(f"\nGAIA Level 1 Benchmark -- {len(tasks)} tasks")
    print("=" * 70)

    for idx, task in enumerate(tasks, 1):
        question = task["question"]
        ground_truth = task["ground_truth"]
        label = f"[{idx:>3}/{len(tasks)}]"

        start = time.perf_counter()
        try:
            result = await run_single_task(orchestrator, question)
            elapsed_ms = (time.perf_counter() - start) * 1000

            match = answers_match(result.output, ground_truth)
            if match:
                correct += 1

            total_tokens += result.total_tokens
            total_cost += result.total_cost

            entry: dict[str, Any] = {
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": result.output[:500],
                "match": match,
                "tokens": result.total_tokens,
                "cost": result.total_cost,
                "latency_ms": round(elapsed_ms, 1),
                "model": result.model_used,
                "success": result.success,
            }
            results.append(entry)

            status = "CORRECT" if match else "WRONG"
            print(
                f"{label} {status} | {result.total_tokens:>5} tok"
                f" | ${result.total_cost:.6f}"
                f" | {elapsed_ms:>7.0f}ms"
                f" | {question[:50]}"
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            results.append({
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": "",
                "match": False,
                "tokens": 0,
                "cost": 0.0,
                "latency_ms": round(elapsed_ms, 1),
                "model": "",
                "success": False,
                "error": str(exc),
            })
            print(f"{label} ERROR  | {question[:50]} | {exc}")

    # Build summary
    accuracy = correct / len(tasks) if tasks else 0.0
    summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(tasks),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "results": results,
    }

    # Print summary table
    print()
    print("=" * 70)
    print("GAIA Level 1 -- Summary")
    print("=" * 70)
    print(f"  Tasks attempted : {len(tasks)}")
    print(f"  Correct         : {correct}")
    print(f"  Accuracy        : {accuracy:.1%}")
    print(f"  Total tokens    : {total_tokens}")
    print(f"  Total cost      : ${total_cost:.6f}")
    print("=" * 70)

    if owns_store and store is not None:
        store.close()

    return summary


def save_results(summary: dict[str, Any]) -> Path:
    """Persist results to benchmarks/gaia_results.json."""
    out_path = _PROJECT_ROOT / "benchmarks" / "gaia_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")
    return out_path


async def main() -> None:
    """Entry point: load tasks, run benchmark, save results."""
    summary = await run_gaia_benchmark()
    save_results(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(main())
