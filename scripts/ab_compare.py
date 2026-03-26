"""A/B comparison script for prompt versions.

Runs the same set of tasks with two prompt versions (v1 vs v2) and compares
GPT-4o-mini judge scores. Uses the task list from validate_single_call.py.

Usage:
    python scripts/ab_compare.py [--version-a v1] [--version-b v2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env.local"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(_PROJECT_ROOT))

from core_gb.langsmith_prompts import (
    PROMPT_TEMPLATES,
    LangSmithPromptManager,
)
from models.openrouter import OpenRouterProvider


# ---------------------------------------------------------------------------
# Task definitions (subset from validate_single_call.py)
# ---------------------------------------------------------------------------

@dataclass
class ABTask:
    id: str
    category: str  # simple_qa, hard_reasoning, tool_execution, creative, analysis
    prompt_name: str
    question: str
    ground_truth_hints: str


@dataclass
class ABResult:
    task_id: str
    prompt_name: str
    version: str
    question: str
    output: str
    quality: int
    judge_reasoning: str


AB_TASKS: list[ABTask] = [
    ABTask(
        "sq_01", "simple_qa", "simple_qa",
        "What is the capital of Japan?",
        "Tokyo",
    ),
    ABTask(
        "sq_02", "simple_qa", "simple_qa",
        "Who wrote '1984'?",
        "George Orwell",
    ),
    ABTask(
        "hr_01", "hard_reasoning", "hard_reasoning",
        "A company has 3 products: A ($50, 30% margin), B ($120, 45% margin), "
        "C ($200, 20% margin). They sold 1000 units of A, 500 of B, 200 of C. "
        "Calculate total revenue and total profit. Show your work.",
        "revenue,profit,50000,60000,40000,margin",
    ),
    ABTask(
        "hr_02", "hard_reasoning", "hard_reasoning",
        "Compare merge sort and quick sort: time complexity (best, average, worst), "
        "space complexity, stability, and when to use each.",
        "O(n log n),O(n^2),stable,unstable,in-place,merge,quick",
    ),
    ABTask(
        "te_01", "tool_execution", "tool_execution",
        "Explain how you would use file system tools to find all Python files "
        "in a project directory and count the total lines of code.",
        "find,count,lines,python,directory",
    ),
    ABTask(
        "cr_01", "creative", "creative",
        "Write a haiku about debugging code at 3am.",
        "haiku,5-7-5,debug,code,night",
    ),
    ABTask(
        "an_01", "analysis", "analysis",
        "Given that Q1 revenue was $1M, Q2 was $1.2M, Q3 was $0.9M, Q4 was $1.5M, "
        "analyze the trend and predict Q1 next year with reasoning.",
        "trend,growth,seasonal,prediction,Q1",
    ),
]


# ---------------------------------------------------------------------------
# Judge (GPT-4o-mini)
# ---------------------------------------------------------------------------

async def judge_quality(
    provider: OpenRouterProvider,
    question: str,
    answer: str,
    ground_truth_hints: str,
) -> tuple[int, str]:
    """Use GPT-4o-mini as judge. Returns (score 1-5, reasoning)."""
    prompt = (
        "You are an impartial answer quality judge. Rate the following answer "
        "on a 1-5 scale.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER TO JUDGE:\n{answer[:2000]}\n\n"
        f"GROUND TRUTH HINTS (keywords the answer should contain): "
        f"{ground_truth_hints}\n\n"
        "SCORING:\n"
        "1 = Wrong, irrelevant, or refuses to answer\n"
        "2 = Partially correct but major gaps or errors\n"
        "3 = Mostly correct but missing important details\n"
        "4 = Good answer, covers the key points\n"
        "5 = Excellent, thorough, accurate, well-structured\n\n"
        'Respond with ONLY a JSON object: {"score": <1-5>, "reasoning": "<one sentence>"}'
    )

    try:
        result = await provider.complete(
            [{"role": "user", "content": prompt}],
            "openai/gpt-4o-mini",
        )
        data = json.loads(result.content)
        return int(data["score"]), str(data["reasoning"])
    except Exception:
        return 3, "judge failed, default score"


# ---------------------------------------------------------------------------
# Run a single task with a given prompt version
# ---------------------------------------------------------------------------

async def run_task_with_version(
    provider: OpenRouterProvider,
    task: ABTask,
    version: str,
) -> ABResult:
    """Run a single task using its prompt template and return judged result."""
    template = PROMPT_TEMPLATES[task.prompt_name]

    # Format the prompt with empty context/examples (standalone test).
    messages_lc = template.format_messages(
        context="No additional context provided.",
        examples="No examples provided.",
        task=task.question,
    )

    # Convert LangChain messages to dicts for OpenRouterProvider.
    messages: list[dict[str, str]] = []
    for msg in messages_lc:
        role = "user"
        if msg.type == "system":
            role = "system"
        elif msg.type == "ai":
            role = "assistant"
        elif msg.type == "human":
            role = "user"
        messages.append({"role": role, "content": msg.content})

    try:
        result = await provider.complete(
            messages,
            "openai/gpt-4o-mini",
            metadata={
                "prompt_name": task.prompt_name,
                "prompt_version": version,
                "tags": [f"prompt:{task.prompt_name}", f"version:{version}"],
            },
        )
        output = result.content
    except Exception as exc:
        output = f"ERROR: {exc}"

    quality, reasoning = await judge_quality(
        provider, task.question, output, task.ground_truth_hints,
    )

    return ABResult(
        task_id=task.id,
        prompt_name=task.prompt_name,
        version=version,
        question=task.question,
        output=output[:500],
        quality=quality,
        judge_reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(version_a: str, version_b: str) -> None:
    print(f"A/B Prompt Comparison: {version_a} vs {version_b}")
    print("=" * 70)

    provider = OpenRouterProvider()
    results_a: list[ABResult] = []
    results_b: list[ABResult] = []

    for i, task in enumerate(AB_TASKS, 1):
        print(
            f"  [{i:2d}/{len(AB_TASKS)}] {task.prompt_name:<18s} "
            f"{task.question[:45]}...",
            end="",
            flush=True,
        )

        ra = await run_task_with_version(provider, task, version_a)
        rb = await run_task_with_version(provider, task, version_b)

        results_a.append(ra)
        results_b.append(rb)

        print(f"  {version_a}={ra.quality}  {version_b}={rb.quality}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PER-TASK COMPARISON")
    print(f"{'=' * 70}")
    print(
        f"{'Task':<10} {'Prompt':<18} "
        f"{version_a:>6} {version_b:>6} {'Delta':>6}"
    )
    print("-" * 50)

    for ra, rb in zip(results_a, results_b):
        delta = rb.quality - ra.quality
        sign = "+" if delta > 0 else ""
        print(
            f"{ra.task_id:<10} {ra.prompt_name:<18} "
            f"{ra.quality:>6} {rb.quality:>6} {sign}{delta:>5}"
        )

    # ------------------------------------------------------------------
    # Aggregate scores
    # ------------------------------------------------------------------
    avg_a = sum(r.quality for r in results_a) / len(results_a)
    avg_b = sum(r.quality for r in results_b) / len(results_b)
    delta_avg = avg_b - avg_a

    print(f"\n{'=' * 70}")
    print("AGGREGATE SCORES")
    print(f"{'=' * 70}")
    print(f"  {version_a} average quality: {avg_a:.2f}/5")
    print(f"  {version_b} average quality: {avg_b:.2f}/5")
    sign = "+" if delta_avg > 0 else ""
    print(f"  Delta ({version_b} - {version_a}): {sign}{delta_avg:.2f}")

    if delta_avg > 0.2:
        print(f"\n  --> {version_b} is meaningfully better than {version_a}")
    elif delta_avg < -0.2:
        print(f"\n  --> {version_a} is meaningfully better than {version_b}")
    else:
        print(f"\n  --> No meaningful difference between versions")

    # ------------------------------------------------------------------
    # Per-category breakdown
    # ------------------------------------------------------------------
    categories = sorted(set(r.prompt_name for r in results_a))
    if len(categories) > 1:
        print(f"\n{'=' * 70}")
        print("PER-CATEGORY BREAKDOWN")
        print(f"{'=' * 70}")
        print(f"{'Category':<18} {version_a:>8} {version_b:>8} {'Delta':>8}")
        print("-" * 45)

        for cat in categories:
            cat_a = [r for r in results_a if r.prompt_name == cat]
            cat_b = [r for r in results_b if r.prompt_name == cat]
            if cat_a and cat_b:
                ca = sum(r.quality for r in cat_a) / len(cat_a)
                cb = sum(r.quality for r in cat_b) / len(cat_b)
                d = cb - ca
                s = "+" if d > 0 else ""
                print(f"{cat:<18} {ca:>8.2f} {cb:>8.2f} {s}{d:>7.2f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_path = _PROJECT_ROOT / "benchmarks" / "ab_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    payload = {
        "version_a": version_a,
        "version_b": version_b,
        "results_a": [asdict(r) for r in results_a],
        "results_b": [asdict(r) for r in results_b],
        "avg_quality_a": avg_a,
        "avg_quality_b": avg_b,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B prompt version comparison")
    parser.add_argument("--version-a", default="v1", help="First version label")
    parser.add_argument("--version-b", default="v2", help="Second version label")
    args = parser.parse_args()
    asyncio.run(main(args.version_a, args.version_b))
