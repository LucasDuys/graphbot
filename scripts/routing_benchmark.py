"""Routing benchmark: measure model selection for all 15 benchmark tasks.

Runs IntakeParser.parse() -> select_model() for each task and prints a table
showing the routing decision, selected provider/model, and timing. No LLM
calls are made -- this is a pure computation benchmark.

Usage:
    python scripts/routing_benchmark.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path for imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core_gb.intake import IntakeParser
from core_gb.types import Domain
from models.smart_router import (
    MODEL_GEMINI_FLASH,
    MODEL_LLAMA_8B,
    MODEL_LLAMA_70B,
    MODEL_QWEN3_32B,
    ModelSelection,
    select_model,
)


# ---------------------------------------------------------------------------
# Benchmark task definitions (same 15 from validate_single_call.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTask:
    """A single benchmark task for routing evaluation."""

    id: str
    category: str
    question: str


BENCHMARK_TASKS: tuple[BenchmarkTask, ...] = (
    # Easy tasks
    BenchmarkTask("easy_01", "easy", "What is the capital of Japan?"),
    BenchmarkTask("easy_02", "easy", "What is 15 * 23?"),
    BenchmarkTask("easy_03", "easy", "Define photosynthesis in one sentence."),
    BenchmarkTask("easy_04", "easy", "Who wrote '1984'?"),
    BenchmarkTask("easy_05", "easy", "What color do you get mixing red and blue?"),
    # Hard tasks
    BenchmarkTask(
        "hard_01", "hard",
        "Compare the economic systems of Sweden, Singapore, and the United States. "
        "For each country, describe the tax rate, healthcare model, and GDP per capita. "
        "Then recommend which system would work best for a developing nation and explain why.",
    ),
    BenchmarkTask(
        "hard_02", "hard",
        "Explain how a neural network learns, starting from a single neuron, "
        "building up to backpropagation, and ending with how transformers use attention. "
        "Use analogies a high school student would understand.",
    ),
    BenchmarkTask(
        "hard_03", "hard",
        "A company has 3 products: A ($50, 30% margin), B ($120, 45% margin), C ($200, 20% margin). "
        "They sold 1000 units of A, 500 units of B, and 200 units of C last quarter. "
        "Calculate total revenue, total profit, profit per product, and recommend which product "
        "to focus marketing on for maximum profit growth. Show your work.",
    ),
    BenchmarkTask(
        "hard_04", "hard",
        "Write a detailed comparison of 5 sorting algorithms (bubble sort, merge sort, quick sort, "
        "heap sort, and radix sort). For each, provide: time complexity (best, average, worst), "
        "space complexity, stability, and a one-line description of when to use it. "
        "Format as a table.",
    ),
    BenchmarkTask(
        "hard_05", "hard",
        "Trace the journey of a HTTP request from typing 'google.com' in a browser to seeing "
        "the page rendered. Include: DNS resolution, TCP handshake, TLS negotiation, HTTP request, "
        "server processing, response, and browser rendering. Be specific about each step.",
    ),
    # Tool-dependent tasks
    BenchmarkTask(
        "tool_01", "tool",
        "What Python files are in the scripts/ directory of this project? List them all.",
    ),
    BenchmarkTask(
        "tool_02", "tool",
        "Run 'python --version' and tell me exactly what Python version is installed.",
    ),
    BenchmarkTask(
        "tool_03", "tool",
        "Search the web for 'Kuzu graph database latest version 2026' and tell me what you find.",
    ),
    BenchmarkTask(
        "tool_04", "tool",
        "Read the first 10 lines of pyproject.toml in this project and tell me the project name and version.",
    ),
    BenchmarkTask(
        "tool_05", "tool",
        "Run 'git log --oneline -5' and summarize what the last 5 commits changed.",
    ),
)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingResult:
    """Result of a single routing decision."""

    task_id: str
    category: str
    domain: Domain
    complexity: int
    provider: str
    model: str
    decision_us: float  # microseconds


def _short_model(model: str) -> str:
    """Abbreviate model name for table display."""
    aliases: dict[str, str] = {
        MODEL_LLAMA_8B: "Llama-8B",
        MODEL_LLAMA_70B: "Llama-70B",
        MODEL_QWEN3_32B: "Qwen3-32B",
        MODEL_GEMINI_FLASH: "Gemini-Flash",
    }
    return aliases.get(model, model)


def run_benchmark() -> list[RoutingResult]:
    """Run the routing benchmark for all 15 tasks and return results."""
    parser = IntakeParser()
    results: list[RoutingResult] = []

    for task in BENCHMARK_TASKS:
        start = time.perf_counter()
        intake = parser.parse(task.question)
        selection = select_model(intake.domain, intake.complexity)
        elapsed_us = (time.perf_counter() - start) * 1_000_000

        results.append(RoutingResult(
            task_id=task.id,
            category=task.category,
            domain=intake.domain,
            complexity=intake.complexity,
            provider=selection.provider,
            model=selection.model,
            decision_us=elapsed_us,
        ))

    return results


def print_table(results: list[RoutingResult]) -> None:
    """Print a formatted table of routing results."""
    header = (
        f"{'Task':<10} {'Cat':<6} {'Domain':<12} {'Cmplx':>5} "
        f"{'Provider':<14} {'Model':<16} {'Time (us)':>10}"
    )
    separator = "-" * len(header)

    print("\nRouting Benchmark Results")
    print("=" * len(header))
    print(header)
    print(separator)

    for r in results:
        print(
            f"{r.task_id:<10} {r.category:<6} {r.domain.value:<12} {r.complexity:>5} "
            f"{r.provider:<14} {_short_model(r.model):<16} {r.decision_us:>10.1f}"
        )

    print(separator)


def print_summary(results: list[RoutingResult]) -> None:
    """Print summary statistics."""
    times = [r.decision_us for r in results]
    avg_us = sum(times) / len(times)
    max_us = max(times)
    min_us = min(times)
    total_us = sum(times)

    print("\nTiming Summary")
    print("-" * 40)
    print(f"  Total decisions:    {len(results)}")
    print(f"  Total time:         {total_us:.1f} us ({total_us / 1000:.3f} ms)")
    print(f"  Average per task:   {avg_us:.1f} us ({avg_us / 1000:.4f} ms)")
    print(f"  Min:                {min_us:.1f} us")
    print(f"  Max:                {max_us:.1f} us")

    # Check against <1ms per decision target
    if avg_us < 1000:
        print(f"  Target (<1ms avg):  PASS")
    else:
        print(f"  Target (<1ms avg):  FAIL ({avg_us / 1000:.3f} ms)")

    # Model distribution
    model_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    for r in results:
        short = _short_model(r.model)
        model_counts[short] = model_counts.get(short, 0) + 1
        provider_counts[r.provider] = provider_counts.get(r.provider, 0) + 1

    print("\nModel Distribution")
    print("-" * 40)
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "#" * int(pct / 2)
        print(f"  {model:<16} {count:>3} ({pct:5.1f}%) {bar}")

    print("\nProvider Distribution")
    print("-" * 40)
    for provider, count in sorted(provider_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "#" * int(pct / 2)
        print(f"  {provider:<14} {count:>3} ({pct:5.1f}%) {bar}")

    # Category breakdown
    print("\nCategory -> Model Mapping")
    print("-" * 40)
    for category in ["easy", "hard", "tool"]:
        cat_results = [r for r in results if r.category == category]
        models_used = set(_short_model(r.model) for r in cat_results)
        print(f"  {category:<6} -> {', '.join(sorted(models_used))}")


def main() -> None:
    """Run the routing benchmark and print results."""
    print("GraphBot Smart Model Routing Benchmark")
    print("No LLM calls -- pure computation timing\n")

    results = run_benchmark()
    print_table(results)
    print_summary(results)


if __name__ == "__main__":
    main()
