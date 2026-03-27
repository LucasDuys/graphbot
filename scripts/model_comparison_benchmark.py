"""Multi-model comparison: Direct vs GraphBot Pipeline.

For each model, runs the same 10 non-tool tasks both:
  A) Direct -- single raw API call, no pipeline
  B) Pipeline -- through GraphBot Orchestrator (graph context, safety, smart routing)

Skips tool tasks (only GraphBot can use tools, unfair comparison).
Quality judged by GPT-4o-mini. Answers the question:
"Does the GraphBot pipeline add value on top of already-capable models?"

Models tested:
  - Llama 8B (free, current default)
  - Llama 70B (free tier via OpenRouter)
  - Gemini 2.5 Flash (fast, cheap)
  - GPT-4o (frontier)
  - Claude Sonnet 4 (frontier)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
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

from core_gb.orchestrator import Orchestrator
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter, DEFAULT_MODEL_MAP


# ---------------------------------------------------------------------------
# Task definitions (non-tool only -- 10 tasks)
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    category: str
    question: str
    ground_truth_hints: str


TASKS: list[Task] = [
    # Easy
    Task("easy_01", "easy", "What is the capital of Japan?", "Tokyo"),
    Task("easy_02", "easy", "What is 15 * 23?", "345"),
    Task("easy_03", "easy", "Define photosynthesis in one sentence.",
         "light,energy,plant,glucose,carbon dioxide"),
    Task("easy_04", "easy", "Who wrote '1984'?", "George Orwell"),
    Task("easy_05", "easy", "What color do you get mixing red and blue?", "purple"),
    # Hard
    Task("hard_01", "hard",
         "Compare the economic systems of Sweden, Singapore, and the United States. "
         "For each country, describe the tax rate, healthcare model, and GDP per capita. "
         "Then recommend which system would work best for a developing nation and explain why.",
         "Sweden,Singapore,United States,tax,healthcare,GDP,developing"),
    Task("hard_02", "hard",
         "Explain how a neural network learns, starting from a single neuron, "
         "building up to backpropagation, and ending with how transformers use attention. "
         "Use analogies a high school student would understand.",
         "neuron,weight,backpropagation,gradient,attention,transformer"),
    Task("hard_03", "hard",
         "A company has 3 products: A ($50, 30% margin), B ($120, 45% margin), C ($200, 20% margin). "
         "They sold 1000 units of A, 500 units of B, and 200 units of C last quarter. "
         "Calculate total revenue, total profit, profit per product, and recommend which product "
         "to focus marketing on for maximum profit growth. Show your work.",
         "revenue,profit,50000,60000,40000,27000,margin,focus,marketing"),
    Task("hard_04", "hard",
         "Write a detailed comparison of 5 sorting algorithms (bubble sort, merge sort, quick sort, "
         "heap sort, and radix sort). For each, provide: time complexity (best, average, worst), "
         "space complexity, stability, and a one-line description of when to use it. Format as a table.",
         "O(n),O(n log n),O(n^2),stable,unstable,in-place,merge,quick,heap,radix,bubble"),
    Task("hard_05", "hard",
         "Trace the journey of a HTTP request from typing 'google.com' in a browser to seeing "
         "the page rendered. Include: DNS resolution, TCP handshake, TLS negotiation, HTTP request, "
         "server processing, response, and browser rendering. Be specific about each step.",
         "DNS,TCP,TLS,SYN,ACK,certificate,HTTP,GET,HTML,DOM,render,paint"),
]


# ---------------------------------------------------------------------------
# Models to test
# ---------------------------------------------------------------------------

MODELS: list[tuple[str, str]] = [
    ("Llama 8B", "meta-llama/llama-3.1-8b-instruct"),
    ("Llama 70B", "meta-llama/llama-3.3-70b-instruct"),
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash-preview"),
    ("GPT-4o", "openai/gpt-4o"),
    ("Claude Sonnet 4", "anthropic/claude-sonnet-4"),
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class Result:
    task_id: str
    category: str
    question: str
    model_name: str
    model_id: str
    mode: str  # "direct" or "pipeline"
    output: str
    quality: int
    judge_reasoning: str
    tokens: int
    cost: float
    latency_s: float
    success: bool


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

async def judge_quality(
    provider: OpenRouterProvider,
    question: str,
    answer: str,
    ground_truth_hints: str,
) -> tuple[int, str]:
    """Use GPT-4o-mini as judge. Returns (score 1-5, reasoning)."""
    prompt = f"""You are an impartial answer quality judge. Rate the following answer on a 1-5 scale.

QUESTION: {question}

ANSWER TO JUDGE:
{answer[:2000]}

GROUND TRUTH HINTS (keywords/facts the answer should ideally contain): {ground_truth_hints}

SCORING:
1 = Wrong, irrelevant, or refuses to answer
2 = Partially correct but major gaps or errors
3 = Mostly correct but missing important details
4 = Good answer, covers the key points
5 = Excellent, thorough, accurate, well-structured

Respond with ONLY a JSON object: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""

    try:
        result = await provider.complete(
            [{"role": "user", "content": prompt}],
            "openai/gpt-4o-mini",
        )
        data = json.loads(result.content)
        return int(data["score"]), str(data["reasoning"])
    except Exception:
        try:
            text = result.content  # type: ignore[possibly-undefined]
            for s in range(5, 0, -1):
                if str(s) in text:
                    return s, "extracted from response"
        except Exception:
            pass
        return 3, "judge failed, default score"


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

async def call_direct(
    provider: OpenRouterProvider, model: str, question: str
) -> tuple[str, int, float, float]:
    """Raw API call. Returns (output, tokens, cost, latency_s)."""
    start = time.time()
    try:
        result = await provider.complete(
            [{"role": "user", "content": question}],
            model,
        )
        latency = time.time() - start
        return result.content, result.tokens_in + result.tokens_out, result.cost, latency
    except Exception as exc:
        latency = time.time() - start
        return f"ERROR: {exc}", 0, 0.0, latency


async def call_pipeline(
    orchestrator: Orchestrator, question: str
) -> tuple[str, int, float, float, bool]:
    """Through GraphBot pipeline. Returns (output, tokens, cost, latency_s, success)."""
    start = time.time()
    try:
        result = await orchestrator.process(question)
        latency = time.time() - start
        return result.output, result.total_tokens, result.total_cost, latency, result.success
    except Exception as exc:
        latency = time.time() - start
        return f"ERROR: {exc}", 0, 0.0, latency, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("Multi-Model Comparison: Direct vs GraphBot Pipeline")
    print("=" * 78)
    print(f"Models: {len(MODELS)} | Tasks: {len(TASKS)} (easy + hard, no tools)")
    print(f"Configs per model: direct + pipeline = {len(MODELS) * 2} runs")
    print(f"Total API calls: ~{len(MODELS) * len(TASKS) * 2} + judging")
    print("=" * 78)

    provider = OpenRouterProvider()
    results: list[Result] = []

    for model_name, model_id in MODELS:
        print(f"\n{'=' * 78}")
        print(f"Model: {model_name} ({model_id})")
        print(f"{'=' * 78}")

        # --- Direct mode ---
        print(f"\n  --- Direct (raw API call) ---")
        for i, task in enumerate(TASKS, 1):
            print(f"  [{i:2d}/{len(TASKS)}] [{task.category:4s}] {task.question[:55]}...", end="", flush=True)

            output, tokens, cost, latency = await call_direct(provider, model_id, task.question)
            success = not output.startswith("ERROR")

            quality, reasoning = await judge_quality(
                provider, task.question, output, task.ground_truth_hints,
            )

            results.append(Result(
                task_id=task.id, category=task.category, question=task.question,
                model_name=model_name, model_id=model_id, mode="direct",
                output=output[:500], quality=quality, judge_reasoning=reasoning,
                tokens=tokens, cost=cost, latency_s=latency, success=success,
            ))
            print(f" q={quality} | {tokens:5d}tok | ${cost:.4f} | {latency:.1f}s")

        # --- Pipeline mode ---
        # Create an orchestrator that uses this model for all complexity levels
        print(f"\n  --- Pipeline (GraphBot + {model_name}) ---")
        db_path = str(_PROJECT_ROOT / "data" / f"bench_{model_name.lower().replace(' ', '_')}.db")
        store = GraphStore(db_path=db_path)
        store.initialize()

        # Override model map: all complexity levels use this model
        custom_map = {k: model_id for k in range(1, 6)}
        router = ModelRouter(provider=provider, model_map=custom_map)
        orchestrator = Orchestrator(store, router)

        for i, task in enumerate(TASKS, 1):
            print(f"  [{i:2d}/{len(TASKS)}] [{task.category:4s}] {task.question[:55]}...", end="", flush=True)

            output, tokens, cost, latency, success = await call_pipeline(orchestrator, task.question)

            quality, reasoning = await judge_quality(
                provider, task.question, output, task.ground_truth_hints,
            )

            results.append(Result(
                task_id=task.id, category=task.category, question=task.question,
                model_name=model_name, model_id=model_id, mode="pipeline",
                output=output[:500], quality=quality, judge_reasoning=reasoning,
                tokens=tokens, cost=cost, latency_s=latency, success=success,
            ))
            print(f" q={quality} | {tokens:5d}tok | ${cost:.4f} | {latency:.1f}s")

        store.close()

    # ------------------------------------------------------------------
    # Results table: Direct vs Pipeline per model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("RESULTS: DIRECT vs PIPELINE PER MODEL")
    print(f"{'=' * 78}")
    print(
        f"{'Model':<20} {'Mode':<10} {'Easy':>6} {'Hard':>6} {'All':>6} "
        f"{'Cost':>10} {'Latency':>8}"
    )
    print("-" * 78)

    for model_name, _ in MODELS:
        for mode in ("direct", "pipeline"):
            mode_results = [r for r in results if r.model_name == model_name and r.mode == mode]
            if not mode_results:
                continue

            easy = [r for r in mode_results if r.category == "easy"]
            hard = [r for r in mode_results if r.category == "hard"]

            easy_q = sum(r.quality for r in easy) / len(easy) if easy else 0
            hard_q = sum(r.quality for r in hard) / len(hard) if hard else 0
            all_q = sum(r.quality for r in mode_results) / len(mode_results)
            total_cost = sum(r.cost for r in mode_results)
            avg_lat = sum(r.latency_s for r in mode_results) / len(mode_results)

            label = model_name if mode == "direct" else ""
            print(
                f"{label:<20} {mode:<10} {easy_q:>6.2f} {hard_q:>6.2f} {all_q:>6.2f} "
                f"${total_cost:>9.4f} {avg_lat:>7.1f}s"
            )
        print()

    # ------------------------------------------------------------------
    # Pipeline delta (pipeline quality - direct quality)
    # ------------------------------------------------------------------
    print(f"{'=' * 78}")
    print("PIPELINE DELTA (pipeline - direct)")
    print(f"{'=' * 78}")
    print(f"{'Model':<20} {'Easy':>8} {'Hard':>8} {'Overall':>8} {'Verdict':<30}")
    print("-" * 78)

    for model_name, _ in MODELS:
        direct = [r for r in results if r.model_name == model_name and r.mode == "direct"]
        pipeline = [r for r in results if r.model_name == model_name and r.mode == "pipeline"]
        if not direct or not pipeline:
            continue

        d_easy = [r for r in direct if r.category == "easy"]
        p_easy = [r for r in pipeline if r.category == "easy"]
        d_hard = [r for r in direct if r.category == "hard"]
        p_hard = [r for r in pipeline if r.category == "hard"]

        delta_easy = (sum(r.quality for r in p_easy) / len(p_easy)) - (sum(r.quality for r in d_easy) / len(d_easy)) if d_easy and p_easy else 0
        delta_hard = (sum(r.quality for r in p_hard) / len(p_hard)) - (sum(r.quality for r in d_hard) / len(d_hard)) if d_hard and p_hard else 0
        delta_all = (sum(r.quality for r in pipeline) / len(pipeline)) - (sum(r.quality for r in direct) / len(direct))

        if delta_all > 0.3:
            verdict = "Pipeline helps significantly"
        elif delta_all > 0:
            verdict = "Pipeline helps slightly"
        elif delta_all > -0.3:
            verdict = "Roughly equal"
        else:
            verdict = "Pipeline hurts -- overhead costs"

        print(
            f"{model_name:<20} {delta_easy:>+8.2f} {delta_hard:>+8.2f} {delta_all:>+8.2f} {verdict:<30}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = _PROJECT_ROOT / "benchmarks" / "model_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
