"""Thesis validation v2 -- tests where GraphBot SHOULD matter.

The v1 benchmark tested easy tasks that any model handles fine.
This version tests tasks specifically designed to expose the gap
between a raw 8B model and the same model with GraphBot's pipeline.

Three categories:
1. EASY -- 8B should handle fine alone (baseline, expect no improvement)
2. HARD -- 8B alone should struggle, GraphBot should help
3. TOOL-DEPENDENT -- impossible without tools, only GraphBot can do them

Each task is run on:
  A) Llama 8B direct (single call, no pipeline)
  B) Llama 8B + GraphBot pipeline (decomposition, graph context, tools, verification)
  C) GPT-4o direct (single call, the "expensive" baseline)

Quality is judged by GPT-4o-mini as an impartial judge (not keyword heuristic).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
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
from models.router import ModelRouter


@dataclass
class Task:
    id: str
    category: str  # easy, hard, tool
    question: str
    difficulty: str  # what makes this hard for raw 8B
    ground_truth_hints: str  # keywords/facts the answer should contain


@dataclass
class Result:
    task_id: str
    category: str
    question: str
    config: str  # 8b_direct, 8b_graphbot, gpt4o_direct
    output: str
    quality: int  # 1-5 from judge
    judge_reasoning: str
    tokens: int
    cost: float
    latency_s: float
    success: bool


TASKS: list[Task] = [
    # --- EASY: 8B handles fine, GraphBot shouldn't hurt ---
    Task("easy_01", "easy", "What is the capital of Japan?",
         "trivial factual recall", "Tokyo"),
    Task("easy_02", "easy", "What is 15 * 23?",
         "simple arithmetic", "345"),
    Task("easy_03", "easy", "Define photosynthesis in one sentence.",
         "basic definition", "light,energy,plant,glucose,carbon dioxide"),
    Task("easy_04", "easy", "Who wrote '1984'?",
         "trivial factual recall", "George Orwell"),
    Task("easy_05", "easy", "What color do you get mixing red and blue?",
         "trivial knowledge", "purple"),

    # --- HARD: 8B alone struggles, decomposition + context should help ---
    Task("hard_01", "hard",
         "Compare the economic systems of Sweden, Singapore, and the United States. "
         "For each country, describe the tax rate, healthcare model, and GDP per capita. "
         "Then recommend which system would work best for a developing nation and explain why.",
         "multi-entity comparison requiring structured knowledge across 3 countries x 3 dimensions + synthesis",
         "Sweden,Singapore,United States,tax,healthcare,GDP,developing"),
    Task("hard_02", "hard",
         "Explain how a neural network learns, starting from a single neuron, "
         "building up to backpropagation, and ending with how transformers use attention. "
         "Use analogies a high school student would understand.",
         "multi-level explanation requiring pedagogical scaffolding across 3 abstraction levels",
         "neuron,weight,backpropagation,gradient,attention,transformer"),
    Task("hard_03", "hard",
         "A company has 3 products: A ($50, 30% margin), B ($120, 45% margin), C ($200, 20% margin). "
         "They sold 1000 units of A, 500 units of B, and 200 units of C last quarter. "
         "Calculate total revenue, total profit, profit per product, and recommend which product "
         "to focus marketing on for maximum profit growth. Show your work.",
         "multi-step math with business reasoning requiring structured calculation then strategic recommendation",
         "revenue,profit,50000,60000,40000,27000,margin,focus,marketing"),
    Task("hard_04", "hard",
         "Write a detailed comparison of 5 sorting algorithms (bubble sort, merge sort, quick sort, "
         "heap sort, and radix sort). For each, provide: time complexity (best, average, worst), "
         "space complexity, stability, and a one-line description of when to use it. "
         "Format as a table.",
         "structured technical comparison requiring precise knowledge across 5 items x 5 dimensions",
         "O(n),O(n log n),O(n^2),stable,unstable,in-place,merge,quick,heap,radix,bubble"),
    Task("hard_05", "hard",
         "Trace the journey of a HTTP request from typing 'google.com' in a browser to seeing "
         "the page rendered. Include: DNS resolution, TCP handshake, TLS negotiation, HTTP request, "
         "server processing, response, and browser rendering. Be specific about each step.",
         "sequential multi-step technical explanation requiring 7+ distinct technical concepts in order",
         "DNS,TCP,TLS,SYN,ACK,certificate,HTTP,GET,HTML,DOM,render,paint"),

    # --- TOOL-DEPENDENT: impossible without tools, only GraphBot can do ---
    Task("tool_01", "tool",
         "What Python files are in the scripts/ directory of this project? List them all.",
         "requires file system access", "validate_thesis,run_capability,stress_test,blind_eval"),
    Task("tool_02", "tool",
         "Run 'python --version' and tell me exactly what Python version is installed.",
         "requires shell execution", "Python 3.13"),
    Task("tool_03", "tool",
         "Search the web for 'Kuzu graph database latest version 2026' and tell me what you find.",
         "requires web search", "Kuzu,graph,database"),
    Task("tool_04", "tool",
         "Read the first 10 lines of pyproject.toml in this project and tell me the project name and version.",
         "requires file reading", "graphbot,pyproject,name,version"),
    Task("tool_05", "tool",
         "Run 'git log --oneline -5' and summarize what the last 5 commits changed.",
         "requires shell execution + interpretation", "commit,git"),
]


async def call_direct(provider: OpenRouterProvider, model: str, question: str) -> tuple[str, int, float, float]:
    """Make a single direct LLM call, return (output, tokens, cost, latency_s)."""
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


async def call_pipeline(orchestrator: Orchestrator, question: str) -> tuple[str, int, float, float, bool]:
    """Run through GraphBot pipeline, return (output, tokens, cost, latency_s, success)."""
    start = time.time()
    try:
        result = await orchestrator.process(question)
        latency = time.time() - start
        return result.output, result.total_tokens, result.total_cost, latency, result.success
    except Exception as exc:
        latency = time.time() - start
        return f"ERROR: {exc}", 0, 0.0, latency, False


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
        # Fallback: try to extract score from response
        try:
            text = result.content
            for s in range(5, 0, -1):
                if str(s) in text:
                    return s, "extracted from response"
        except Exception:
            pass
        return 3, "judge failed, default score"


async def main() -> None:
    print("Thesis Validation v2 -- Testing where GraphBot SHOULD matter")
    print("=" * 70)

    db_path = str(_PROJECT_ROOT / "data" / "thesis_v2.db")
    store = GraphStore(db_path)
    store.initialize()

    provider = OpenRouterProvider()
    router = ModelRouter(provider=provider)
    orchestrator = Orchestrator(store, router)

    results: list[Result] = []

    configs = [
        ("8b_direct", "Llama 8B Direct"),
        ("8b_graphbot", "Llama 8B + GraphBot"),
        ("gpt4o_direct", "GPT-4o Direct"),
    ]

    for config_id, config_label in configs:
        print(f"\n{'=' * 70}")
        print(f"Configuration: {config_label}")
        print(f"{'=' * 70}")

        for i, task in enumerate(TASKS, 1):
            print(f"  [{i:2d}/{len(TASKS)}] [{task.category:4s}] {task.question[:60]}...", end="", flush=True)

            if config_id == "8b_direct":
                output, tokens, cost, latency = await call_direct(
                    provider, "meta-llama/llama-3.1-8b-instruct", task.question
                )
                success = not output.startswith("ERROR")
            elif config_id == "8b_graphbot":
                output, tokens, cost, latency, success = await call_pipeline(
                    orchestrator, task.question
                )
            elif config_id == "gpt4o_direct":
                output, tokens, cost, latency = await call_direct(
                    provider, "openai/gpt-4o", task.question
                )
                success = not output.startswith("ERROR")
            else:
                continue

            # Judge quality
            quality, reasoning = await judge_quality(
                provider, task.question, output, task.ground_truth_hints
            )

            result = Result(
                task_id=task.id,
                category=task.category,
                question=task.question,
                config=config_id,
                output=output[:1000],
                quality=quality,
                judge_reasoning=reasoning,
                tokens=tokens,
                cost=cost,
                latency_s=latency,
                success=success,
            )
            results.append(result)

            print(f" q={quality} | {tokens:5d}tok | ${cost:.6f} | {latency:.1f}s")

    # Compute summaries by category x config
    print(f"\n{'=' * 70}")
    print("RESULTS BY CATEGORY")
    print(f"{'=' * 70}")

    for category in ["easy", "hard", "tool"]:
        print(f"\n--- {category.upper()} TASKS ---")
        print(f"{'Config':<25} {'Quality':>8} {'Tokens':>8} {'Cost':>12} {'Latency':>10} {'Success':>8}")
        print("-" * 75)

        for config_id, config_label in configs:
            cat_results = [r for r in results if r.category == category and r.config == config_id]
            if not cat_results:
                continue
            avg_q = sum(r.quality for r in cat_results) / len(cat_results)
            avg_tok = sum(r.tokens for r in cat_results) / len(cat_results)
            avg_cost = sum(r.cost for r in cat_results) / len(cat_results)
            avg_lat = sum(r.latency_s for r in cat_results) / len(cat_results)
            success_rate = sum(1 for r in cat_results if r.success) / len(cat_results) * 100
            print(f"{config_label:<25} {avg_q:>8.2f} {avg_tok:>8.0f} ${avg_cost:>11.6f} {avg_lat:>9.1f}s {success_rate:>7.0f}%")

    # Overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<25} {'Quality':>8} {'Tokens':>8} {'Cost':>12} {'Latency':>10}")
    print("-" * 65)

    for config_id, config_label in configs:
        cfg_results = [r for r in results if r.config == config_id]
        avg_q = sum(r.quality for r in cfg_results) / len(cfg_results)
        total_tok = sum(r.tokens for r in cfg_results)
        total_cost = sum(r.cost for r in cfg_results)
        avg_lat = sum(r.latency_s for r in cfg_results) / len(cfg_results)
        print(f"{config_label:<25} {avg_q:>8.2f} {total_tok:>8} ${total_cost:>11.6f} {avg_lat:>9.1f}s")

    # Key thesis metrics
    hard_8b = [r for r in results if r.category == "hard" and r.config == "8b_direct"]
    hard_gb = [r for r in results if r.category == "hard" and r.config == "8b_graphbot"]
    hard_4o = [r for r in results if r.category == "hard" and r.config == "gpt4o_direct"]
    tool_8b = [r for r in results if r.category == "tool" and r.config == "8b_direct"]
    tool_gb = [r for r in results if r.category == "tool" and r.config == "8b_graphbot"]

    if hard_8b and hard_gb and hard_4o:
        q_8b = sum(r.quality for r in hard_8b) / len(hard_8b)
        q_gb = sum(r.quality for r in hard_gb) / len(hard_gb)
        q_4o = sum(r.quality for r in hard_4o) / len(hard_4o)
        c_gb = sum(r.cost for r in hard_gb) / len(hard_gb)
        c_4o = sum(r.cost for r in hard_4o) / len(hard_4o)

        print(f"\n{'=' * 70}")
        print("KEY THESIS METRICS (hard tasks only)")
        print(f"{'=' * 70}")
        print(f"  8B direct quality:        {q_8b:.2f}/5")
        print(f"  8B + GraphBot quality:    {q_gb:.2f}/5  ({'+' if q_gb >= q_8b else ''}{q_gb - q_8b:.2f} vs 8B direct)")
        print(f"  GPT-4o quality:           {q_4o:.2f}/5")
        print(f"  Quality gap (GB vs 4o):   {q_gb - q_4o:+.2f}")
        if c_4o > 0:
            print(f"  Cost ratio (GB / 4o):     {c_gb / c_4o * 100:.1f}%")

    if tool_8b and tool_gb:
        tool_8b_success = sum(1 for r in tool_8b if r.success and r.quality >= 3) / len(tool_8b) * 100
        tool_gb_success = sum(1 for r in tool_gb if r.success and r.quality >= 3) / len(tool_gb) * 100
        print(f"\n  Tool tasks (8B direct):   {tool_8b_success:.0f}% usable answers")
        print(f"  Tool tasks (8B+GraphBot): {tool_gb_success:.0f}% usable answers")

    # Save results
    output_path = _PROJECT_ROOT / "benchmarks" / "thesis_validation_v2.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
