# Thesis Validation Guide

Step-by-step instructions for running the full GraphBot thesis validation
benchmark and blind evaluation.

## Overview

The validation pipeline has two stages:

1. **Thesis validation** (`scripts/validate_thesis.py`) -- Runs 30 tasks across
   four configurations (Llama 8B direct, Llama 8B pipeline, Llama 70B direct,
   GPT-4o direct) and records quality, tokens, cost, and latency.
2. **Blind evaluation** (`scripts/blind_eval.py`) -- Takes the outputs from
   step 1, creates randomized pairwise comparisons, and sends them to an
   LLM judge for unbiased quality assessment.

A convenience script (`scripts/run_full_validation.py`) runs both in sequence.

---

## Prerequisites

### API Keys

You need an OpenRouter API key to call the LLM models. Create a `.env.local`
file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
```

The scripts load this file automatically at startup. No other environment
variable configuration is required for the default setup.

For dry-run mode (mock providers, no API calls), no API key is needed.

### Python Environment

Install the project dependencies (assumes you already have a virtual environment):

```bash
pip install -r requirements.txt
```

Verify the graph database is available:

```bash
python -c "from graph.store import GraphStore; print('OK')"
```

### Disk Space

The validation creates a temporary Kuzu database at `data/thesis_validation.db`
(typically under 50 MB). Results are saved as JSON in `benchmarks/`.

---

## Step 1: Run the Thesis Validation

### Dry Run (recommended first)

A dry run uses mock providers that return deterministic placeholder responses.
No API calls are made and no costs are incurred.

```bash
python scripts/validate_thesis.py --dry-run
```

This verifies the full pipeline works end-to-end and produces output at
`benchmarks/thesis_validation.json`.

### Full Run

```bash
python scripts/validate_thesis.py
```

This runs all 30 tasks across all four configurations using real LLM APIs.
Expect approximately 120 API calls (30 tasks x 4 configurations). The pipeline
configuration uses the GraphBot orchestrator, so those calls may generate
additional sub-calls depending on task decomposition.

**Estimated time:** 5-15 minutes depending on API latency.
**Estimated cost:** Under $1 USD (dominated by GPT-4o calls).

### Single Task

To test a single task before running the full suite:

```bash
python scripts/validate_thesis.py --task qa_01
python scripts/validate_thesis.py --task qa_01 --dry-run
```

### Output

Results are saved to `benchmarks/thesis_validation.json` with this structure:

```json
{
  "timestamp": "2026-03-24T16:12:54.517615",
  "task_count": 30,
  "configuration_count": 4,
  "configurations": [...],
  "results": [
    {
      "config_id": "llama8b_direct",
      "task_id": "qa_01",
      "quality": 5,
      "tokens": 160,
      "cost": 0.0,
      "latency_ms": 50.0,
      "success": true,
      "keyword_hits": 1,
      "keyword_total": 1
    }
  ],
  "summaries": {
    "llama8b_direct": {
      "avg_quality": 4.5,
      "avg_tokens": 160.0,
      "success_rate": 1.0,
      "token_reduction_vs_gpt4o_pct": -45.5
    }
  }
}
```

A summary table is also printed to the console showing per-configuration
averages and the key thesis metrics (quality lift, quality gap, cost ratio).

---

## Step 2: Run the Blind Evaluation

The blind evaluation reads outputs from `benchmarks/thesis_validation.json`
(auto-detected if present) and runs pairwise LLM-as-judge comparisons.

### Dry Run

```bash
python scripts/blind_eval.py --dry-run
```

In dry-run mode, if `benchmarks/thesis_validation.json` exists it will be
loaded; otherwise fake task data is generated automatically. The judge uses a
mock that decides winners based on answer length with some randomness.

### Full Run

```bash
python scripts/blind_eval.py
```

This sends pairwise comparisons to the judge model (default:
`anthropic/claude-sonnet-4-20250514`). Two comparison pairs are evaluated for
each task:

- **8B+GraphBot vs 70B direct** -- Does the pipeline close the parameter gap?
- **8B+GraphBot vs GPT-4o direct** -- Does the pipeline match frontier quality?

### Options

```bash
# Use a different judge model
python scripts/blind_eval.py --judge-model openai/gpt-4o

# Set a random seed for reproducible blinding order
python scripts/blind_eval.py --seed 42

# Combine options
python scripts/blind_eval.py --dry-run --seed 42
```

### Output

Results are saved to `benchmarks/blind_eval.json`:

```json
{
  "timestamp": "...",
  "judge_model": "anthropic/claude-sonnet-4-20250514",
  "dry_run": false,
  "task_count": 30,
  "verdict_count": 60,
  "verdicts": [
    {
      "task_id": "qa_01",
      "pair_label": "8B+GraphBot vs 70B",
      "winner_system": "8b_graphbot",
      "reasoning": "..."
    }
  ],
  "summaries": {
    "8B+GraphBot vs 70B": {
      "wins_a": 18,
      "wins_b": 8,
      "ties": 4,
      "total": 30,
      "win_rate_a": 60.0,
      "win_rate_b": 26.7,
      "tie_rate": 13.3
    }
  }
}
```

---

## Step 3: Interpret the Results

### Thesis Validation Metrics

The summary table from Step 1 shows four key columns:

| Metric | What it means |
|--------|---------------|
| **Avg Quality** | Mean heuristic score (1-5) across 30 tasks. Higher is better. |
| **Avg Tokens** | Mean total tokens (in + out) per task. Lower means more efficient. |
| **Avg Cost** | Mean USD cost per task. Free models show $0. |
| **Token Reduction vs GPT-4o** | How many fewer tokens this config uses compared to GPT-4o. Positive = more efficient. |

**Key thesis metrics** (printed below the table):

- **Quality lift (8B pipeline vs 8B direct):** The improvement that the GraphBot
  pipeline adds to the base 8B model. A positive number supports the thesis.
  Target: +0.5 or higher on the 1-5 scale.
- **Quality gap (8B pipeline vs GPT-4o):** How close the pipeline gets to
  frontier quality. A number near 0 or positive means the pipeline matches or
  exceeds GPT-4o. Target: within -0.5.
- **Cost ratio (8B pipeline / GPT-4o):** What fraction of GPT-4o's cost the
  pipeline uses. Lower is better. Target: under 10%.

### Blind Evaluation Metrics

The blind eval provides the most rigorous quality comparison:

- **Win rate for 8b_graphbot vs 70b_direct:** If >50%, the pipeline with an 8B
  model outperforms a 70B model used directly. This directly validates the core
  thesis that context beats parameters.
- **Win rate for 8b_graphbot vs gpt4o_direct:** If >40%, the pipeline achieves
  competitive quality at a fraction of the cost.
- **Tie rate:** High tie rates (>30%) suggest the systems are comparable, which
  still supports the thesis given the cost difference.

### What Good Results Look Like

For a successful thesis defense, aim for:

1. Quality lift of +0.5 or more (8B pipeline vs 8B direct)
2. Quality gap within -0.5 (8B pipeline vs GPT-4o)
3. Cost under 10% of GPT-4o
4. Blind eval win rate >50% against 70B direct
5. Blind eval win rate >35% against GPT-4o direct (with ties counting as draws)

### Where to Find Results

| File | Description |
|------|-------------|
| `benchmarks/thesis_validation.json` | Raw validation data with per-task scores |
| `benchmarks/blind_eval.json` | Blind evaluation verdicts and summaries |
| `data/thesis_validation.db` | Kuzu database used during validation (can be deleted) |

---

## Convenience: Run Everything at Once

The `run_full_validation.py` script runs both steps in sequence:

```bash
# Dry run (no API calls)
python scripts/run_full_validation.py --dry-run

# Full run
python scripts/run_full_validation.py

# Full run with custom judge and seed
python scripts/run_full_validation.py --judge-model openai/gpt-4o --seed 42
```

This script:
1. Runs `validate_thesis.py` (produces `benchmarks/thesis_validation.json`)
2. Runs `blind_eval.py` (reads the output from step 1, produces `benchmarks/blind_eval.json`)
3. Prints a combined summary

All arguments are forwarded to both scripts where applicable (`--dry-run` is
shared; `--judge-model` and `--seed` apply only to the blind eval step).
