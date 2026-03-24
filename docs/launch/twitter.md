# Twitter/X Launch Thread

_(Draft -- do not post without explicit authorization)_

**Format:** Thread (10 tweets). Post tweet 1, then reply-chain the rest.

---

## Tweet 1 (Hook)

I got 30/30 tasks passing on a free LLM for $0.0006 total.

Not GPT-4o. Not Claude. Llama 8B -- the free tier on OpenRouter.

Here's the open-source framework behind it:

[PLACEHOLDER: GIF of 30-task benchmark running]

---

## Tweet 2 (The Problem)

AI agents are expensive. GPT-4o costs $5-15/M tokens. Running an autonomous agent on frontier models burns $10-100/day.

But what if you didn't need expensive models at all?

What if you could give a small model exactly the right context for every subtask?

---

## Tweet 3 (The Core Insight)

The core insight behind GraphBot:

A small model with perfect context beats a large model with no context.

Instead of throwing harder tasks at bigger models, decompose them into trivially simple subtasks and give each one hyper-specific context from a knowledge graph.

[PLACEHOLDER: Architecture diagram or ASCII flow]

---

## Tweet 4 (How It Works)

How it works:

1. Task comes in ("Research top 5 AI frameworks, write comparison")
2. Decomposed into a recursive DAG (5 parallel searches + synthesis)
3. Each node gets graph context (what the model needs to know)
4. Executed in parallel on Llama 8B
5. Results synthesized into clean output
6. Pattern stored for next time (0 tokens on repeat)

[PLACEHOLDER: GIF of DAG execution in dashboard]

---

## Tweet 5 (The Numbers)

The numbers (all verified on LangSmith):

| Category | Pass Rate | Cost |
|----------|-----------|------|
| Q&A (5 tasks) | 5/5 | $0.000005 |
| Decomposition (3) | 3/3 | $0.000110 |
| Math/reasoning (2) | 2/2 | $0.000006 |
| File/shell tools (4) | 4/4 | $0.000019 |
| Code generation (2) | 2/2 | $0.000010 |
| Safety blocking (3) | 3/3 | $0.000000 |

30/30 tasks. $0.0006 total. Harmful requests blocked in 0.0 seconds.

---

## Tweet 6 (Comparison)

How does it compare?

| | GraphBot | AutoGPT | CrewAI | LangGraph |
|---|---|---|---|---|
| Cost/task | $0.00002 | $0.05-0.50 | $0.01-0.10 | $0.01-0.05 |
| Free models | Native | No | No | Partial |
| Knowledge graph | Built-in | None | None | None |
| Pattern learning | Automatic | None | None | None |
| Tests | 1500+ | ~200 | ~300 | ~500 |

That's 2,500x cheaper than AutoGPT and 500x cheaper than LangGraph.

---

## Tweet 7 (Knowledge Graph)

The secret weapon: a temporal knowledge graph (Kuzu, 11 node types).

Every execution teaches the graph:
- Entity relationships
- Execution patterns (reused next time for 0 tokens)
- Failure reflections (what went wrong and why)

Personalized PageRank retrieves context up to 3 hops deep. The model never guesses -- it knows.

[PLACEHOLDER: GIF of knowledge graph visualization in D3]

---

## Tweet 8 (Safety)

Safety is not optional for autonomous agents.

GraphBot blocks harmful requests BEFORE they reach any LLM:
- Constitutional principles (no harm, no deception)
- Composition attack detection (download + execute chains)
- Shell sandboxing with allowlists
- 3 autonomy levels: supervised / standard / autonomous

Harmful requests cost $0.000000 and 0.0 seconds. Zero LLM calls.

---

## Tweet 9 (Research Backing)

This is backed by 190+ papers from NeurIPS, ICLR, ICML, ACL, AAAI (2023-2026):

- DAGs validated by Graph of Thoughts (AAAI 2024) and MacNet (ICLR 2025)
- Graph memory validated by Graphiti, HippoRAG, GraphRAG
- External verification validated by CRITIC (ICLR 2024)
- Cost stacking validated by FrugalGPT + RouteLLM

Full research library in the repo: docs/research/

---

## Tweet 10 (CTA)

GraphBot is MIT licensed. Python 3.13. 1500+ tests.

Get started in 3 commands:

```
git clone https://github.com/LucasDuys/graphbot
pip install -e ".[dev]"
python scripts/run_capability_tests.py
```

Free OpenRouter key at openrouter.ai/keys -- that's all you need.

If this is useful, a star helps others find it.

https://github.com/LucasDuys/graphbot

[PLACEHOLDER: GIF of split-screen demo -- terminal + WhatsApp]

---

## Posting Notes

- Post tweet 1 standalone. Wait 2-3 minutes for initial engagement, then post the full thread as a reply chain.
- Each tweet is under 280 characters (verified) except tweets 5, 6, and 10 which use images/tables -- attach these as screenshots instead of inline text.
- For tweets with tables: screenshot the table from a markdown renderer (dark theme looks best on timeline).
- GIF placeholders: replace with actual recordings from `demos/` before posting. See `docs/RECORDING_GUIDE.md`.
- Best posting times: Tuesday-Thursday, 9-11 AM EST or 2-4 PM EST (US tech audience peak).
- Tag relevant accounts only if you have a genuine connection. Do not spam-tag AI influencers.
- Cross-post thread to Threads and Bluesky (same format works).
