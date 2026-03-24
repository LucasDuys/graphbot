# Show HN: I built an autonomous agent that runs on free LLMs for $0.0006/30 tasks

_(Draft -- do not post without explicit authorization)_

---

I have been building an open-source agent framework called **GraphBot** that makes free/cheap LLMs dramatically more capable by giving them structured context from a knowledge graph. It runs 30 diverse tasks on Llama 8B for a total cost of $0.0006.

**The core idea:** A small model with perfect context beats a large model with no context.

Instead of throwing harder tasks at bigger models, GraphBot decomposes any task into a recursive DAG of trivially simple subtasks, gives each one hyper-specific context from a temporal knowledge graph (Kuzu), and executes them in parallel on free models via OpenRouter. The knowledge graph eliminates guessing. The DAG eliminates complex reasoning. Pattern caching eliminates repeated work.

## What it actually does

- Takes a task like "Research the top 5 AI agent frameworks and write a comparison report"
- Decomposes it into 5 parallel web searches + 1 synthesis step (constrained JSON output, no prompt hacking)
- Each search node gets graph context about what the model already knows about that framework
- Executes all 5 searches simultaneously on Llama 8B (free tier)
- Synthesizes results into a coherent report
- Learns the pattern so next time a similar task costs 0 tokens

## The numbers

| What We Tested | Tasks | Pass Rate | Avg Latency | Total Cost |
|----------------|-------|-----------|-------------|------------|
| Simple Q&A | 5 | **5/5** | 4.3s | $0.000005 |
| Task decomposition (parallel) | 3 | **3/3** | 27.7s | $0.000110 |
| Math & reasoning | 2 | **2/2** | 2.0s | $0.000006 |
| File & shell tools | 4 | **4/4** | 10.2s | $0.000019 |
| Web search | 1 | **1/1** | 11.0s | $0.000014 |
| Code generation | 2 | **2/2** | 3.9s | $0.000010 |
| Translation (3 languages) | 1 | **1/1** | 14.4s | $0.000023 |
| Creative writing | 2 | **2/2** | 2.3s | $0.000008 |
| Safety (harmful requests) | 3 | **3/3 blocked** | 0.0s | $0.000000 |
| Analysis & classification | 3 | **3/3** | 4.1s | $0.000013 |
| Pattern cache hit | 1 | **1/1** | 3.2s | $0.000001 |
| **Total** | **30** | **30/30** | **8.0s avg** | **$0.0006** |

All traces verified on LangSmith.

## How it compares

| | GraphBot | AutoGPT | CrewAI | LangGraph |
|---|---|---|---|---|
| **Cost per task** | $0.00002 | $0.05-0.50 | $0.01-0.10 | $0.01-0.05 |
| **Free model support** | Native (Llama 8B) | GPT-4 required | GPT-4 recommended | Model agnostic |
| **Task decomposition** | Recursive DAG | Serial chain | Role-based | Graph states |
| **Knowledge graph** | Built-in (Kuzu) | None | None | None |
| **Pattern learning** | Automatic | None | None | None |
| **Self-correction** | 3-layer verification | Retry loop | None | Custom |
| **Safety** | Constitutional + autonomy levels | None | None | None |
| **Tests** | 1500+ | ~200 | ~300 | ~500 |

## Architecture in 30 seconds

```
User Message -> Intake Parser (rule-based, zero-cost)
  -> Knowledge Graph Query (Kuzu)
  -> Pattern Match? YES -> Instantiate Template (0 tokens)
                   NO  -> Recursive Decomposer (small model + graph context)
  -> Pipelined DAG Executor (7 concurrent stages)
  -> Graph Update Loop (learn pattern for next time)
  -> Response
```

Key components: recursive DAG decomposition, temporal knowledge graph (Kuzu, 11 node types), 3-layer verification (format + self-consistency + KG fact-check), semantic pattern caching (sentence-transformers), multi-provider failover, constitutional safety with 3 autonomy levels.

## What I learned building it

This started as a research project to test whether structured context injection could close the gap between small and large models. I read 190+ papers across planning, tool use, memory, verification, safety, and cost optimization. The research is all in the repo under `docs/research/`.

Key findings that shaped the architecture:
- DAGs are the right abstraction for task decomposition -- independently validated by Graph of Thoughts (AAAI 2024), MacNet (ICLR 2025), and "Beyond Entangled Planning" (2026)
- Graph memory beats vector stores for agent context -- confirmed by Graphiti, HippoRAG, GraphRAG
- External verification works, self-correction does not -- CRITIC (ICLR 2024), "Cannot Self-Correct" (ICLR 2024)
- Cost stacking compounds -- FrugalGPT + RouteLLM + caching gets you to 2-8% of naive cost

## Honest limitations

- Llama 8B still struggles with genuinely novel multi-hop reasoning that requires knowledge not in the graph
- Latency is higher than a single GPT-4o call for simple tasks (the decomposition overhead is not worth it for "What is the capital of France?")
- The knowledge graph needs time to become useful -- first few runs are slower, later runs benefit from cached patterns
- Browser automation (Playwright) is fragile for complex web interactions
- This is a solo research project, not a production-hardened framework

## Try it

```bash
git clone https://github.com/LucasDuys/graphbot
cd graphbot
pip install -e ".[dev]"
echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env.local
python scripts/healthcheck.py
python scripts/run_capability_tests.py
```

Free OpenRouter key: https://openrouter.ai/keys

## Tech stack

Python 3.13, Kuzu (embedded graph DB), LiteLLM, Playwright, sentence-transformers, Next.js dashboard, LangSmith tracing.

1500+ tests. MIT license.

GitHub: https://github.com/LucasDuys/graphbot

---

_I'm an independent researcher (2nd year CS student at TU/e). Happy to answer questions about the architecture, research, or specific implementation decisions._
