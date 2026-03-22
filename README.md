# GraphBot

A recursive DAG execution engine powered by a temporal knowledge graph that enables free/cheap LLMs to match the capability of expensive frontier models.

## Core Thesis

**A small model with perfect context beats a large model with no context.**

The knowledge graph eliminates the need for expensive inference by pre-computing what the model needs to know. The recursive DAG eliminates the need for complex reasoning by decomposing until each leaf is trivially simple. Parallel execution eliminates latency.

## How It Works

```
User Message -> Intake Parser (rule-based, zero-cost)
  -> Knowledge Graph Query (Kuzu)
  -> Pattern Match? YES -> Instantiate Template (0 tokens)
                   NO  -> Recursive Decomposer (small model + graph context)
  -> Parallel DAG Executor (streaming topological dispatch)
  -> Graph Update Loop (record task, update entities, extract patterns)
  -> Response
```

1. **Intake**: Rule-based intent classification. Zero LLM calls.
2. **Pattern Cache**: Check if a matching execution template exists. If so, skip decomposition entirely.
3. **Decomposition**: Break complex tasks into trivially simple subtasks using constrained JSON output.
4. **Parallel Execution**: Execute independent subtasks concurrently on free models via OpenRouter.
5. **Learning**: Extract reusable patterns from completed tasks. Next time, skip the LLM decomposition step.

## Architecture

| Component | Description |
|-----------|-------------|
| `core_gb/intake.py` | Rule-based intent classification (zero tokens) |
| `core_gb/decomposer.py` | LLM-based task tree generation with constrained JSON |
| `core_gb/dag_executor.py` | Parallel streaming DAG execution via `graphlib.TopologicalSorter` |
| `core_gb/orchestrator.py` | Main pipeline: intake -> pattern check -> decompose -> execute -> learn |
| `core_gb/patterns.py` | Pattern extraction, matching, and graph storage |
| `graph/store.py` | Kuzu graph database: schema, CRUD, context assembly |
| `graph/resolver.py` | 3-layer entity resolution (exact, Levenshtein, BM25) |
| `graph/updater.py` | Records execution outcomes in the knowledge graph |
| `models/router.py` | Multi-provider rotation with rate limiting + circuit breaking |
| `models/openrouter.py` | OpenRouter provider via LiteLLM |
| `models/google.py` | Google AI Studio provider (Gemini 2.5 Pro free tier) |
| `models/groq.py` | Groq direct provider (30 RPM free tier) |
| `tools_gb/` | File, web, shell, and code edit tools |
| `nanobot/channels/graphbot_telegram.py` | Telegram bot bridge to Orchestrator |
| `ui/frontend/` | Next.js dashboard with live DAG visualization |

## Quick Start

```bash
git clone https://github.com/LucasDuys/graphbot
cd graphbot
pip install -e ".[dev,langsmith]"
```

Create `.env.local`:
```
OPENROUTER_API_KEY=sk-or-v1-...
GOOGLE_API_KEY=...              # Optional: Gemini 2.5 Pro free tier
GROQ_API_KEY=...                # Optional: Groq 30 RPM free tier
TELEGRAM_BOT_TOKEN=...          # Optional: Telegram bot integration
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_PROJECT=graphbot
LANGSMITH_TRACING=true
```

Run:
```bash
python scripts/seed_graph.py          # Seed knowledge graph
python scripts/test_live.py           # Live integration test
python scripts/compare.py "your task" # A/B comparison
python scripts/run_gaia.py            # GAIA Level 1 benchmark
python scripts/warm_cache.py          # Pattern cache warming
python -m pytest tests/ -v            # Run test suite (1000+ tests)
```

## Benchmark Results

### Knowledge Tasks (15 tasks, 2026-03-21)

| Category | Tasks | Success | Avg Tokens | Avg Cost | Avg Latency |
|----------|-------|---------|-----------|----------|-------------|
| Simple (no decomposition) | 5 | 5/5 | 383 | $0.000015 | 11.5s |
| Parallel (3+ concurrent leaves) | 5 | 5/5 | 3,545 | $0.000348 | 32.4s |
| Sequential (ordered chain) | 5 | 5/5 | 3,044 | $0.000227 | 33.1s |
| **Total** | **15** | **15/15** | **2,324** | **$0.000197** | **25.6s** |

### Real-World Tool Tasks (10 tasks, 2026-03-22)

| Category | Tasks | Success | Notes |
|----------|-------|---------|-------|
| File (list, read, search) | 5 | 5/5 | All tool-routed |
| Web (search, summarize) | 3 | 3/3 | All tool-routed |
| Shell (command execution) | 2 | 2/2 | Fixed: stdout interpretation |
| **Total** | **10** | **10/10** | **$0.0005 total cost** |

**Tokens halved vs Phase 9** (15,935 down from 29,997). Total cost for all 25 tasks: $0.0035.

See [benchmarks/RESULTS.md](benchmarks/RESULTS.md) and [benchmarks/REAL_TASKS_RESULTS.md](benchmarks/REAL_TASKS_RESULTS.md) for full breakdown.

## Performance

- **1000+ tests**, all passing
- **Entity resolution**: ~3ms (meets <10ms target)
- **Pattern matching**: <5ms
- **Decomposition success**: 100% on 70B models with JSON mode
- **Real-world tasks**: 10/10 success (file 5/5, web 3/3, shell 2/2)
- **Cost**: $0.0002 average per task on free models
- **Multi-provider rotation**: OpenRouter -> Google -> Groq fallback chain
- **Pattern cache**: 30%+ token reduction after warming
- **SSE observability**: tool.invoke / tool.result events per leaf node

## Features

- **Live DAG Visualization**: Real-time per-node animations (pending/running/completed) via SSE
- **Knowledge Graph Panel**: D3 force layout showing entities and relationships
- **Dark Mode**: Toggleable, persisted to localStorage
- **Multi-Provider Rotation**: Automatic fallback across OpenRouter, Google AI Studio, Groq
- **Telegram Bot**: Mobile access via Telegram channel integration
- **GAIA Benchmark**: Level 1 benchmark runner for standardized evaluation
- **Pattern Cache Warming**: Pre-populate templates for 30%+ token savings

## Tech Stack

- **Graph DB**: Kuzu v0.11.3 (embedded, no server)
- **LLM Gateway**: OpenRouter + Google AI Studio + Groq via LiteLLM
- **Frontend**: Next.js + React Flow + D3.js + Jotai
- **Observability**: LangSmith (EU endpoint) + SSE event streaming
- **Rate Limiting**: aiolimiter (per-provider)
- **Circuit Breaking**: aiobreaker (per-provider)
- **Tracing**: Custom `@traced` decorator + LangSmith callbacks
- **Channels**: Telegram (python-telegram-bot)

## Origin

GraphBot started as a fork of [Nanobot](https://github.com/HKUDS/nanobot) (v0.1.4.post5) -- a lightweight agent framework. The Nanobot codebase provides the agent loop, tool system, MCP integration, and channel support. GraphBot replaces the memory system with a temporal knowledge graph, adds recursive DAG decomposition, and implements parallel execution.

## License

MIT
