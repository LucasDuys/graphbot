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
| `models/router.py` | Complexity-based model selection with rate limiting + circuit breaking |
| `models/openrouter.py` | OpenRouter provider via LiteLLM |

## Quick Start

```bash
git clone https://github.com/LucasDuys/graphbot
cd graphbot
pip install -e ".[dev,langsmith]"
```

Create `.env.local`:
```
OPENROUTER_API_KEY=sk-or-v1-...
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
python -m pytest tests/ -v            # Run test suite (287 tests)
```

## Test Results

| Test | Description | Nodes | Tokens | Cost |
|------|-------------|-------|--------|------|
| Simple math | "What is 247 * 38?" | 1 | 39 | $0.000001 |
| Context query | "What do you know about Lucas?" | 1 | 151 | $0.000005 |
| Parallel decomposition | "Compare weather in 3 cities" | 4 | 2,731 | $0.000318 |

## Performance

- **287 tests**, all passing
- **Entity resolution**: ~3ms (meets <10ms target)
- **Pattern matching**: <5ms
- **Parallel speedup**: independent leaves execute concurrently
- **Cost**: $0.0003 for a 4-node parallel task on free models

## Tech Stack

- **Graph DB**: Kuzu v0.11.3 (embedded, no server)
- **LLM Gateway**: OpenRouter via LiteLLM
- **Observability**: LangSmith (EU endpoint)
- **Rate Limiting**: aiolimiter (per-provider)
- **Circuit Breaking**: aiobreaker (per-provider)
- **Tracing**: Custom `@traced` decorator + LangSmith callbacks

## Origin

GraphBot started as a fork of [Nanobot](https://github.com/HKUDS/nanobot) (v0.1.4.post5) -- a lightweight agent framework. The Nanobot codebase provides the agent loop, tool system, MCP integration, and channel support. GraphBot replaces the memory system with a temporal knowledge graph, adds recursive DAG decomposition, and implements parallel execution.

## License

MIT
