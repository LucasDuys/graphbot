# GraphBot

An open-source autonomous agent framework powered by recursive DAG decomposition and a temporal knowledge graph. Any model -- free or frontier -- gets better and cheaper through task decomposition, parallel execution, and learned patterns.

## Core Thesis

**A small model with perfect context beats a large model with no context.**

The knowledge graph pre-computes what the model needs to know. The recursive DAG decomposes complex tasks until each leaf is trivially simple. Parallel execution across free LLMs eliminates latency. The result: Llama 8B with GraphBot context approaches the quality of much larger models at near-zero cost.

## Benchmark Results (30-task capability suite)

| Category | Tasks | Pass Rate | Avg Latency | Avg Cost | Notes |
|----------|-------|-----------|-------------|----------|-------|
| Simple Q&A | 5 | 5/5 | 4.3s | $0.000005 | Single LLM call, no decomposition |
| Decomposition | 3 | 3/3 | 27.7s | $0.000110 | Parallel subtasks + LLM synthesis |
| Reasoning | 2 | 2/2 | 2.0s | $0.000006 | Math, logic |
| Tool: File | 2 | 2/2 | 12.7s | $0.000034 | List dirs, read files |
| Tool: Shell | 2 | 2/2 | 9.6s | $0.000004 | Python version, git log |
| Tool: Web | 1 | 1/1 | 11.0s | $0.000014 | Web search + summarize |
| Knowledge | 3 | 3/3 | 10.3s | $0.000021 | Thermodynamics, HTTP/S, Turing |
| Creative | 2 | 2/2 | 2.3s | $0.000008 | Haiku, name generation |
| Safety (blocked) | 3 | 3/3 | 0.0s | $0.000000 | rm -rf, spam, malware -- all blocked pre-decomposition |
| Analysis | 2 | 2/2 | 1.1s | $0.000007 | Sentiment, paradigm classification |
| Translation | 1 | 1/1 | 14.4s | $0.000023 | 3-language parallel |
| Summarization | 1 | 1/1 | 1.7s | $0.000022 | Concise 2-sentence summary |
| Cache Hit | 1 | 1/1 | 3.2s | $0.000001 | Repeated task hits pattern cache |
| Code | 2 | 2/2 | 3.9s | $0.000010 | Prime checker, list comprehension |
| **Total** | **30** | **30/30** | **8.0s avg** | **$0.000624** | |

**Total cost for 30 tasks: $0.000624 (less than a tenth of a cent).**

All traces visible in [LangSmith](https://eu.api.smith.langchain.com) under the `graphbot` project.

## How It Works

```
User Message
  -> Pre-decomposition Safety Check (zero cost, blocks harmful requests in 0ms)
  -> Trivial Query Fast Path (greetings/acks return instantly)
  -> Intake Parser (rule-based intent + complexity classification)
  -> Pattern Cache (semantic embeddings, domain-scoped matching)
     HIT  -> Instantiate Template (0 tokens)
     MISS -> Recursive Decomposer (constrained JSON, graph context)
  -> Safety + Constitutional Check on decomposed plan
  -> Parallel DAG Executor (streaming topological dispatch)
     - Per-node verification (format, self-consistency, KG fact-check)
     - Per-node autonomy enforcement (risk scoring)
     - Tool execution with fallback (file, web, shell, browser, code)
     - Re-decomposition on failure (mutable DAG)
     - Output sanitization between nodes
  -> LLM Synthesis Aggregation (clean prose, no JSON artifacts)
  -> Graph Update (record task, extract patterns, store reflections)
  -> Conversation Memory (per-chat history for follow-ups)
  -> Response
```

## Quick Start

```bash
git clone https://github.com/LucasDuys/graphbot
cd graphbot
pip install -e ".[dev,langsmith]"
```

Create `.env.local`:
```bash
# Required (at least one provider)
OPENROUTER_API_KEY=sk-or-v1-...          # https://openrouter.ai/keys

# Optional providers (for multi-provider fallback)
GOOGLE_API_KEY=...                        # https://aistudio.google.com/apikey
GROQ_API_KEY=...                          # https://console.groq.com/keys

# Optional channels
TELEGRAM_BOT_TOKEN=...                    # @BotFather on Telegram
WHATSAPP_BRIDGE_URL=ws://localhost:3001   # Baileys bridge (see below)
WHATSAPP_BRIDGE_TOKEN=...                 # Optional auth token

# Optional observability
LANGSMITH_API_KEY=lsv2_pt_...             # https://smith.langchain.com/settings
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_PROJECT=graphbot
LANGSMITH_TRACING=true

# Optional safety
AUTONOMY_LEVEL=standard                   # supervised | standard | autonomous
SHELL_ALLOWLIST=echo,ls,python,git,cat    # Restrict shell commands
SHELL_BLOCKLIST=                           # Block specific commands
BROWSER_ALLOWLIST=                         # Restrict browser domains
BROWSER_BLOCKLIST=                         # Block browser domains
BROWSER_ALLOW_FORMS=false                  # Block form submissions by default
```

Run:
```bash
python scripts/seed_graph.py                # Seed knowledge graph
python scripts/healthcheck.py               # Verify all keys and services
python scripts/run_capability_tests.py      # Run 30-task benchmark suite
python scripts/compare_models.py            # Compare model tiers (free/mid/frontier)
python scripts/warm_cache.py                # Pre-populate pattern cache
python scripts/maintain_graph.py            # Consolidate + forget stale memories
python scripts/evaluate_goals.py            # Check progress on active goals
python -m pytest tests/ -v                  # Run test suite (1500+ tests)
```

## WhatsApp Setup

GraphBot can be used as a WhatsApp bot for daily mobile use.

```bash
# Terminal 1: Start the WhatsApp bridge
cd bridge && npm install && npm run build && npm start
# Scan the QR code with your phone (WhatsApp > Linked Devices)

# Terminal 2: Start the GraphBot listener
python -c "import asyncio; from nanobot.channels.graphbot_whatsapp import GraphBotWhatsAppChannel; bot = GraphBotWhatsAppChannel(); asyncio.run(bot.start())"
```

Then message the linked WhatsApp number. GraphBot processes your message through the full pipeline and replies with the answer + a stats footer (node count, cost).

Conversation memory is enabled -- follow-up questions use context from previous messages in the same chat.

## Architecture

| Component | Description |
|-----------|-------------|
| **Core Pipeline** | |
| `core_gb/orchestrator.py` | Main pipeline: safety -> intake -> pattern -> decompose -> execute -> learn |
| `core_gb/intake.py` | Rule-based intent classification + trivial query fast path |
| `core_gb/decomposer.py` | LLM-based task tree generation with constrained JSON |
| `core_gb/dag_executor.py` | Parallel streaming DAG with conditional/loop nodes, re-decomposition |
| `core_gb/patterns.py` | Domain-scoped pattern extraction, semantic matching, slot validation |
| `core_gb/aggregator.py` | LLM synthesis aggregation (clean prose output) |
| `core_gb/conversation.py` | Per-chat conversation memory with graph persistence |
| **Verification** | |
| `core_gb/verification.py` | 3-layer: format/type (L1), self-consistency (L2), KG fact-check (L3) |
| `core_gb/confidence.py` | Cascade confidence estimation with logprobs + heuristics |
| **Safety** | |
| `core_gb/safety.py` | Pre-decomposition blocking, composition attack detection |
| `core_gb/constitution.py` | 5 principles: no harm, no deception, no unauthorized access, privacy, minimal side effects |
| `core_gb/autonomy.py` | 3 autonomy levels with per-action risk scoring |
| `core_gb/sanitizer.py` | Output sanitization between DAG nodes (prompt injection prevention) |
| `core_gb/transaction.py` | Transactional execution with rollback for file/shell operations |
| **Knowledge Graph** | |
| `graph/store.py` | Kuzu embedded graph: schema, CRUD, activation-aware context assembly |
| `graph/resolver.py` | 3-layer entity resolution (exact, Levenshtein, BM25) |
| `graph/retrieval.py` | Personalized PageRank for multi-hop context (finds 3+ hop relationships) |
| `graph/activation.py` | ACT-R activation model (recency + frequency scoring) |
| `graph/consolidation.py` | Dedup entities, merge properties, generate summaries |
| `graph/forgetting.py` | Archive stale nodes to cold storage (protected: User, Project) |
| `graph/reflection.py` | Post-failure LLM reflection stored for future decomposition |
| **Models** | |
| `models/router.py` | Multi-provider rotation + model cascade (cheap first, escalate on low confidence) |
| `models/openrouter.py` | OpenRouter provider via LiteLLM |
| `models/google.py` | Google AI Studio (Gemini 2.5 Pro free tier) |
| `models/groq.py` | Groq direct (30 RPM free tier) |
| **Tools** | |
| `tools_gb/registry.py` | Tool registry with quality tracking (success/failure/latency stats) |
| `tools_gb/file.py` | File read, list, search |
| `tools_gb/web.py` | Web search (DuckDuckGo) + URL fetch |
| `tools_gb/shell.py` | Shell execution with allowlist/blocklist sandboxing |
| `tools_gb/browser.py` | Playwright browser automation (navigate, click, fill, extract, screenshot) |
| `tools_gb/browser_planner.py` | Planner-grounder pattern for multi-step browser workflows |
| `core_gb/tool_factory.py` | Dynamic tool creation via LLM (sandbox tested, persisted as Skill nodes) |
| **Goals** | |
| `core_gb/goals.py` | Persistent goals with decomposition, progress tracking, auto-completion |
| `scripts/evaluate_goals.py` | Cron-triggered goal evaluation with WhatsApp notifications |
| **Channels** | |
| `nanobot/channels/graphbot_whatsapp.py` | WhatsApp bot (Baileys bridge) |
| `nanobot/channels/graphbot_telegram.py` | Telegram bot |
| `ui/frontend/` | Next.js dashboard: live DAG visualization, D3 knowledge graph, dark mode |

## Features

**Execution Engine**
- Recursive DAG decomposition with parallel streaming execution
- Dynamic execution: conditional nodes, loop nodes, lazy expansion, re-decomposition on failure
- Intermediate result feedback with optional re-planning
- Model cascade: try cheapest model first, escalate on low confidence
- Multi-provider rotation: OpenRouter -> Google -> Groq automatic failover

**Intelligence**
- Temporal knowledge graph (Kuzu) with 11 node types, 14+ edge types
- ACT-R activation model for relevance-ranked context assembly
- Personalized PageRank retrieval (finds context 3+ hops deep)
- Semantic pattern matching via sentence-transformers embeddings
- Failure reflection: learns from mistakes, retrieves lessons for future tasks
- Memory consolidation + forgetting (graph stays clean as it grows)
- Conversation memory: multi-turn follow-ups per chat

**Safety**
- Pre-decomposition blocking (harmful requests caught in 0ms, zero LLM calls)
- Composition attack detection (download + execute chains)
- Constitutional principles (5 principles checked on every plan)
- Multi-model cross-validation for high-risk plans
- 3 autonomy levels: supervised, standard, autonomous
- Output sanitization between DAG nodes
- Transactional rollback for file/shell operations
- Shell + browser allowlist/blocklist sandboxing

**Tools**
- File, web, shell, browser (Playwright), code editing
- Dynamic tool creation: LLM generates, sandbox tests, registers, and persists new tools
- Tool quality tracking with automatic degradation warnings
- Browser policy guards (domain filtering, form control, audit logging)

**Verification**
- Layer 1: Format/type checking on every node (zero cost)
- Layer 2: 3-way self-consistency sampling for important nodes
- Layer 3: CRITIC-style fact-checking against knowledge graph

**Goals**
- Persistent Goal nodes spanning multiple sessions
- Automatic decomposition into session-sized sub-tasks
- Progress tracking with auto-completion
- Cron-triggered evaluation with WhatsApp notifications

## Performance

- **1500+ tests**, all passing
- **30/30 capability benchmark** (Q&A, decomposition, tools, safety, code, translation, etc.)
- **$0.000624 total cost** for 30 real tasks on free models
- **0.0s safety blocking** (harmful requests caught pre-decomposition)
- **1-5s simple Q&A** latency (down from 1-17s after optimization)
- **3-layer verification** with configurable per-node levels
- **Multi-provider failover** across 3 free LLM providers
- **Pattern cache** with 30%+ token reduction on repeated task types
- **190+ research papers** analyzed across 14 research documents

## Research

GraphBot's architecture is informed by 190+ papers from 2023-2026. The full research library is in `docs/research/`:

| Document | Papers | Topic |
|----------|--------|-------|
| [01](docs/research/01-planning-and-decomposition.md) | 24 | Task planning, hierarchical decomposition, DAG execution |
| [02](docs/research/02-tool-use-and-function-calling.md) | 24 | Tool-augmented LLMs, function calling, API integration |
| [03](docs/research/03-memory-and-knowledge.md) | 22 | Long-term memory, RAG, knowledge graphs for agents |
| [04](docs/research/04-self-correction-and-verification.md) | 23 | Self-debugging, CRITIC, LLM-as-judge, verification |
| [05](docs/research/05-multi-agent-systems.md) | 23 | AutoGen, CrewAI, MetaGPT, agent coordination |
| [06](docs/research/06-browser-and-computer-use.md) | 25 | WebArena, SeeAct, computer use, GUI automation |
| [07](docs/research/07-long-horizon-execution.md) | 17 | Voyager, DEPS, persistent agents, self-improvement |
| [08](docs/research/08-real-world-agent-frameworks.md) | 16 | OpenHands, Devin, SWE-Agent, LangGraph, DSPy |
| [09](docs/research/09-cost-optimization-and-model-routing.md) | 21 | FrugalGPT, RouteLLM, semantic caching, prompt compression |
| [10](docs/research/10-safety-and-alignment.md) | 22 | Agent safety, sandboxing, alignment, OWASP |
| [11](docs/research/11-architecture-gap-analysis.md) | -- | Gap analysis + Phase 15-20 roadmap |
| [12](docs/research/12-pattern-caching-best-practices.md) | 16 | Cache pollution prevention, domain scoping |
| [13](docs/research/13-agent-latency-optimization.md) | 20 | Pipeline optimization, fast paths, lazy evaluation |
| [14](docs/research/14-conversation-memory-for-agents.md) | 10+ | Multi-turn memory, context management |

## Tech Stack

- **Graph DB**: Kuzu v0.11.3 (embedded, no server)
- **LLM Gateway**: OpenRouter + Google AI Studio + Groq via LiteLLM
- **Frontend**: Next.js + React Flow + D3.js + Jotai
- **Browser**: Playwright (async, policy-guarded)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, local)
- **Observability**: LangSmith (EU endpoint) + SSE event streaming
- **Rate Limiting**: aiolimiter (per-provider)
- **Circuit Breaking**: aiobreaker (per-provider)
- **Channels**: WhatsApp (Baileys), Telegram (python-telegram-bot)

## Origin

GraphBot started as a fork of [Nanobot](https://github.com/HKUDS/nanobot) (v0.1.4.post5). The Nanobot codebase provides the agent loop, tool system, MCP integration, and channel support. GraphBot replaces the memory system with a temporal knowledge graph, adds recursive DAG decomposition, implements parallel execution, and layers on verification, safety, and autonomous agent capabilities.

## License

MIT
