# GraphBot Progress

## Current Phase
**Phase 1: Foundation -- COMPLETE**
Next: Begin Phase 2 (Recursive Decomposition + Constrained JSON)

## Completed
- [x] Full architecture design (2026-03-20)
- [x] Created design docs: CLAUDE.md, AGENTS.md, PROGRESS.md, PLANNING.md, RESEARCH.md, TESTS.md
- [x] Cloned Nanobot (HKUDS/nanobot v0.1.4.post5) as project base
- [x] Set up GraphBot overlay: core_gb/, graph/, agents_gb/, models/, tools_gb/
- [x] Wrote core types: TaskNode, ExecutionResult, Pattern, GraphContext (core_gb/types.py)
- [x] Wrote graph schema: 10 node types, 12 edge types (graph/schema.py)
- [x] Wrote test fixtures and type tests (tests/conftest.py, tests/test_core/test_types.py)
- [x] Research: Kuzu API -- ARCHIVED but functional, sub-5ms 2-hop, LadybugDB fork exists
- [x] Research: Free LLM tiers -- Groq (Llama 3.3 70B, Qwen3 32B), Cerebras (Qwen3 235B), Google (Gemini 2.5 Pro)
- [x] Research: Nanobot audit -- keep agent loop/tools/MCP/providers/channels, replace memory/skills
- [x] Research: Recursive decomposition -- ROMA (open source), ReAcTree (2.4x improvement), NoThinking (9x latency)
- [x] Research: Temporal KG -- Graphiti supports Kuzu, context sweet spot 1.5-3K tokens for 8B models
- [x] Research: OpenClaw -- 250K stars, serial execution, $300-750/mo, no native KG
- [x] Research: Async DAG -- graphlib.TopologicalSorter + streaming dispatch + aiolimiter + aiobreaker
- [x] Updated all ADRs with research findings (PLANNING.md)
- [x] Corrected model references (Llama 3.1 70B -> 3.3 70B, Qwen 2.5 Coder -> Qwen3 32B)
- [x] Corrected Medprompt claim (matches specialists, not reasoning models)
- [x] Cross-referenced Pitchr/Sidekick research (29 docs) -- extracted 6 new findings
- [x] Added ADRs 011-016: observability, hybrid verification, LLM-as-judge, A/B comparison, longitudinal metrics, prompt engineering patterns
- [x] Integrated 4 missing testing areas into TESTS.md: quality scoring, A/B comparison, pattern cache tracking, graph growth metrics

## Phase 1 Completed (2026-03-21)
- [x] T001: Environment setup -- pyproject.toml updated, kuzu/litellm/langsmith installed, Python 3.13 verified
- [x] T002: Core types -- ExecutionResult/Pattern/GraphContext frozen, CompletionResult added, 20 tests
- [x] T003: GraphStore schema creation -- 10 node + 12 edge tables, idempotent init, 7 tests
- [x] T004: OpenRouter provider -- ModelProvider ABC, CompletionResult, error hierarchy, 10 tests
- [x] T005: Graph CRUD -- create/get/update/delete nodes + edges, parameterized Cypher, 22 tests
- [x] T006: Model router -- complexity-based selection, DEFAULT_MODEL_MAP, scale-ready, 9 tests
- [x] T007: LangSmith observability -- litellm callback, silent degradation, 8 tests
- [x] T008: Context assembly -- 2-hop traversal, token budget, active memory filtering, 13 tests
- [x] T009: Entity resolution -- 3-layer (exact/Levenshtein/BM25), >90% accuracy on 50 pairs, 13 tests
- [x] T010: Graph benchmarks -- bench_graph.py at 100-10K nodes, JSONL tracking, 9 tests
- [x] T011: SimpleExecutor -- E2E path (graph context -> LLM call -> result), 5 tests
- **Total: 111 new tests, all passing**

## In Progress
_Nothing -- ready for Phase 2._

## Blocked
_Nothing blocked._

## Key Decisions Made
| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-03-20 | Recursive DAG executor with knowledge graph | Combines ROMA + MiroFish + NoThinking | Flat routing, single-agent RAG |
| 2026-03-20 | Fork Nanobot | 5.3K core lines, model-agnostic, MCP, 8+ channels | OpenClaw (300-500K lines), from scratch |
| 2026-03-20 | Kuzu v0.11.3 (with LadybugDB migration plan) | Embedded, pip install, sub-5ms perf, archived but functional | DuckDB+DuckPGQ, SQLite+custom, NetworkX |
| 2026-03-20 | Multi-provider rotation | No single free tier has enough capacity | Single provider |
| 2026-03-20 | Streaming DAG dispatch | graphlib.TopologicalSorter, immediately unblock dependents | Batch-per-level |
| 2026-03-20 | Graphiti for temporal KG (prototype) | Supports Kuzu driver, temporal model for free | Custom from scratch |
| 2026-03-20 | Constrained JSON for decomposition | 8B models need structured output enforcement | Free-form generation |

## Open Questions
- [ ] Graphiti LLM overhead with local 7-8B models -- acceptable or need custom solution?
- [ ] Constrained JSON decoding: use model's native JSON mode or external library (outlines, guidance)?
- [ ] Pattern matching: Graphiti's entity resolution sufficient or need custom for task patterns?
- [ ] LangSmith free tier (5K traces/month) sufficient for sustained development?

## Metrics
| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Lines of code (new) | <8,000 | ~1,200 | types + store + CRUD + context + resolver + provider + router + executor |
| Test count | -- | 111 | All passing |
| 2-hop query (100 nodes) | <1ms | ~13ms | Needs optimization (full table scan approach) |
| Context assembly (100 nodes) | <10ms | ~23ms | Needs indexed queries |
| Entity resolution (100 nodes) | <10ms | ~3ms | Meets target |
| Simple task latency | <1s | -- | Needs integration test with real API |
| API cost (simple) | $0.00 | -- | |

## Session Log
### 2026-03-20 -- Session 0: Design + Research + Setup
- Designed full architecture across brainstorming sessions
- Cloned HKUDS/nanobot as project base (5.3K core, 28K total)
- Created overlay: core_gb/types.py, graph/schema.py, test fixtures
- Dispatched 7 parallel research agents, all completed
- **Key research findings that changed the design:**
  - Kuzu archived -> plan for LadybugDB migration
  - Llama 3.1 70B deprecated -> use 3.3 70B
  - Qwen 2.5 Coder unavailable -> use Qwen3 32B
  - Graphiti supports Kuzu driver -> can prototype faster
  - Context sweet spot: 1.5-3K tokens for 8B models
  - Decomposition needs constrained JSON output
- Updated all ADRs and model references
- Next session: Begin Phase 1 -- install Kuzu, implement graph store, wire up single model execution

### 2026-03-21 -- Session 1: Phase 1 Foundation (complete)
- Used Forge framework for spec/plan/execute pipeline
- 11 tasks across 5 tiers, executed with full autonomy via parallel subagents
- Built bottom-up: env -> types -> graph store -> provider -> CRUD -> router/langsmith -> context/resolver -> benchmarks/executor
- Key implementation decisions:
  - OpenRouter as single provider gateway (user provides API key)
  - Raw Kuzu (no Graphiti) for maximum performance at scale
  - TaskNode stays mutable (execution state), output types frozen
  - 3-layer entity resolution: exact/Levenshtein/BM25 (no LLM calls)
  - Context assembly via explicit per-edge-type queries (Kuzu v0.11.3 variable-length path limitations)
- Benchmark results: entity resolution fast (3ms), 2-hop and context assembly need optimization (13-23ms at 100 nodes)
- Optimization path: indexed lookups, cached entity lists, batched queries
- Next session: Phase 2 -- recursive decomposition with constrained JSON output
