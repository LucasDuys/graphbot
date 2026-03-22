# GraphBot Progress

## Current Phase
**Phase 15: Foundation for Autonomy -- COMPLETE**
All phases through 15 complete. System now learns from failures, has safety guards, model cascade, and semantic caching.

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

## Phase 2 Completed (2026-03-21)
- [x] T012: IntakeParser -- rule-based zero-cost intent classification, complexity scoring
- [x] T013: Decomposition JSON schema -- JSON schema + validator for task trees
- [x] T014: DecompositionPrompt -- prompt template for LLM-based decomposition
- [x] T015: Tree validation -- structural validation of decomposition output
- [x] T016: Decomposer -- LLM-based task decomposition with retry/fallback
- [x] T017: Orchestrator -- intake -> decompose -> execute pipeline, sequential DAG execution
- **Total: 213+ tests, all passing**

## Phase 3 Completed (2026-03-21)
- [x] T020: Rate limiting -- per-provider rate limiting with aiolimiter
- [x] T021: Circuit breaking -- per-provider circuit breaking with aiobreaker
- [x] T022: Tracing -- @traced decorator for pipeline stage timing
- [x] T023: DAGExecutor -- parallel streaming topological dispatch with semaphore-bounded concurrency
- [x] T024: ModelRouter wiring -- rate_limiter and circuit_breaker params in ModelRouter
- [x] T025/T026: Orchestrator integration -- replaced sequential _execute_dag with parallel DAGExecutor
- **Total: 757 tests, all passing**

## Phase 4 Completed (2026-03-21)
- [x] T027: PatternExtractor -- extracts reusable templates from completed trees
- [x] T028: PatternMatcher -- regex + Levenshtein matching against cached patterns
- [x] T029: PatternStore -- pattern persistence in Kuzu graph (with tree_template)
- [x] T030: GraphUpdater -- records Task/ExecutionTree nodes after execution
- [x] T031: Orchestrator wiring -- pattern cache check before decomposition, graph update after
- **Total: 282 tests, all passing**

## Phase 5 Completed (2026-03-21)
- [x] T033: Graph seed script (seed_graph.py) -- user, projects, services, memories
- [x] T033: A/B comparison script (compare.py) -- GraphBot vs baseline single LLM
- [x] T033: Canonical integration tests -- 5 test tasks with mocked provider
- [x] T033: QUICKSTART.md documentation
- [x] LangSmith EU endpoint configured, claude-agent-sdk installed
- **Total: 287 tests, all passing**

## Phase 7 Completed (2026-03-21)
- [x] T046/T047: End-to-end pipeline pattern verification
  - test_simple_task_1_call: simple task = exactly 1 LLM call, no decomposition
  - test_complex_task_1_plus_n_calls: complex task = 1 decompose + 3 leaves = 4 LLM calls, 0 aggregation
  - test_aggregated_output_has_template: template_fill produces correct slot-filled output
  - test_pattern_cache_skips_decomposition: pre-seeded PatternNode = 0 decompose + 3 leaves = 3 LLM calls
- **Total: 868 tests (864 pre-existing + 4 new), all passing**

## Phase 11 Completed (2026-03-22)
- [x] T088/T089: Structured tool params + smart decomposition
- [x] T090/T091: Tool failure retry + CodeEditAgent
- [x] T097: Shell command extraction fix
- [x] T098: Track tool vs LLM usage in ExecutionResult
- [x] Benchmark: 8/10 real tasks, $0.0005 cost, file 5/5, web 3/3
- **Total: 978 tests, all passing**

## Phase 12 Completed (2026-03-22)
- [x] T099: SSE tool.invoke + tool.result observability events in _process_task
- [x] T100: Updated PROGRESS.md + README.md with latest benchmarks
- [x] T101: Test coverage for tool observability events
- **Total: 979 tests, all passing**

## Phase 13 Completed (2026-03-22)
- [x] T102: Fix shell output interpretation + observability test failure (8 new tests, committed b0e053a)
- [x] T103: Git history squash -- 1385 Nanobot commits -> 1 "Initial fork" commit
- [x] T104: UI per-node live animations + dark mode toggle (committed 6339611)
- [x] T105: UI knowledge graph D3 force layout panel (committed 4236222)
- [x] T106: Multi-provider rotation -- OpenRouter + Google + Groq with fallback (14 tests, committed 13a498f)
- [x] T107: Telegram channel integration with Orchestrator bridge (committed dce0055)
- [x] T108: GAIA Level 1 benchmark runner -- 25 tasks (23 tests, committed 177d4f6)
- [x] T109: Pattern cache warming script -- 36 tasks, 6 categories (14 tests, committed 32dd225)
- [x] T110: Updated README.md + PROGRESS.md with Phase 13 results
- **Total: 1000+ tests, all passing**

## Phase 14 Completed (2026-03-22)
- [x] T112: WhatsApp channel bridge with Orchestrator integration (32 tests, committed 955a341)
- [x] T113: Model tier comparison benchmark -- free/mid/frontier with direct vs pipeline (47 tests, committed f594984)
- [x] T114: Research: planning and decomposition -- 24 papers
- [x] T115: Research: tool use and function calling -- 24 papers
- [x] T116: Research: memory and knowledge -- 22 papers
- [x] T117: Research: self-correction and verification -- 23 papers
- [x] T118: Research: multi-agent systems -- 23 papers
- [x] T119: Research: browser and computer use -- 25 papers
- [x] T120: Research: long-horizon execution -- 17 papers
- [x] T121: Research: real-world agent frameworks -- 16 papers
- [x] T122: Research: cost optimization and model routing -- 21 papers
- [x] T123: Research: safety and alignment -- 22 papers
- [x] T124: Research README index -- 190+ unique papers indexed (committed d4e7749)
- [x] T125: Daily use readiness -- healthcheck, .env.local.example (43 tests, committed e62e6cc)
- [x] T126: Architecture gap analysis -- 10 gaps, Phase 15+ roadmap (committed 1502d0d)
- **Key finding: Current DAG + graph architecture CAN support full autonomy (4 arch changes + 6 new features needed)**
- **Total: 630+ GraphBot tests, all passing**

## Phase 15 Completed (2026-03-22)
- [x] T127: Failure reflection engine -- post-execution LLM reflection stored in graph (committed 95db339)
- [x] T128: Pattern success/failure counters + weighted matching (11 tests, committed 6830ef3)
- [x] T129: DAG intent classifier + dangerous command blocking (committed 5b525e3)
- [x] T130: Model cascade mode -- try cheap first, escalate on low confidence (18 tests, committed 03464ad)
- [x] T131: Reflection retrieval in context assembly + decomposer injection (16 tests, committed 2f1145a)
- [x] T132: Failure deprioritization -- skip <20% success rate patterns (committed aece5ac)
- [x] T133: Composition attack detection + shell sandbox allowlist/blocklist (25 tests, committed 9bf14ed)
- [x] T134: Cascade confidence estimation + token budget directives (30 tests, committed 7863ad9)
- [x] T135: Semantic embedding layer for pattern matching -- sentence-transformers (15 tests, committed 92c8cd9)
- [x] T136: Output sanitization + recursion limits + safety integration tests (42 tests, committed 3e26d87)
- **Key capabilities: learns from failures, safety guards, model cascade, semantic caching**
- **Total: 792 tests, all passing**

## In Progress
_Nothing._

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
| Lines of code (new) | <8,000 | ~3,500 | core_gb + graph + models + scripts + tests |
| Test count | -- | 1000+ | All passing (Phase 13 complete) |
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

### 2026-03-21 -- Session 2: Phase 2 + Phase 3 (complete)
- Phase 2: Recursive decomposition pipeline (T012-T017)
  - IntakeParser: rule-based complexity scoring, zero LLM calls
  - Decomposer: LLM-based task tree generation with JSON schema validation
  - Orchestrator: full intake -> decompose -> execute pipeline
- Phase 3: Parallel DAG execution (T020-T026)
  - Rate limiting (aiolimiter) and circuit breaking (aiobreaker) per provider
  - @traced decorator for pipeline stage observability
  - DAGExecutor: streaming topological dispatch with asyncio.wait + semaphore
  - Wired DAGExecutor into Orchestrator, replacing sequential _execute_dag
  - Removed ~160 lines of sequential execution code from Orchestrator
- 757 tests passing, 0 failures
- Next session: Phase 4 -- pattern learning and caching

### 2026-03-22 -- Session 5: Phase 11 + Phase 12 (complete)
- Phase 11: Tool system refinements (T088-T098)
  - Structured tool params in TaskNode for domain-specific routing
  - Smart decomposition with INTEGRATED classification
  - Tool failure retry with LLM fallback
  - Shell command extraction fix
  - Tool vs LLM usage tracking in ExecutionResult
  - Benchmark: 8/10 real tasks, $0.0005 cost, tokens halved vs Phase 9
- Phase 12: SSE observability + documentation (T099-T101)
  - tool.invoke and tool.result SSE events for per-node visibility
  - Updated PROGRESS.md and README.md with latest benchmarks
  - 979 tests passing, 0 failures
