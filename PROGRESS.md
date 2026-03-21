# GraphBot Progress

## Current Phase
**Phase 0: Pre-Implementation -- COMPLETE**
Next: Begin Phase 1 (Foundation -- Graph + Types + Single Model)

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

## In Progress
_Nothing -- ready for Phase 1._

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
| Lines of code (new) | <8,000 | ~300 | types.py + schema.py + tests |
| Test coverage | >80% | -- | Not yet running |
| Simple task latency | <1s | -- | |
| Complex task latency | <8s | -- | |
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
