---
spec: phase1-foundation
total_tasks: 11
estimated_tokens: 58000
depth: standard
---

# Phase 1 Foundation Frontier

## Tier 1 (parallel -- no dependencies)
- [T001] Environment setup and dependency installation | est: ~2k tokens | repo: graphbot | reqs: R001
  - Create/update pyproject.toml with all Phase 1 deps
  - Verify kuzu, litellm imports work
  - Confirm Python 3.11+, run existing tests
  - Deliverable: `pip install -e .` succeeds, `pytest tests/ -v` exits 0

- [T002] Core types validation and extension | est: ~3k tokens | repo: graphbot | reqs: R011
  - Read existing core_gb/types.py, verify all fields match spec
  - Extend TaskNode with complexity (1-5), provides, consumes, status
  - Extend ExecutionResult with context_tokens, cost
  - Ensure all dataclasses are frozen
  - Write/update tests in tests/test_core/test_types.py
  - Deliverable: all type tests pass, frozen immutability verified

## Tier 2 (depends on Tier 1)
- [T003] Kuzu graph store -- schema creation | est: ~6k tokens | repo: graphbot | depends: T001, T002 | reqs: R002
  - Implement GraphStore class in graph/store.py
  - __init__(db_path: str | None) -- None for in-memory
  - initialize() creates all 10 node + 12 edge tables via Cypher DDL
  - Idempotent schema creation (IF NOT EXISTS or catch-and-skip)
  - Temporal columns (valid_from, valid_until) on Memory, Task, etc.
  - close() for clean teardown
  - Test: create, initialize, verify tables via CALL show_tables()
  - Deliverable: GraphStore creates full schema, tests pass

- [T004] OpenRouter model provider | est: ~6k tokens | repo: graphbot | depends: T001, T002 | reqs: R007
  - Define ModelProvider protocol/ABC in models/base.py
  - Define CompletionResult dataclass (content, model, tokens_in/out, latency_ms, cost)
  - Implement OpenRouterProvider in models/openrouter.py using litellm.acompletion()
  - Model format: "openrouter/{model_name}"
  - API key from OPENROUTER_API_KEY env var
  - Typed errors: RateLimitError, AuthError, ProviderError
  - Test with mocked litellm: routing, error handling, result parsing
  - Deliverable: provider passes all unit tests, integration test marked (requires key)

## Tier 3 (depends on Tier 2)
- [T005] Kuzu graph store -- CRUD operations | est: ~6k tokens | repo: graphbot | depends: T003 | reqs: R003
  - Implement create_node, get_node, update_node, delete_node
  - Implement create_edge, query (raw Cypher)
  - ALL queries use parameterized Cypher (injection safe)
  - Test: full CRUD cycle on Entity, Memory, Task node types
  - Test: edge creation and traversal queries
  - Deliverable: all CRUD tests pass with parameterized queries

- [T006] Model router (scale-ready) | est: ~5k tokens | repo: graphbot | depends: T004 | reqs: R008
  - Implement ModelRouter in models/router.py
  - route(task: TaskNode) selects model by task.complexity
  - Configurable model mapping dict (not hardcoded)
  - Interface accepts single provider now, list[provider] later
  - Rate limiter / circuit breaker hooks (empty for Phase 1)
  - Test: route different complexities, verify model selection
  - Test: provider failure propagation
  - Deliverable: router correctly maps complexity to models, tests pass

- [T007] LangSmith observability setup | est: ~3k tokens | repo: graphbot | depends: T004 | reqs: R009
  - Enable litellm.success_callback = ["langsmith"] in provider init
  - Read LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2 from env
  - Silent degradation when env vars missing (no crash)
  - Test: verify callback registration with mock
  - Deliverable: tracing enabled when configured, silent when not

## Tier 4 (depends on Tier 3)
- [T008] Graph store -- context assembly | est: ~7k tokens | repo: graphbot | depends: T005 | reqs: R004
  - Implement get_context(entity_ids, max_tokens=2500) -> GraphContext
  - 2-hop traversal from given entities
  - Token budget enforcement with heuristic (words * 1.3)
  - Truncate least-relevant results when over budget
  - Benchmark: <10ms on 100 nodes
  - Test: seed 100 nodes, query context, verify structure + budget
  - Deliverable: context assembly within latency and token targets

- [T009] Graph store -- entity resolution | est: ~7k tokens | repo: graphbot | depends: T005 | reqs: R005
  - Implement resolve_entity(mention) -> list[tuple[str, float]]
  - Layer 1: exact normalized match (lowercase, strip)
  - Layer 2: Levenshtein edit distance (configurable threshold)
  - Layer 3: BM25 keyword match on name + description
  - Sorted by confidence, top-k configurable (default 5)
  - Zero LLM calls -- purely algorithmic
  - Test: exact matches, fuzzy matches, no-match cases
  - Benchmark: >90% accuracy on 50-pair test set
  - Deliverable: 3-layer resolution with >90% accuracy

## Tier 5 (depends on Tier 4)
- [T010] Graph query performance benchmarks | est: ~5k tokens | repo: graphbot | depends: T008, T009 | reqs: R006
  - Implement scripts/bench_graph.py
  - Seed synthetic graphs at 100, 1K, 5K, 10K nodes
  - Benchmark 2-hop traversal and context assembly at each scale
  - Output table to stdout, append to benchmarks/graph_perf.jsonl
  - Validate: 100 <1ms, 1K <2ms, 5K <4ms, 10K <5ms
  - Deliverable: benchmark script runs, targets met, results persisted

- [T011] Single-task executor (minimal E2E) | est: ~8k tokens | repo: graphbot | depends: T008, T006, T007 | reqs: R010
  - Implement SimpleExecutor in core_gb/executor.py
  - Flow: input -> resolve entities -> assemble context -> route to model -> return result
  - Context injected at beginning of prompt (ADR-009)
  - No decomposition (single node, single LLM call)
  - Returns full ExecutionResult with timing and token tracking
  - Test (mocked): full flow input to result
  - Test (mocked): empty graph still produces valid result
  - Integration test: "What is 247 * 38?" -> 9386 (TESTS.md Test 1)
  - Deliverable: E2E path works, simple task completes in <1s
