---
domain: phase1-foundation
status: approved
created: 2026-03-21
complexity: medium
linked_repos: []
---

# Phase 1: Foundation -- Graph + Types + Single Model

## Overview

Build the foundational layers of GraphBot bottom-up: environment and tooling, then the Kuzu graph store with temporal schema, then the OpenRouter model provider, then a minimal single-task execution path that connects them. Every layer is designed for scale (async interfaces, typed contracts, connection pooling, benchmark validation) even though Phase 1 only exercises the simplest case.

Approach: Bottom-Up Foundation. Each layer is tested and benchmarked before the next builds on it. OpenRouter is the single provider gateway. Raw Kuzu with custom temporal schema (no Graphiti).

## Requirements

### R001: Environment Setup and Dependency Installation
Install all Phase 1 dependencies and verify the toolchain works.
**Acceptance Criteria:**
- [ ] `pyproject.toml` (or `requirements.txt`) includes: `kuzu>=0.11.0`, `litellm`, `langsmith`, `aiolimiter`, `aiobreaker`, `pytest`, `pytest-asyncio`
- [ ] `pip install -e .` succeeds in a fresh venv on Windows 11
- [ ] `import kuzu` succeeds and can create an in-memory database
- [ ] `import litellm` succeeds
- [ ] Existing tests in `tests/` pass (`pytest tests/ -v` exits 0)
- [ ] Python 3.11+ confirmed (for `graphlib.TopologicalSorter` streaming support)

### R002: Kuzu Graph Store -- Schema Creation
Implement the graph schema from `graph/schema.py` as live Kuzu DDL.
**Acceptance Criteria:**
- [ ] `GraphStore` class in `graph/store.py` with async-compatible interface
- [ ] `GraphStore.__init__(db_path: str | None)` -- `None` creates in-memory DB for testing
- [ ] `GraphStore.initialize()` creates all 10 node tables and 12 edge tables from `graph/schema.py`
- [ ] Schema creation is idempotent (calling `initialize()` twice does not error)
- [ ] All node tables have `id: STRING PRIMARY KEY`
- [ ] Temporal nodes (Memory, Task, etc.) have `valid_from: TIMESTAMP` and `valid_until: TIMESTAMP` columns
- [ ] `GraphStore.close()` cleanly releases the Kuzu connection
- [ ] Test: create store, initialize, verify all tables exist via `CALL show_tables()`

### R003: Kuzu Graph Store -- CRUD Operations
Implement typed create/read/update/delete operations on graph nodes and edges.
**Acceptance Criteria:**
- [ ] `GraphStore.create_node(table: str, properties: dict) -> str` returns node ID
- [ ] `GraphStore.get_node(table: str, node_id: str) -> dict | None`
- [ ] `GraphStore.update_node(table: str, node_id: str, properties: dict) -> bool`
- [ ] `GraphStore.delete_node(table: str, node_id: str) -> bool`
- [ ] `GraphStore.create_edge(table: str, from_id: str, to_id: str, properties: dict) -> bool`
- [ ] `GraphStore.query(cypher: str, params: dict | None) -> list[dict]` for raw Cypher
- [ ] All methods use parameterized queries (no string interpolation -- injection safe)
- [ ] Test: full CRUD cycle on Entity, Memory, and Task node types
- [ ] Test: create edges between nodes and query traversals

### R004: Kuzu Graph Store -- Context Assembly
Implement the context assembly pipeline that gathers relevant graph context for a task.
**Acceptance Criteria:**
- [ ] `GraphStore.get_context(entity_ids: list[str], max_tokens: int = 2500) -> GraphContext`
- [ ] Performs 2-hop traversal from given entities
- [ ] Returns `GraphContext` dataclass (from `core_gb/types.py`) with entities, memories, relationships
- [ ] Respects `max_tokens` budget -- truncates least-relevant results when over budget
- [ ] Token counting uses a simple heuristic (words * 1.3) not a tokenizer dependency
- [ ] Benchmark: <10ms for context assembly on 100 nodes (pytest-benchmark or manual timing)
- [ ] Test: seed 100 nodes, query context, verify result structure and token budget respected

### R005: Kuzu Graph Store -- Entity Resolution
Implement entity resolution to match incoming text to existing graph entities.
**Acceptance Criteria:**
- [ ] `GraphStore.resolve_entity(mention: str) -> list[tuple[str, float]]` returns (entity_id, confidence) pairs
- [ ] Layer 1: Exact normalized match (lowercase, strip whitespace) -- expected to catch 60-70%
- [ ] Layer 2: Edit distance (Levenshtein) for typo tolerance -- threshold configurable
- [ ] Layer 3: BM25 keyword match against entity name + description fields
- [ ] Results sorted by confidence descending, top-k configurable (default 5)
- [ ] No LLM calls -- purely algorithmic (LLM fallback deferred to Phase 2+)
- [ ] Benchmark: >90% accuracy on a test set of 50 mention/entity pairs
- [ ] Test: seed entities, resolve exact matches, fuzzy matches, and no-match cases

### R006: Graph Query Performance Benchmarks
Validate that Kuzu meets performance targets at various scales.
**Acceptance Criteria:**
- [ ] `scripts/bench_graph.py` seeds synthetic graphs at 100, 1K, 5K, 10K nodes
- [ ] Benchmarks 2-hop traversal latency at each scale
- [ ] Benchmarks context assembly latency at each scale
- [ ] Results printed as table and appended to `benchmarks/graph_perf.jsonl`
- [ ] 100 nodes: <1ms, 1K: <2ms, 5K: <4ms, 10K: <5ms (targets from TESTS.md)
- [ ] Test: benchmark script runs without error and produces valid output

### R007: OpenRouter Model Provider
Implement the model provider abstraction with OpenRouter as the backend.
**Acceptance Criteria:**
- [ ] `ModelProvider` protocol/ABC in `models/base.py` with async `complete(messages, model, **kwargs) -> CompletionResult`
- [ ] `CompletionResult` dataclass with: `content: str`, `model: str`, `tokens_in: int`, `tokens_out: int`, `latency_ms: float`, `cost: float`
- [ ] `OpenRouterProvider` in `models/openrouter.py` implements `ModelProvider`
- [ ] Uses `litellm.acompletion()` with `model="openrouter/{model_name}"` format
- [ ] API key read from `OPENROUTER_API_KEY` env var (no hardcoding)
- [ ] Structured error handling: rate limit errors raise `RateLimitError`, auth errors raise `AuthError`, others raise `ProviderError`
- [ ] All errors include the provider name and model in the message
- [ ] Test: mock litellm.acompletion, verify correct routing, error handling, result parsing
- [ ] Integration test (requires API key): send a simple completion, verify response structure

### R008: Model Router (Scale-Ready)
Implement a model router that selects models based on task complexity, designed for multi-provider rotation later.
**Acceptance Criteria:**
- [ ] `ModelRouter` in `models/router.py` with async `route(task: TaskNode) -> CompletionResult`
- [ ] Accepts a `ModelProvider` (single provider for Phase 1, list of providers for Phase 3+)
- [ ] Model selection based on `task.complexity` field (1-5 scale from TaskNode)
- [ ] Default model mapping configurable via dict (not hardcoded):
  - complexity 1-2: small model (e.g., `meta-llama/llama-3.1-8b-instruct`)
  - complexity 3: medium model (e.g., `meta-llama/llama-3.3-70b-versatile`)
  - complexity 4-5: large model (e.g., `google/gemini-2.5-pro`)
- [ ] Interface supports adding rate limiters and circuit breakers (Phase 3 wiring)
- [ ] Test: route tasks of different complexities, verify correct model selection
- [ ] Test: provider failure raises appropriate error through router

### R009: LangSmith Observability
Enable LangSmith tracing for all LLM calls via LiteLLM callback.
**Acceptance Criteria:**
- [ ] LangSmith enabled via `litellm.success_callback = ["langsmith"]` in provider initialization
- [ ] Environment variables: `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT="graphbot"`, `LANGCHAIN_TRACING_V2="true"`
- [ ] When env vars are missing, tracing is silently disabled (no crash)
- [ ] Test: with mock callback, verify tracing is registered
- [ ] Manual verification: make an OpenRouter call, confirm trace appears in LangSmith dashboard

### R010: Single-Task Executor (Minimal E2E Path)
Wire graph + provider into a minimal executor that handles a single atomic task.
**Acceptance Criteria:**
- [ ] `SimpleExecutor` in `core_gb/executor.py` with async `execute(task: str) -> ExecutionResult`
- [ ] Flow: parse input -> resolve entities in graph -> assemble context -> call model via router -> return result
- [ ] Uses `GraphStore` for context, `ModelRouter` for LLM call
- [ ] Returns `ExecutionResult` (from `core_gb/types.py`) with: output, tokens, latency, model_used, context_used
- [ ] No decomposition -- single node, single LLM call (decomposition is Phase 2)
- [ ] Context injected at beginning of prompt (per ADR-009: beginning/end, never middle)
- [ ] Test: with mocked provider, verify full flow from input to result
- [ ] Test: with empty graph (no context), still produces valid result
- [ ] Integration test: "What is 247 * 38?" returns correct answer (Test 1 from TESTS.md)

### R011: Core Types Validation
Verify and extend existing types to support Phase 1 implementation.
**Acceptance Criteria:**
- [ ] `core_gb/types.py` types are importable and constructable: `TaskNode`, `ExecutionResult`, `Pattern`, `GraphContext`
- [ ] `TaskNode` has fields: `id`, `task`, `complexity` (1-5), `provides`, `consumes`, `status`
- [ ] `ExecutionResult` has fields: `output`, `tokens_in`, `tokens_out`, `latency_ms`, `model_used`, `cost`, `context_tokens`
- [ ] `GraphContext` has fields: `entities`, `memories`, `relationships`, `total_tokens`
- [ ] All dataclasses are frozen (immutable) for safe concurrent use
- [ ] Existing tests in `tests/test_core/test_types.py` still pass
- [ ] Test: construct each type, verify immutability (assignment raises FrozenInstanceError)

## Future Considerations (NOT Phase 1)

- Multi-provider rotation with rate limiting and circuit breaking (Phase 3)
- Recursive decomposition into DAG (Phase 2)
- Parallel DAG execution (Phase 3)
- Pattern cache and template instantiation (Phase 4)
- Graphiti evaluation for entity resolution upgrade (Phase 2+)
- Langfuse/custom tracing decorator (Phase 3+)
