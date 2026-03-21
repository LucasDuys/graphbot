---
domain: phase3-dag-executor
status: approved
created: 2026-03-21
complexity: complex
linked_repos: []
---

# Phase 3: Parallel DAG Executor Pipeline

## Overview

Replace the Phase 2 sequential leaf execution with a true parallel DAG executor using asyncio + graphlib.TopologicalSorter streaming dispatch. When a node completes, immediately unblock and start dependents. Add rate limiting per provider and circuit breaking for resilience.

## Requirements

### R001: Async DAG Executor
Replace sequential execution with parallel streaming dispatch.
**Acceptance Criteria:**
- [ ] `DAGExecutor` class in `core_gb/dag_executor.py`
- [ ] `async execute(nodes: list[TaskNode], store: GraphStore, router: ModelRouter) -> ExecutionResult`
- [ ] Uses `graphlib.TopologicalSorter` with streaming dispatch (`prepare()`, `get_ready()`, `done()`)
- [ ] Independent nodes execute concurrently via `asyncio.gather()`
- [ ] When a node completes, calls `sorter.done()` to immediately unblock dependents
- [ ] Global concurrency limit via `asyncio.Semaphore(10)` (configurable)
- [ ] Data forwarding between nodes via provides/consumes contracts
- [ ] Returns aggregated ExecutionResult with all node results
- [ ] Test: 3 independent nodes execute concurrently (verify via timing -- parallel < 2x single)
- [ ] Test: sequential chain A -> B -> C executes in order
- [ ] Test: mixed DAG (parallel gather + sequential synthesis) executes correctly
- [ ] Test: node failure doesn't crash other branches

### R002: Rate Limiter Integration
Per-provider rate limiting using aiolimiter.
**Acceptance Criteria:**
- [ ] `RateLimiter` wrapper in `models/rate_limiter.py`
- [ ] Wraps `aiolimiter.AsyncLimiter` with per-provider instances
- [ ] `async acquire(provider: str) -> None` -- blocks until rate limit allows
- [ ] Configurable: max_rate (requests per period), time_period (seconds)
- [ ] Default: 30 requests per 60 seconds (matches Groq free tier)
- [ ] Integrated into ModelRouter -- acquire before each completion call
- [ ] Test: rate limiter delays requests when at capacity
- [ ] Test: multiple providers have independent limiters

### R003: Circuit Breaker Integration
Per-provider circuit breaking using aiobreaker.
**Acceptance Criteria:**
- [ ] `CircuitBreakerManager` in `models/circuit_breaker.py`
- [ ] Wraps `aiobreaker.CircuitBreaker` with per-provider instances
- [ ] States: CLOSED (normal), OPEN (failing, reject calls), HALF_OPEN (test recovery)
- [ ] Config: fail_max=3, timeout_duration=30s
- [ ] When circuit opens, router skips that provider (Phase 3 has single provider, but interface ready for multi)
- [ ] Test: 3 consecutive failures trips the breaker
- [ ] Test: after timeout, breaker allows one test call (half-open)
- [ ] Test: successful test call closes the breaker

### R004: Update Orchestrator for Parallel Execution
Wire DAGExecutor into the Orchestrator.
**Acceptance Criteria:**
- [ ] Orchestrator uses DAGExecutor instead of sequential leaf execution for complex tasks
- [ ] Simple tasks still route directly to SimpleExecutor
- [ ] Rate limiter and circuit breaker wired into ModelRouter
- [ ] Test: orchestrator uses DAGExecutor for multi-node decompositions
- [ ] Integration test: "Compare weather in 3 cities" executes leaves in parallel

### R005: Traced Decorator for Pipeline Stages
Custom tracing for non-LLM operations.
**Acceptance Criteria:**
- [ ] `@traced` decorator in `core_gb/tracing.py`
- [ ] Records: stage_name, duration_ms, extra metadata dict
- [ ] Stores trace entries in a list (in-memory for Phase 3, graph storage Phase 5)
- [ ] Applied to: intake.parse, decomposer.decompose, dag_executor.execute
- [ ] Test: traced function records correct timing and metadata
- [ ] Test: trace entries accessible after execution
