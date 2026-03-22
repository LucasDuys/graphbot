---
domain: phase11-viability
status: approved
created: 2026-03-22
complexity: complex
linked_repos: []
---

# Phase 11: From Prototype to Viable Agent

## Overview

Five critical fixes to make GraphBot actually usable as a coding agent.
Based on deep research comparing OpenClaw architecture, scientific literature
on decomposition quality, and an honest codebase audit.

METHODOLOGY: Research-first for each fix. Test-first implementation. Live
validation with real tasks after each fix.

## Requirements

### R001: Structured Tool Parameters in TaskNode
The decomposer must output explicit tool methods and parameters, not free text.
**Acceptance Criteria:**
- [ ] TaskNode gets new fields: `tool_method: str | None`, `tool_params: dict[str, str]`
- [ ] DecompositionSchema updated: nodes can include `tool_method` and `tool_params`
- [ ] DecompositionPrompt updated: "For tool tasks, specify tool_method (file_read, web_search, shell_run) and tool_params ({path: ..., query: ...})"
- [ ] ToolRegistry uses tool_method + tool_params instead of regex-based description parsing
- [ ] Domain override still works as fallback when tool_method not provided
- [ ] Test: decompose "Read README.md" -> node has tool_method="file_read", tool_params={"path": "README.md"}
- [ ] Test: decompose "Search web for X" -> node has tool_method="web_search", tool_params={"query": "X"}
- [ ] Test: ToolRegistry executes node using tool_params directly
- [ ] Benchmark: re-run 10 real tasks, verify tool_method is set on >50% of tool-domain nodes

### R002: Smart Decomposition -- Know When NOT to Decompose
Not every task benefits from decomposition. Comparative/integrated tasks should stay as single LLM calls.
**Acceptance Criteria:**
- [ ] IntakeParser gets `task_type` classification: ATOMIC, DATA_PARALLEL, SEQUENTIAL, INTEGRATED
- [ ] ATOMIC: simple single-step (math, facts) -> direct execution, no decomposition
- [ ] DATA_PARALLEL: independent subtasks (weather in 3 cities) -> decompose + parallel
- [ ] SEQUENTIAL: ordered chain (read -> parse -> format) -> decompose + sequential
- [ ] INTEGRATED: requires cross-cutting reasoning (compare X vs Y) -> single LLM call with full context, no decomposition
- [ ] Orchestrator routes based on task_type, not just is_simple
- [ ] Test: "Compare PostgreSQL vs MongoDB" classified as INTEGRATED -> single call
- [ ] Test: "Weather in 3 cities" classified as DATA_PARALLEL -> decomposed
- [ ] Test: "Read file then summarize" classified as SEQUENTIAL -> decomposed
- [ ] Quality test: INTEGRATED tasks produce better output than decomposed (manual check)

### R003: Code Editing Agent
Multi-turn agent that can read, analyze, edit, and test code.
**Acceptance Criteria:**
- [ ] `core_gb/code_agent.py` with CodeEditAgent class
- [ ] `async edit(instruction: str, file_path: str) -> CodeEditResult`
- [ ] Flow: read file -> understand context -> generate diff -> apply edit -> verify
- [ ] Uses FileTool.read() and FileTool.edit() internally
- [ ] Generates search-and-replace edits (old_text -> new_text), not full rewrites
- [ ] Can chain: edit -> run pytest -> if fails, retry edit
- [ ] Max 3 retry cycles to prevent infinite loops
- [ ] ToolRegistry routes "edit", "fix", "refactor", "modify" tasks to CodeEditAgent
- [ ] Test: "Fix the typo in README.md where 'teh' should be 'the'" -> actual file edit
- [ ] Test: edit that breaks tests triggers retry
- [ ] Test: max retries exceeded returns failure gracefully

### R004: Tool Failure Recovery
Retry failed tools and fall back to LLM when tools can't handle it.
**Acceptance Criteria:**
- [ ] DAGExecutor retries failed tool calls once with 1s backoff
- [ ] On second failure: falls back to LLM call (SimpleExecutor) for that node
- [ ] Error context preserved: "Tool web_search failed (timeout), falling back to LLM"
- [ ] Node result includes `fallback_used: bool` field
- [ ] Test: mock tool that fails once then succeeds -> retry works
- [ ] Test: mock tool that always fails -> falls back to LLM
- [ ] Test: error message includes tool name and failure reason

### R005: Execution Observability
Surface tool decisions and execution details to users.
**Acceptance Criteria:**
- [ ] SSE events include: tool.invoke (method, params), tool.result (success, output preview), decomposition.reasoning
- [ ] DAGExecutor emits events via callback during execution
- [ ] UI StatusStepper shows which tool is being used ("Searching web..." not just "Executing")
- [ ] NodeDetail panel shows: tool used, parameters, response time, retry count
- [ ] Error nodes show: what went wrong, which fallback was used
- [ ] Test: SSE stream includes tool.invoke events for tool-domain nodes
