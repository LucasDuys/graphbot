---
domain: phase12-remaining-gaps
status: approved
created: 2026-03-22
complexity: medium
linked_repos: []
---

# Phase 12: Fix Remaining Gaps

## Requirements

### R001: Fix Shell Command Execution
Shell tasks fail because commands with quotes/special chars get mangled.
**Acceptance Criteria:**
- [ ] ShellTool properly handles quoted commands in task descriptions
- [ ] ToolRegistry extracts commands from backticks, quotes, or "run X" patterns
- [ ] Commands like `git log --oneline -10` and `python -m pytest tests/ --co -q` execute correctly
- [ ] Test: "Run git log --oneline -5" succeeds with actual git output
- [ ] Test: "Run python --version" succeeds
- [ ] Re-run real tasks: shell tasks 2/2 success

### R002: Tool Usage Metric Tracking
Aggregated ExecutionResult doesn't show which nodes used tools vs LLM.
**Acceptance Criteria:**
- [ ] ExecutionResult gets `tools_used: int` and `llm_calls: int` fields
- [ ] DAGExecutor tracks tool vs LLM usage per node
- [ ] run_real_tasks.py correctly reports tool usage from result fields
- [ ] Test: DAG with mixed tool/LLM nodes reports correct counts

### R003: SSE Observability Events for Tools
Wire tool execution details into the SSE stream for the UI.
**Acceptance Criteria:**
- [ ] SSE backend emits tool.invoke (method, params) and tool.result (success, preview) events
- [ ] Orchestrator accepts optional event callback, passes to DAGExecutor
- [ ] UI StatusStepper shows tool names ("Reading file..." not just "Executing")
- [ ] Test: SSE stream includes tool events for tool-domain nodes

### R004: Update All Documentation
Reflect current state accurately.
**Acceptance Criteria:**
- [ ] PROGRESS.md updated with Phase 11+12 completion
- [ ] README.md benchmarks updated with latest numbers
- [ ] benchmarks/RESULTS.md updated
- [ ] All test counts accurate
