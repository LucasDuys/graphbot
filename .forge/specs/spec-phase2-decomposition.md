---
domain: phase2-decomposition
status: approved
created: 2026-03-21
complexity: complex
linked_repos: []
---

# Phase 2: Intake Parser + Recursive Decomposer

## Overview

Build the intake parser (rule-based, zero-cost intent classification) and the recursive decomposer that breaks complex tasks into DAG-structured subtask trees using constrained JSON output from small models. This phase bridges Phase 1's single-task executor to Phase 3's parallel DAG executor.

Approach: Intake first (zero LLM cost), then decomposer with constrained JSON, then tree validation. OpenRouter for model calls.

## Requirements

### R001: Intake Parser -- Rule-Based Intent Classification
Zero-cost, zero-token intent classification using pattern matching.
**Acceptance Criteria:**
- [ ] `IntakeParser` class in `core_gb/intake.py`
- [ ] `parse(message: str) -> IntakeResult` returns domain, complexity estimate, extracted entities
- [ ] `IntakeResult` dataclass with: `domain: Domain`, `complexity: int (1-5)`, `entities: list[str]`, `is_simple: bool`, `raw_message: str`
- [ ] Domain classification via keyword lists per domain (file, web, code, comms, system, synthesis)
- [ ] Complexity estimation heuristic: word count, conjunction count ("and", "then", commas), question depth
- [ ] Simple task detection: single-domain, complexity <= 2, no conjunctions -> skip decomposition
- [ ] Test: classify 20 sample messages across all domains with >80% accuracy
- [ ] Test: complexity estimation within +/-1 of expected for 15 test cases
- [ ] Latency: <5ms per parse (no LLM calls)

### R002: Decomposition Schema -- Constrained JSON Output
Define the JSON schema that the decomposer model must produce.
**Acceptance Criteria:**
- [ ] `DecompositionSchema` in `core_gb/decomposer.py` as a JSON schema dict
- [ ] Schema defines: task tree with nodes having `id`, `description`, `domain`, `complexity`, `depends_on`, `provides`, `consumes`
- [ ] Max depth: 3 levels (configurable)
- [ ] Max children per node: 5 (configurable)
- [ ] Each leaf must be `is_atomic: true`
- [ ] MECE task types enforced: RETRIEVE, WRITE, THINK, CODE (from ROMA)
- [ ] Schema is valid JSON Schema (validate with jsonschema library)
- [ ] Test: validate 5 example trees against schema -- all pass
- [ ] Test: validate 3 invalid trees against schema -- all fail with clear errors

### R003: Decomposition Prompt Engineering
Build the prompt that drives small models to produce valid task trees.
**Acceptance Criteria:**
- [ ] `DecompositionPrompt` class in `core_gb/decomposer.py`
- [ ] `build(task: str, context: GraphContext) -> list[dict]` returns messages for the LLM
- [ ] System prompt uses XML-tag structure: `<rules>`, `<output_schema>`, `<examples>` (ADR-016)
- [ ] Includes 2-3 few-shot examples of correct decompositions (parallel + sequential)
- [ ] Includes 1 bad example with explanation of why it's wrong
- [ ] Sandwich defense: output format restated after task description
- [ ] Context injected at beginning of system message (ADR-009)
- [ ] Total prompt tokens < 1500 (leaves room for model output within 2K budget)
- [ ] Test: prompt builds correctly with and without graph context
- [ ] Test: prompt token count stays under budget

### R004: Recursive Decomposer
The core decomposition engine that calls the LLM and validates the output.
**Acceptance Criteria:**
- [ ] `Decomposer` class in `core_gb/decomposer.py`
- [ ] `async decompose(task: str, context: GraphContext, max_depth: int = 3) -> list[TaskNode]`
- [ ] Calls ModelRouter with complexity 2-3 (decomposition needs a capable model)
- [ ] Parses JSON response, validates against DecompositionSchema
- [ ] On invalid JSON: retry once with `json_repair` library (already in deps)
- [ ] On second failure: fall back to single-node atomic task (no decomposition)
- [ ] Converts validated tree to list of TaskNode objects with proper parent/child relationships
- [ ] Respects max_depth -- if model produces deeper tree, flatten
- [ ] Assigns unique IDs to all nodes
- [ ] Test (mocked LLM): valid decomposition produces correct TaskNode tree
- [ ] Test (mocked LLM): invalid JSON triggers retry then fallback
- [ ] Test (mocked LLM): single atomic task returns 1-node list
- [ ] Integration test: decompose "Weather in Amsterdam, London, Berlin" -> parallel tree with 3+ leaves

### R005: Tree Validator
Validate decomposition trees for structural correctness.
**Acceptance Criteria:**
- [ ] `validate_tree(nodes: list[TaskNode]) -> list[str]` returns list of error messages (empty = valid)
- [ ] Checks: no circular dependencies
- [ ] Checks: all `requires` references point to existing node IDs
- [ ] Checks: all leaves are `is_atomic: true`
- [ ] Checks: root node has no parent
- [ ] Checks: every non-root node has a valid parent
- [ ] Checks: `provides`/`consumes` contracts are satisfiable (every consumed key is provided by a dependency)
- [ ] Test: valid parallel tree passes
- [ ] Test: valid sequential tree passes
- [ ] Test: circular dependency detected
- [ ] Test: missing dependency detected
- [ ] Test: unsatisfied data contract detected

### R006: Orchestrator -- Intake to Execution
Wire intake + decomposer + executor into a single orchestration flow.
**Acceptance Criteria:**
- [ ] `Orchestrator` class in `core_gb/orchestrator.py`
- [ ] `async process(message: str) -> ExecutionResult`
- [ ] Flow: IntakeParser.parse -> if simple: SimpleExecutor.execute -> else: Decomposer.decompose -> (Phase 3 DAG executor, for now: execute leaves sequentially)
- [ ] For Phase 2: sequential leaf execution (iterate nodes in topological order, execute each with SimpleExecutor)
- [ ] Graph context assembled once and shared across all nodes
- [ ] Returns aggregated ExecutionResult with all node results
- [ ] Test (mocked): simple task routes to SimpleExecutor directly
- [ ] Test (mocked): complex task routes through decomposition
- [ ] Test (mocked): sequential execution of 3-node tree produces correct output
- [ ] Integration test: "What is 247 * 38?" -> direct execution (no decomposition)

## Future Considerations (NOT Phase 2)
- Parallel DAG execution (Phase 3)
- Pattern matching before decomposition (Phase 4)
- Constrained JSON via outlines/guidance library (evaluate if model JSON mode is insufficient)
