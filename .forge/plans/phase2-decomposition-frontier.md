---
spec: phase2-decomposition
total_tasks: 8
estimated_tokens: 70000
depth: standard
---

# Phase 2 Decomposition Frontier

## Tier 1 (parallel -- no dependencies)
- [T012] Intake parser -- rule-based intent classification | est: ~6k tokens | repo: graphbot | reqs: R001
  - IntakeParser class, IntakeResult dataclass
  - Keyword-based domain classification, complexity heuristic
  - Simple task detection (skip decomposition)
  - 20-message accuracy test, latency test <5ms
  - Deliverable: tests/test_core/test_intake.py passes

- [T013] Decomposition schema -- constrained JSON | est: ~5k tokens | repo: graphbot | reqs: R002
  - DecompositionSchema as JSON schema dict
  - Max depth 3, max children 5, MECE task types
  - Valid/invalid tree validation tests
  - Deliverable: schema validates correct trees, rejects invalid ones

## Tier 2 (depends on Tier 1)
- [T014] Decomposition prompt engineering | est: ~7k tokens | repo: graphbot | depends: T013 | reqs: R003
  - DecompositionPrompt class with XML-tag structure
  - Few-shot examples (2 good + 1 bad)
  - Sandwich defense, context injection
  - Token budget validation (<1500 tokens)
  - Deliverable: prompt builds correctly, under budget

- [T015] Tree validator | est: ~6k tokens | repo: graphbot | depends: T013 | reqs: R005
  - validate_tree function
  - Circular dep detection, missing dep, unsatisfied contracts
  - Tests for valid parallel, sequential, and invalid trees
  - Deliverable: all validation tests pass

## Tier 3 (depends on Tier 2)
- [T016] Recursive decomposer | est: ~10k tokens | repo: graphbot | depends: T013, T014, T015 | reqs: R004
  - Decomposer class with async decompose()
  - LLM call via ModelRouter, JSON parsing + validation
  - Retry with json_repair, fallback to single node
  - Tree-to-TaskNode conversion
  - Mocked tests + integration test
  - Deliverable: decomposer produces valid trees from mocked LLM

## Tier 4 (depends on Tier 3)
- [T017] Orchestrator -- intake to execution | est: ~8k tokens | repo: graphbot | depends: T012, T016 | reqs: R006
  - Orchestrator class wiring intake + decomposer + executor
  - Simple tasks -> direct execution, complex -> decompose + sequential execute
  - Aggregated ExecutionResult
  - Deliverable: orchestrator routes correctly, integration test passes

## Tier 5 (integration)
- [T018] Phase 2 integration test + live validation | est: ~6k tokens | repo: graphbot | depends: T017
  - Run canonical test tasks through orchestrator with real models
  - Update PROGRESS.md, TESTS.md benchmark table
  - Commit and push
  - Deliverable: live tests pass, docs updated

- [T019] Phase 2 commit + push | est: ~2k tokens | repo: graphbot | depends: T018
  - Final commit with all Phase 2 work
  - Push to GitHub
  - Update forge state
