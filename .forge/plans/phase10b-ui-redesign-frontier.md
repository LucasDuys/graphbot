---
spec: phase10b-ui-redesign
total_tasks: 9
estimated_tokens: 80000
depth: thorough
---

# Phase 10B UI Redesign Frontier

## Tier 1 -- RESEARCH (no code)
- [T076] Research: Linear design system analysis | est: ~6k | reqs: R001

## Tier 2 -- FOUNDATION (from research)
- [T077] Design tokens + global CSS (colors, type, spacing, shadows) | est: ~8k | depends: T076 | reqs: R002
- [T078] Layout shell -- 3-panel dashboard | est: ~10k | depends: T076 | reqs: R003

## Tier 3 -- COMPONENTS (parallel)
- [T079] DAG canvas redesign (card nodes, bezier edges, animations) | est: ~12k | depends: T077, T078 | reqs: R004
- [T080] Task input bar (command-bar style) | est: ~6k | depends: T077, T078 | reqs: R005
- [T081] Status + progress stepper | est: ~6k | depends: T077, T078 | reqs: R008

## Tier 4 -- PANELS
- [T082] Node detail panel (collapsible cards) | est: ~8k | depends: T079 | reqs: R006
- [T083] Result panel (markdown, metrics, copy) | est: ~8k | depends: T079 | reqs: R007

## Tier 5 -- INTEGRATION
- [T084] Full E2E: connect to backend, verify build, commit + push | est: ~6k | depends: T080, T081, T082, T083
