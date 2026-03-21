---
spec: phase9-tools-and-scale
total_tasks: 14
estimated_tokens: 120000
depth: thorough
---

# Phase 9 Tools + Scale Frontier

## Tier 1 -- RESEARCH ONLY (parallel, no code)
- [T058] Research: web scraping best practices | est: ~8k | reqs: R001
- [T059] Research: file operations best practices | est: ~6k | reqs: R002
- [T060] Research: shell execution best practices | est: ~6k | reqs: R003
- [T061] Research: non-technical visualization UX | est: ~6k | reqs: R004

## Tier 2 -- IMPLEMENTATION (parallel, depends on research)
- [T062] Web tools (Playwright + content cleaning) | est: ~12k | depends: T058 | reqs: R005
- [T063] File tools (read/write/edit/list/search) | est: ~10k | depends: T059 | reqs: R006
- [T064] Shell tools (sandboxed execution) | est: ~8k | depends: T060 | reqs: R007

## Tier 3 -- WIRING (depends on tools)
- [T065] Tool registry + domain routing | est: ~8k | depends: T062, T063, T064 | reqs: R008
- [T066] Enhanced UI narration + progress | est: ~10k | depends: T061 | reqs: R010

## Tier 4 -- PROOF (depends on wiring)
- [T067] Real-world task suite (10 tasks) | est: ~8k | depends: T065 | reqs: R009
- [T068] Run real tasks with live models + tools | est: ~10k | depends: T067
- [T069] Document results + update README | est: ~5k | depends: T068

## Tier 5
- [T070] Commit + push Phase 9 | est: ~3k | depends: T069
