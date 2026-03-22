---
domain: phase13-polish-and-scale
status: approved
created: 2026-03-22
complexity: complex
linked_repos: []
---

# Phase 13: Polish, Demo-Ready, Channels, Benchmarks

## Overview

Four workstreams: quick fixes, demo-ready UI, channel integration for daily use,
and standardized benchmarks. Prioritized for maximum impact.

## Requirements

### R001: Quick Fixes (30 min each)
**Acceptance Criteria:**
- [ ] Shell tasks: fix output interpretation (parse stdout for answer, not just return raw)
- [ ] Git history: squash Nanobot's 1385 commits into single "initial fork" so contributor count shows only Lucas + Claude
- [ ] Fix the 1 pre-existing test failure (test_observability callback test)
- [ ] All 10/10 real tasks pass

### R002: Demo-Ready UI with Live Animations
**Acceptance Criteria:**
- [ ] DAG nodes animate individually in real-time (not batched at end)
- [ ] SSE events drive per-node status transitions: pending -> running (amber pulse) -> completed (green)
- [ ] Edge animation: dashed cyan line flows when data transfers between nodes
- [ ] Knowledge graph panel (bottom-right): D3 force layout showing entities/relationships
- [ ] Empty state: centered message with subtle animation when no task running
- [ ] Dark mode toggle in header (persisted to localStorage)
- [ ] Input shows elapsed timer during execution
- [ ] Responsive: works on 1440px+ screens
- [ ] Screenshot-worthy: capture 3 screenshots for README (idle, executing, complete)

### R003: Channel Integration (Telegram)
Wire GraphBot orchestrator into Nanobot's Telegram channel for daily mobile use.
**Acceptance Criteria:**
- [ ] `nanobot/channels/graphbot_telegram.py` bridges Telegram messages to Orchestrator
- [ ] Receives message via Telegram -> runs through Orchestrator.process() -> sends response
- [ ] Shows node count + cost in response footer
- [ ] Persistent graph (data/graphbot.db) across sessions
- [ ] Config: TELEGRAM_BOT_TOKEN in .env.local
- [ ] Test: send message, receive response (mocked Telegram API)

### R004: Multi-Provider Rotation
Rotate across free LLM providers for sustained $0 usage.
**Acceptance Criteria:**
- [ ] ModelRouter accepts list of providers (not just one)
- [ ] Rotation strategy: try primary, on rate limit -> next provider, on circuit open -> skip
- [ ] Providers: OpenRouter (primary), Google AI Studio (Gemini 2.5 Pro free, 5 RPM), Groq direct (30 RPM free)
- [ ] Each provider has independent rate limiter + circuit breaker
- [ ] Test: primary rate limited -> falls back to secondary
- [ ] Test: all providers down -> graceful error

### R005: Standardized Benchmark (GAIA Level 1)
Run GraphBot against GAIA benchmark for comparable results.
**Acceptance Criteria:**
- [ ] scripts/run_gaia.py downloads GAIA Level 1 dataset (simple tasks, no tools needed)
- [ ] Runs each task through Orchestrator
- [ ] Compares output to ground truth
- [ ] Reports accuracy, tokens, cost
- [ ] Results saved to benchmarks/gaia_results.json
- [ ] At least 20 GAIA tasks attempted

### R006: Pattern Cache Warming
Build up pattern templates through repeated usage.
**Acceptance Criteria:**
- [ ] scripts/warm_cache.py runs 30+ diverse tasks through Orchestrator
- [ ] After warming: pattern cache has 10+ templates
- [ ] Second run of same task types shows cache hits (fewer LLM calls)
- [ ] Graph stats show growth (Task nodes, Pattern nodes, ExecutionTree nodes)
- [ ] Benchmark: cache-warmed run uses 30%+ fewer tokens than cold run
