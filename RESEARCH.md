# GraphBot Research Log

All research findings logged here with sources, conclusions, and action items.

## Research Queue
_All initial research completed 2026-03-20._
- [ ] Evaluate Langfuse vs custom tracing for GraphBot observability (Phase 3+)

## Completed Research

### 2026-03-20 -- Kuzu Graph Database

**CRITICAL: Kuzu was archived Oct 10, 2025** (Apple acquisition). Last release: v0.11.3 on PyPI. Still installable and functional but no future updates.

**Findings:**
- Schema via Cypher DDL, not Python ORM: `CREATE NODE TABLE Person(id STRING PRIMARY KEY, name STRING)`
- Temporal: native TIMESTAMP/DATE types, but no bi-temporal semantics -- model as regular properties (valid_from, valid_until)
- Performance: 2-hop traversal on 100K nodes = 9ms. On 10K nodes: sub-5ms easily achievable. 374x faster than Neo4j on path queries
- Truly embedded: `pip install kuzu`, no server, works on Windows
- Community forks: **LadybugDB** (active, same API), **Vela-Engineering/kuzu** (multi-writer support)

**Decision:** Start with `pip install kuzu` v0.11.3. Monitor LadybugDB for migration when needed.

**Sources:** [Kuzu Python Docs](https://docs.kuzudb.com/client-apis/python/), [Kuzu Benchmarks](https://github.com/prrao87/kuzudb-study), [Vela Partners Analysis](https://www.vela.partners/blog/kuzudb-ai-agent-memory-graph-database)

---

### 2026-03-20 -- Free LLM API Tiers

**Key changes from original assumptions:**
- Groq **deprecated Llama 3.1 70B** (Jan 2025). Use `llama-3.3-70b-versatile` instead
- Llama 4 Scout 17B (MoE, 16 experts) now free on Groq
- Qwen 2.5 Coder 7B **not available** on any free tier. Use Qwen3 32B (Groq) or Qwen3 Coder 480B (OpenRouter)
- Google AI Studio: Gemini 2.5 Pro free at 5 RPM -- arguably best free model quality

**Provider priority (for $0 cost):**

| Provider | Best Free Models | RPM | Daily Limit |
|----------|-----------------|-----|-------------|
| Groq | Llama 3.3 70B, Llama 4 Scout, Qwen3 32B | 30-60 | 1K-14.4K req/day |
| Cerebras | Qwen3 235B, Llama 3.1 8B | 30 | 1M tokens/day |
| Google AI Studio | Gemini 2.5 Pro/Flash | 5-15 | 100-1K req/day |
| OpenRouter | 27 free models (need $10 credit for 1K/day) | 20 | 50 req/day free |

**Action items:**
- [x] Update model references: `llama-3.1-70b` -> `llama-3.3-70b-versatile`
- [x] Update code agent model: Qwen 2.5 Coder 7B -> Qwen3 32B (Groq)
- [ ] Implement multi-provider rotation (Groq -> Cerebras -> Google) for sustained usage
- [ ] Consider Google AI Studio as synthesis model (Gemini 2.5 Pro quality for free)

**Sources:** [Groq Rate Limits](https://console.groq.com/docs/rate-limits), [Cerebras Pricing](https://www.cerebras.ai/pricing), [Free LLM API Resources](https://github.com/cheahjs/free-llm-api-resources)

---

### 2026-03-20 -- Nanobot Codebase Audit

**Size confirmed:** ~4K core agent lines (agent, tools, bus, session). ~28K total with channels, providers, CLI.

**Architecture:** MessageBus pattern. AgentLoop consumes from inbound queue, runs tool-call iteration loop (max 40), publishes to outbound. Single processing lock per agent.

**Keep/Strip decision:**

| Keep | Why |
|------|-----|
| `agent/loop.py` | Core agent iteration loop, clean tool-call pattern |
| `agent/tools/*` (base, registry, mcp, filesystem, shell, web) | Well-designed tool abstraction + MCP integration |
| `agent/context.py` | System prompt builder |
| `bus/` | Elegant MessageBus (~40 lines) |
| `providers/` (base, litellm, registry) | Model-agnostic via LiteLLM, 100+ providers |
| `channels/` (base, manager, registry + needed platforms) | Clean channel abstraction |
| `session/manager.py` | JSONL session storage |

| Strip/Replace | Why |
|---------------|-----|
| `agent/memory.py` | Replace with knowledge graph. Keep consolidation trigger logic only |
| `skills/`, `agent/skills.py` | Replace with GraphBot skill system |
| `cron/`, `heartbeat/` | Optional, defer |
| `cli/` | Rebuild for GraphBot |
| `templates/` | Replace with GraphBot prompts |

**Sources:** [HKUDS/nanobot](https://github.com/HKUDS/nanobot)

---

### 2026-03-20 -- Recursive Decomposition Frameworks

| Framework | Claim | Status | Source |
|-----------|-------|--------|--------|
| ROMA | Recursive plan-execute-aggregate | **Confirmed**, open source | [GitHub](https://github.com/sentient-agi/ROMA), [arXiv:2602.01848](https://arxiv.org/abs/2602.01848) |
| ReAcTree | Agent trees, ~2x vs ReAct | **Confirmed** (2.4x on some benchmarks) | [GitHub](https://github.com/Choi-JaeWoo/ReAcTree), [arXiv:2511.02424](https://arxiv.org/abs/2511.02424) |
| Deep Agent | Hierarchical Task DAGs | **Confirmed**, proprietary (no code) | [arXiv:2502.07056](https://arxiv.org/abs/2502.07056) |
| NoThinking | 9x latency reduction | **Confirmed** | [arXiv:2504.09858](https://arxiv.org/abs/2504.09858) |
| Medprompt | Matches reasoning models | **REVISED** -- matches specialist models, NOT reasoning models | [arXiv:2311.16452](https://arxiv.org/abs/2311.16452) |

**New frameworks discovered:**
- **AgentOrchestra** (89% on GAIA benchmark) -- [arXiv:2506.12508](https://arxiv.org/abs/2506.12508)
- **TDAG** -- dynamic task decomposition + agent generation -- [GitHub](https://github.com/yxwang8775/TDAG)
- **LangChain Deep Agents** -- `write_todos` + `task` subagent spawning -- [GitHub](https://github.com/langchain-ai/deepagents)
- **AdaptThink** -- teaches models WHEN to think vs skip -- [arXiv:2505.13417](https://arxiv.org/abs/2505.13417)

**Critical finding for decomposition on small models:** No base 7B model produces valid task trees without fine-tuning or heavy prompt engineering. Solutions: constrained JSON output, explicit examples, limit depth to 2-3 levels, MECE task type classification (ROMA approach).

**Action items:**
- [ ] Study ROMA's DSPy signatures for decomposition prompt design
- [ ] Study ReAcTree's prompt templates in `resource/wah/sys_prompt/`
- [ ] Implement constrained JSON decoding for decomposition output
- [ ] Revise Medprompt claim in project docs

---

### 2026-03-20 -- Temporal Knowledge Graphs & Entity Resolution

**Graphiti (Zep):** Confirmed open-source temporal KG. Supports Kuzu driver natively (`pip install graphiti-core[kuzu]`). 4 temporal fields: created_at, valid_at, invalid_at, expired_at. 3-tier entity resolution: exact + fuzzy cosine + LLM reasoning.

**BUT:** Graphiti makes multiple LLM calls per `add_episode()` -- expensive/slow with local 7-8B models.

**Entity resolution recommendation (no GPU):**
1. Exact string match (normalized) -- 60-70% of matches
2. Edit distance (Levenshtein) -- typos
3. BM25 keyword matching -- structured data
4. Sentence Transformers CPU (`all-MiniLM-L6-v2`, ~50ms/embedding) -- semantic
5. LLM fallback -- ambiguous cases only

**Context window sweet spot for 7-8B models:** 1,500-3,000 tokens injected context. Quality degrades sharply beyond ~5,000 words. Place relevant info at beginning/end, never middle.

**Decision:** Start with Graphiti + Kuzu driver as prototype. If LLM overhead is a bottleneck, extract temporal schema design and reimplement lighter version with algorithmic entity resolution.

**Sources:** [Graphiti GitHub](https://github.com/getzep/graphiti), [Zep Paper](https://arxiv.org/abs/2501.13956), [Context Rot Research](https://research.trychroma.com/context-rot)

---

### 2026-03-20 -- OpenClaw Competitor Analysis

**Actual name: OpenClaw** (github.com/openclaw/openclaw). 250K+ GitHub stars (surpassed React). 300-500K lines TypeScript.

**All claims confirmed:** Lane Queue (serial), Semantic Snapshots (accessibility trees), lazy skill loading, hybrid memory (JSONL+Markdown+vector), expensive models ($300-750/month), security issues (CVE-2026-25253 CVSS 8.8, 512 vulnerabilities found).

**Recent improvements:** Pluggable ContextEngine (March 2026), cheaper model support for sub-agents, core reduced to ~8MB.

**No native knowledge graph yet.** Community project `openclaw-memory-architecture` provides 12-layer memory with KG, but not in core.

**Ideas to steal:**
1. Semantic Snapshots (accessibility trees for browsing)
2. Pluggable ContextEngine pattern
3. Container isolation (from NanoClaw fork)

**Sources:** [OpenClaw GitHub](https://github.com/openclaw/openclaw), [Architecture Deep Dive](https://gist.github.com/royosherove/971c7b4a350a30ac8a8dad41604a95a0)

---

### 2026-03-20 -- Python Async DAG Execution

**Recommended stack:**
- **DAG engine:** `graphlib.TopologicalSorter` (stdlib, Python 3.9+) with streaming dispatch
- **Rate limiting:** `aiolimiter` (leaky bucket, per-provider instances)
- **Circuit breaker:** `aiobreaker` (per-provider, with fallback routing)
- **Concurrency:** 3-layer: global semaphore(10) + per-provider semaphore + per-provider rate limiter
- **Data forwarding:** Frozen dataclasses as typed contracts in shared `PipelineContext` dict
- **Critical path:** Forward+backward pass, longest-path-first priority scheduling

**Key pattern:** Streaming dispatch (not batch-per-level). When node completes, call `sorter.done()` to immediately unblock dependents.

**Libraries to add:** `aiolimiter`, `aiobreaker`

**Sources:** [graphlib docs](https://docs.python.org/3/library/graphlib.html), [aiolimiter](https://github.com/mjpieters/aiolimiter), [aiobreaker](https://github.com/arlyon/aiobreaker)

---

### 2026-03-20 -- Cross-Referenced from Pitchr/Sidekick Research

Source: `C:\Users\20243455\Downloads\research\` (29 documents from prior Sidekick agent project). Cross-referenced for applicable insights not already captured.

#### New Finding 1: Observability and Pipeline Tracing (NOT YET COVERED)

GraphBot currently has **zero observability infrastructure planned**. The Sidekick research contains detailed patterns for instrumenting LLM pipelines with OpenTelemetry, LangSmith, and Langfuse.

**What applies to GraphBot:**
- Every pipeline stage (Decompose, Schedule, Contextualize, Execute, Verify, Forward, Aggregate) needs timing spans
- Token usage tracking per node and per stage
- Cost tracking per provider per task
- Trace structure: user message -> intake -> graph query -> decomposition -> execution tree -> aggregation -> response
- P50/P95 latency metrics as the benchmark standard (not averages)
- Langfuse is open-source, self-hostable, supports OpenTelemetry -- fits our $0 cost model better than LangSmith

**Recommended approach for GraphBot:**
- Use OpenTelemetry spans (stdlib `time.perf_counter` for simple timing, OTEL for full traces when needed)
- Build a lightweight tracing decorator (`@traced`) that records: stage name, duration_ms, tokens_in, tokens_out, model_used, cache_hit, provider
- Store traces in the knowledge graph (ExecutionTree nodes already in schema)
- Add Langfuse integration as optional (Phase 5)

**Action items:**
- [ ] Add `@traced` decorator pattern to core_gb/executor.py design
- [ ] Add trace fields to ExecutionResult dataclass (per-node timing breakdown)
- [ ] Design trace storage in graph (link to ExecutionTree nodes)
- [ ] Evaluate Langfuse for Phase 5 dashboard

**Sources:** [02-agent-latency-and-performance/benchmarking_llm_pipelines.md], [02-agent-latency-and-performance/observability_and_tracing.md], [Langfuse GitHub](https://github.com/langfuse/langfuse)

---

#### New Finding 2: LLM-as-Judge for Quality Evaluation (PARTIALLY COVERED)

We identified the need for automated quality scoring but had no implementation plan. The Sidekick research has concrete patterns from the MT-Bench paper and Hamel Husain's practical guide.

**What applies to GraphBot:**
- Use **single answer grading** (binary pass/fail) for the Verify stage -- "Is this output correct and complete? Yes/No"
- Use **reference-guided grading** for benchmark regression tests -- compare GraphBot output to a known-good reference
- Use **pairwise comparison** for the North Star benchmark -- "Which response is better, GraphBot's or Nanobot+Sonnet's?"
- Key biases to mitigate: position bias (swap order), verbosity bias (shorter can be better), self-enhancement bias (don't judge with the same model that generated)
- **Use a different free model as judge** than the one that generated the output (e.g., generate with Groq Llama 3.3 70B, judge with Google Gemini 2.5 Flash)

**Action items:**
- [ ] Add LLM-as-judge to Phase 3 Verify stage (binary pass/fail on leaf outputs)
- [ ] Add pairwise comparison to `scripts/compare.py` for North Star benchmark
- [ ] Use different provider for judge vs generator to avoid self-enhancement bias

**Sources:** [04-evaluator-and-output-validation/llm_as_judge_pattern.md], [Zheng et al. 2023 - MT-Bench](https://arxiv.org/abs/2306.05685), [Hamel Husain's guide](https://hamel.dev/blog/posts/llm-judge/)

---

#### New Finding 3: Hybrid Inline/Async Evaluation (NEW)

The Sidekick research describes a hybrid evaluation architecture: fast deterministic checks inline (<5ms, regex/keyword), expensive LLM checks async (background). This maps directly to GraphBot's Verify stage.

**What applies to GraphBot:**
- **Inline (in Verify stage):** JSON parseable? Output matches expected schema? No empty/null responses? Confidence above threshold? All under 5ms, zero LLM calls.
- **Async (post-response):** LLM-as-judge quality scoring, factual accuracy checks against graph, comparison with prior runs. Fire-and-forget, doesn't block response.
- **Short-circuit:** If inline check fails, escalate immediately (retry on stronger model) without waiting for async evaluation.

**Implementation for GraphBot's Verify stage:**
```
Verify Stage:
  1. Schema validation (inline, <1ms) -- is output valid JSON matching contract?
  2. Confidence scoring (inline, <5ms) -- heuristic based on output length, structure
  3. Threshold check (inline, <1ms) -- confidence > 0.7? Pass. < 0.3? Escalate.
  4. LLM-as-judge (async, fire-and-forget) -- quality score stored in graph for learning
```

**Action items:**
- [ ] Design Verify stage as hybrid inline/async
- [ ] Add inline schema validation to core_gb/verifier.py
- [ ] Add async LLM-as-judge as optional post-execution hook

**Sources:** [04-evaluator-and-output-validation/inline_vs_async_evaluation.md], [04-evaluator-and-output-validation/guardrails_and_output_filtering.md]

---

#### New Finding 4: Tool Definition Design as ACI (APPLICABLE)

Anthropic's concept of Agent-Computer Interface (ACI) -- that tool design deserves the same rigor as HCI design. The Sidekick research has concrete patterns.

**What applies to GraphBot sub-agent tools:**
- **Namespace prefixes**: `file_read`, `file_write`, `web_search`, `web_fetch` (not just `read`, `search`)
- **Descriptions as junior dev explanations**: Include what the tool returns, when to use it vs alternatives, edge cases
- **Enum constraints**: Use enums for domains, complexity levels, flow types to constrain model output
- **Error messages that guide correction**: "BrickNotFound: available bricks are [x, y, z]" not just "Error 404"
- **Response format parameter**: Let the model request `concise` vs `detailed` output from tools

**Action items:**
- [ ] Apply ACI design principles to all tool definitions in agents_gb/
- [ ] Add `response_format` parameter to tools that return variable-length data
- [ ] Ensure all tool errors include actionable guidance for the model

**Sources:** [03-prompting-guides/tool_use_prompting_patterns.md], [Anthropic Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

#### New Finding 5: Latency Physics -- Concrete Numbers (VALIDATES ADR-009)

The Sidekick research has concrete latency measurements that validate our context budget decision.

**Key numbers:**
- Every 1,000 additional input tokens adds ~200ms to TTFT (measured on GPT-4 Turbo)
- Output generation is 20-400x slower per token than input processing
- 100 input tokens have approximately the same latency impact as 1 output token
- Tool chain depth is the #1 latency multiplier (each LLM round-trip = TTFT + generation time)

**What this means for GraphBot:**
- Our 2,000-3,000 token context budget (ADR-009) adds ~400-600ms TTFT per leaf node -- acceptable
- Each decomposition call (~400 tokens output at ~50 tok/s) = ~8s on a paid model, but Groq's 300+ tok/s makes it ~1.3s
- The biggest latency risk is **decomposition depth** -- each level adds a full LLM round-trip. Limiting depth to 2-3 levels (ADR-010) is critical.
- Pre-fetching context (speculative pre-fetch in our pipeline) eliminates round-trips, matching the Sidekick "pre-fetch in graph nodes" pattern exactly.

**Sources:** [02-agent-latency-and-performance/latency_tradeoffs.md]

---

#### New Finding 6: System Prompt Structure for Reliable Output (APPLICABLE)

The Sidekick research describes a 4-layer defense pattern for controlling LLM output. Applicable to GraphBot's decomposition and sub-agent prompts.

**What applies:**
- **XML-tag structure**: Wrap critical instructions in `<output_control>`, `<rules>`, `<examples>` tags -- models treat tagged content with higher priority
- **Few-shot examples with bad/good pairs**: Show the model what wrong output looks like alongside correct output. Most reliable way to control format.
- **Sandwich defense**: Restate critical rules after the user input section to reinforce against prompt injection
- **Defense in depth**: Prompt instructions + JSON schema enforcement + application-level validation. Never rely on prompt alone.

**For GraphBot decomposition prompts specifically:**
- Include 2-3 examples of correct decomposition trees (with valid JSON structure)
- Include 1 example of a BAD decomposition (circular deps, non-atomic leaves) with explanation of why it's wrong
- Wrap the output format specification in `<output_schema>` tags
- Restate "output must be valid JSON matching the schema" after the task description

**Action items:**
- [ ] Apply XML-tag structure to decomposition prompt (core_gb/decomposer.py)
- [ ] Add bad/good few-shot examples to decomposition prompt
- [ ] Apply sandwich defense pattern to all sub-agent prompts

**Sources:** [03-prompting-guides/system_prompt_best_practices.md], [03-prompting-guides/anthropic_official_prompting.md]

---

#### Findings Already Covered (no new info)

| Topic | Sidekick Research | GraphBot Status |
|-------|------------------|-----------------|
| Context window management | Token limits, compression, selective loading | Already covered in ADR-009 (context budget) |
| RAG vs full context | When to use RAG vs full context vs lazy loading | N/A -- GraphBot uses graph queries, not RAG |
| Parallel tool execution | API-level and graph-level parallelism | Already covered in ADR-007 (streaming DAG dispatch) |
| Lost-in-the-middle effect | Info at beginning/end, not middle | Already in ADR-009 research findings |
| Prompt caching | Anthropic's cache_control for repeated context | Not directly applicable (free APIs don't support prompt caching) but pattern cache achieves same effect |
| UI/UX panel design | Copilot panel patterns, resizable panels | Not applicable to GraphBot |
| Chat history persistence | Browser storage, React state | Not applicable (GraphBot uses graph for memory) |
| Brick auto-identification | Context injection, implicit selection | Not applicable |
