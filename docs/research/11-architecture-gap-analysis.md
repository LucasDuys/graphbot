# 11 -- Architecture Gap Analysis

**Date:** 2026-03-22
**Scope:** Synthesizes all 10 research files (01 through 10), the existing gap-analysis.md, README.md, and a line-by-line audit of the GraphBot codebase into a comprehensive architecture gap analysis.
**Purpose:** Map what GraphBot has today against what the literature says is needed for full autonomous agent capability, identify the top 10 gaps ranked by impact, and propose a Phase 15+ roadmap.

---

## Part 1: Current Architecture Inventory

This section documents what GraphBot actually implements today, grounded in source code inspection, not aspirational descriptions.

### 1.1 Core Pipeline (core_gb/)

| Component | File | What It Does | What It Does NOT Do |
|-----------|------|-------------|-------------------|
| **IntakeParser** | `core_gb/intake.py` | Rule-based, zero-cost intent classification. Classifies domain via keyword counting, estimates complexity (1-5) via word count and conjunction heuristics, extracts entities via capitalization, classifies task type (ATOMIC, DATA_PARALLEL, SEQUENTIAL, INTEGRATED). | No semantic understanding. No learned classification. No ambiguity resolution. No multi-turn context. |
| **Decomposer** | `core_gb/decomposer.py` | Single-level LLM decomposition with constrained JSON output. Validates against schema (max depth 3, max 5 children). Supports `depends_on`, `provides/consumes` data contracts. Domain override via keyword matching. json_repair fallback. Retry once on failure. Falls back to single atomic node. | No recursive decomposition (single LLM call produces entire tree). No dynamic re-decomposition during execution. No quality scoring of decomposition. No alternative decomposition generation. Max depth hard-capped at 3. |
| **DAGExecutor** | `core_gb/dag_executor.py` | Parallel streaming topological dispatch via `graphlib.TopologicalSorter`. Semaphore-bounded concurrency (max 10). Data forwarding from completed nodes to dependents via `provides/consumes`. Tool-first execution with LLM fallback. Single retry on tool failure. Deterministic aggregation. | No dynamic DAG modification during execution. No backtracking. No checkpoint/rollback. No intermediate quality checks. No node-level retry with alternative strategies. No execution budget enforcement. |
| **Orchestrator** | `core_gb/orchestrator.py` | Wires intake, pattern cache, decomposer, and executor into a single flow. Pattern cache check before decomposition. Simple/integrated path bypasses decomposition. Entity resolution for context assembly. Graph update after execution. | No re-planning loop. No quality assessment of final output. No failure reflection. No learning from failed executions. No task queuing or persistence. |
| **PatternMatcher** | `core_gb/patterns.py` | Regex-based trigger matching with Levenshtein fallback. Variable slot extraction via capture groups. Threshold-based scoring (default 0.7). | No embedding-based similarity. No context-aware matching. No negative patterns (anti-patterns). No success-rate weighting. No temporal decay of patterns. |
| **PatternExtractor** | `core_gb/patterns.py` | Extracts reusable templates from successful multi-node executions. Generalizes by finding varying words across leaf descriptions. Stores as JSON node templates in knowledge graph. | Only extracts from successes (ignores failures). No cross-execution pattern discovery. No pattern quality scoring. No pattern evolution or refinement. |
| **GraphUpdater** | `core_gb/updater.py` | Records Task node, ExecutionTree node, and DERIVED_FROM edge after each execution. Triggers pattern extraction. | No failure analysis storage. No reflection generation. No execution trace detail. No intermediate result recording. |

### 1.2 Knowledge Graph (graph/)

| Component | File | What It Does | What It Does NOT Do |
|-----------|------|-------------|-------------------|
| **GraphStore** | `graph/store.py` | Kuzu embedded graph database. Full CRUD on typed node/edge tables. 2-hop context assembly with token budget. Memory temporal filtering (valid_until). | No Personalized PageRank. No community detection. No graph-native retrieval. No activation decay. No memory consolidation. No forgetting mechanism. No semantic search. |
| **Schema** | `graph/schema.py` | 10 node types (User, Project, File, Service, Contact, PatternNode, Memory, Task, Skill, ExecutionTree). 12 edge types. Static, prescribed ontology. | No schema evolution. No agent-driven relationship creation. No temporal metadata on edges (no valid_from/valid_until on relationships). No provenance tracking beyond source_episode on Memory. |
| **EntityResolver** | `graph/resolver.py` | 3-layer resolution: exact match (confidence 1.0), Levenshtein ratio (threshold 0.8), BM25-style keyword match. Loads all candidates into memory. | No embedding-based resolution. No entity linking to external KBs. No coreference resolution. No disambiguation beyond string similarity. |

### 1.3 Model Layer (models/)

| Component | File | What It Does | What It Does NOT Do |
|-----------|------|-------------|-------------------|
| **ModelRouter** | `models/router.py` | Complexity-based model selection (5 tiers mapped to models). Multi-provider rotation with circuit breaking and rate limiting. | No per-task model selection (only complexity-based). No learned routing. No cascade (try cheap first, escalate on failure). No confidence-based escalation. No cost budget enforcement. |
| **Providers** | `openrouter.py`, `google.py`, `groq.py` | Three provider backends via LiteLLM. Provider-specific rate limiters and circuit breakers. | No semantic caching. No prompt compression. No token budget directives. No shared KV cache. |

### 1.4 Tools (tools_gb/)

| Component | File | What It Does | What It Does NOT Do |
|-----------|------|-------------|-------------------|
| **ToolRegistry** | `tools_gb/registry.py` | Maps Domain enum to tool instances. Structured tool_method dispatch. Fallback to description-based parsing. Code agent for file edits. | No dynamic tool creation. No tool composition. No tool quality tracking. No browser automation. No computer use. No MCP integration in the core engine. |
| **FileTool** | `tools_gb/file.py` | Read, list_dir, search operations on local filesystem. | No write (explicitly rejected). No edit without code agent. |
| **WebTool** | `tools_gb/web.py` | DuckDuckGo search, URL fetch, search-and-summarize. | No browser automation. No interactive web navigation. No form filling. No screenshot-based understanding. |
| **ShellTool** | `tools_gb/shell.py` | Subprocess command execution with timeout. | No sandboxing. No permission model. No output size limits. |

### 1.5 Channels and UI

| Component | Location | What It Does |
|-----------|----------|-------------|
| **Nanobot** | `nanobot/` | Agent framework fork providing agent loop, tool system, MCP integration, channel support, session management, cron, heartbeat. |
| **Telegram Bridge** | `nanobot/channels/graphbot_telegram.py` | Routes Telegram messages to Orchestrator. |
| **WhatsApp Bridge** | `nanobot/channels/graphbot_whatsapp.py` | Routes WhatsApp messages to Orchestrator. |
| **React Frontend** | `ui/frontend/` | Next.js dashboard with React Flow DAG visualization, D3 knowledge graph panel, SSE streaming, dark mode, task input. |

---

## Part 2: Research-to-Component Mapping

Each of the 10 research areas maps to specific GraphBot components. This table identifies where the literature's recommendations apply and whether GraphBot currently implements them.

| Research Area | Key Papers | Maps To | Current Status |
|--------------|-----------|---------|---------------|
| **01: Planning & Decomposition** | GoT, ADaPT, ReAcTree, DynTaskMAS | `decomposer.py`, `dag_executor.py` | Single-level decomposition only. No recursive expansion, no dynamic re-planning, no control flow nodes. DAG executor is static -- no mid-execution modification. |
| **02: Tool Use & Function Calling** | LATM, CREATOR, Voyager skills, ToolChain* | `tools_gb/registry.py`, `tools_gb/*.py` | Fixed tool registry (file, web, shell). No dynamic tool creation, no tool composition, no A* search for tool selection, no tool quality tracking. |
| **03: Memory & Knowledge** | Graphiti, HippoRAG, GraphRAG, Mem0, ACT-R | `graph/store.py`, `graph/resolver.py`, `graph/updater.py` | Basic 2-hop traversal with token budget. No PPR retrieval, no community detection, no activation decay, no memory consolidation, no forgetting, no graph-native search. |
| **04: Self-Correction & Verification** | CRITIC, CoVe, Self-Refine, "Cannot Self-Correct" | `dag_executor.py` (retry logic) | Single retry on tool failure with LLM fallback. No systematic verification. No self-consistency checks. No CRITIC-style tool-grounded correction. No output quality assessment. |
| **05: Multi-Agent Systems** | MacNet, GPTSwarm, AFlow, MAST failures | `dag_executor.py`, `orchestrator.py` | DAG-based parallel execution validates architecture. No multi-agent debate, no topology optimization, no agent importance scoring, no failure mode detection per MAST taxonomy. |
| **06: Browser & Computer Use** | SeeAct, Browser-Use, Claude CU, AWM | `tools_gb/web.py` | Web search and URL fetch only. No browser automation, no screenshot-based navigation, no computer use, no interactive web tasks. |
| **07: Long-Horizon Execution** | Voyager, DEPS, Reflexion, ExpeL, HiAgent | `patterns.py`, `orchestrator.py` | Pattern cache stores successful templates. No failure reflection, no negative pattern caching, no success-rate tracking, no hierarchical context management, no dynamic re-planning from intermediate results. |
| **08: Real-World Frameworks** | OpenHands, LangGraph, DSPy, SWE-Agent | Full architecture | DAG architecture is a genuine differentiator vs. serial ReAct loops. No event sourcing, no checkpointing, no declarative agent definitions, no prompt optimization. |
| **09: Cost Optimization** | FrugalGPT, RouteLLM, LLMLingua, GPTCache | `models/router.py`, `patterns.py` | Multi-provider rotation and pattern cache exist. No per-leaf model routing, no learned cascade, no prompt compression, no semantic caching, no token budget directives. |
| **10: Safety & Alignment** | ToolEmu, OWASP Top 10, Constitutional AI, MAST | `tools_gb/shell.py`, `orchestrator.py` | Shell tool has timeout but no sandboxing. No DAG-level intent classification. No composition-aware safety checks. No instruction priority hierarchy. No autonomy levels. No output sanitization between DAG levels. |

---

## Part 3: Top 10 Architectural Gaps Ranked by Impact

Ranked by the combination of: (a) how much autonomous capability the gap blocks, (b) how many research papers converge on the same recommendation, and (c) practical impact on daily use.

### Gap 1: No Self-Correction or Verification Pipeline

**Description:** GraphBot executes DAG nodes and accepts whatever output they produce. The only "correction" is a single retry on tool failure. There is no mechanism to detect incorrect reasoning, no quality assessment of outputs, no fact-checking against the knowledge graph, and no iterative refinement.

**Why It Matters:** Research paper 04 ("Cannot Self-Correct") proves that self-correction without external feedback is unreliable, but correction WITH external feedback (CRITIC, CoVe) is consistently effective. GraphBot has the ideal external feedback source -- its own knowledge graph -- but never uses it for verification. Every DAG execution is fire-and-forget.

**Informing Papers:**
- CRITIC (ICLR 2024): Tool-interactive critiquing with external verification
- CoVe (ACL 2024): Chain-of-verification with decoupled fact-checking
- Self-Consistency (ICLR 2023): Sample multiple outputs, take majority
- "Let's Verify Step by Step" (OpenAI 2023): Process supervision beats outcome supervision
- "Cannot Self-Correct" (ICLR 2024): External feedback is mandatory for reasoning tasks

**Estimated Complexity:** MEDIUM. Layer 1 (format/type checking) is trivial. Layer 2 (self-consistency) requires 3x LLM cost per critical node. Layer 3 (KG verification) requires a query generation step.

**Recommended Approach:** Implement a layered verification architecture:
1. Every node: format and type validation (rule-based, zero cost)
2. Important nodes: 3-way self-consistency with confidence-weighted selection (CISC)
3. Critical nodes: CRITIC-style verification against knowledge graph
4. Configurable per-node verification level via task metadata

**Classification:** New feature on existing architecture. The DAG executor already has per-node hooks; verification is an interceptor between node execution and result propagation.

---

### Gap 2: No Recursive or Dynamic Decomposition

**Description:** The Decomposer makes a single LLM call that produces the entire task tree. If that decomposition is wrong, there is no recovery. If a leaf node is too complex for a small model, there is no way to further decompose it. The DAG is static once created -- no modification during execution.

**Why It Matters:** ADaPT (NAACL 2024) shows that adaptive decomposition -- trying direct execution first, decomposing only on failure -- outperforms fixed-depth decomposition. ReAcTree (2025) shows dynamic agent tree construction nearly doubles success rates. The existing gap-analysis.md already identified that "decomposition quality sometimes HURTS output" when isolated nodes lose context. Recursive decomposition solves this by decomposing only when needed.

**Informing Papers:**
- ADaPT (NAACL 2024): As-needed decomposition on failure
- ReAcTree (2025): Dynamic agent tree with control flow nodes
- SELFGOAL (NAACL 2025): GoalTree grows during execution
- GoalAct (NCIIP 2025): Continuously updated global plan
- Beyond Entangled Planning (2026): DAG-based sub-goal decoupling validates GraphBot's architecture

**Estimated Complexity:** LARGE. Requires modifying the DAGExecutor to support mid-execution graph mutation, adding a re-decomposition trigger, and implementing "lazy node expansion" where leaf nodes can be replaced with sub-DAGs.

**Recommended Approach:**
1. Add a `max_retries` field to TaskNode. On leaf failure, attempt re-decomposition of that node into a sub-DAG.
2. Implement "lazy expansion": nodes can be marked as `expandable`, and the executor decomposes them only when reached.
3. Add control flow nodes (conditional, loop) as first-class constructs.
4. Feed intermediate results back to the orchestrator for re-planning of unexecuted sub-DAGs.

**Classification:** Architecture change. The DAGExecutor's static graph assumption must be relaxed to support mutation.

---

### Gap 3: No Failure Reflection or Learning from Failures

**Description:** GraphUpdater records Task nodes and ExecutionTrees but only extracts patterns from successful executions. Failed executions are logged as status="failed" with no analysis of why they failed. The pattern cache contains only positive examples.

**Why It Matters:** Reflexion (NeurIPS 2023), ExpeL (AAAI 2024), and Trial-and-Error (ACL 2024) all demonstrate that learning from failures is as important as learning from successes. Reflexion achieves 91% on HumanEval (vs. GPT-4's 80%) purely through verbal reflection on failures stored in episodic memory. ExpeL shows consistent improvement from accumulated failure insights. GraphBot's pattern cache is its "most strategic asset" (per research 07) but it only stores half the picture.

**Informing Papers:**
- Reflexion (NeurIPS 2023): Verbal reinforcement from failure reflections
- ExpeL (AAAI 2024): Cross-task knowledge from comparing successes and failures
- Trial and Error / ETO (ACL 2024): Contrastive learning from failure trajectories
- SASR (ICLR 2025): Success-rate-based reward from historical experience
- RAGEN (2025): Echo Trap warning -- pattern cache bias toward common patterns

**Estimated Complexity:** SMALL. The infrastructure exists (GraphUpdater, PatternStore, knowledge graph). The missing piece is a post-execution reflection step that generates verbal analysis of failures and stores them alongside execution records.

**Recommended Approach:**
1. After failed DAG execution, prompt the LLM to generate a verbal reflection: "What went wrong? Why? What should be done differently?"
2. Store reflections as Memory nodes linked to the failed Task via a new REFLECTION_OF edge.
3. During future decomposition, retrieve relevant failure reflections from the knowledge graph and include them in the decomposer's context.
4. Add success/failure counters to PatternNode. Track success_rate = success_count / total_count. Prefer higher-success-rate patterns during matching.
5. Store "negative patterns" -- decompositions that failed -- as anti-patterns with negative trigger templates.

**Classification:** New feature on existing architecture. Uses existing graph infrastructure; adds a new node type, edge type, and post-execution step.

---

### Gap 4: No Browser or Computer Use Integration

**Description:** GraphBot's web capability is limited to DuckDuckGo search and URL fetch. There is no browser automation, no interactive web navigation, no form filling, no screenshot-based understanding, and no desktop computer use.

**Why It Matters:** Research 06 shows browser/computer use is a high-value capability frontier. Claude Computer Use reached 72.5% on OSWorld (approaching human-level 72.4%). Agent Workflow Memory (AWM) shows 51.1% improvement on WebArena by reusing learned navigation patterns -- exactly what GraphBot's temporal knowledge graph could provide. Browser-Use is a mature, pip-installable library that provides dual-mode perception (DOM + screenshots). Enterprise web automation (WorkArena) represents a differentiated market opportunity.

**Informing Papers:**
- SeeAct (ICML 2024): Planner-Grounder separation for web navigation
- Browser-Use (2024): Open-source web automation with dual-mode perception
- Claude Computer Use (Anthropic): 72.5% on OSWorld
- AWM (2024): 51.1% improvement from reusing learned web workflows
- ST-WebAgentBench (2024): Safety-first web agent evaluation

**Estimated Complexity:** MEDIUM. Browser-Use provides the automation layer via `pip install browser-use`. Integration requires: (a) new BrowserTool in tools_gb/, (b) new Domain.BROWSER or extending Domain.WEB, (c) planning-grounding separation in DAG patterns, (d) caching navigation patterns in the knowledge graph.

**Recommended Approach:**
1. Integrate Browser-Use as a new tool backend (it wraps Playwright with LLM-driven navigation).
2. Implement as a two-node DAG pattern: Planner node (LMM decides what to do) and Executor node (Browser-Use executes the action).
3. Store successful navigation sequences in the knowledge graph as reusable workflow templates (AWM pattern).
4. Implement policy guards per ST-WebAgentBench: consent checking, boundary enforcement, action logging.
5. Start with read-only web tasks (search, extract data). Add interactive tasks (form filling, clicking) as a second phase.

**Classification:** New feature on existing architecture. The DAG executor handles the multi-step orchestration; browser use adds a new tool type.

---

### Gap 5: No Security or Safety Architecture

**Description:** GraphBot has no safety checks at any level. The shell tool executes arbitrary commands with no sandboxing. The decomposer has no guardrails against harmful task decomposition. There is no DAG-level intent classification, no output sanitization between nodes, no instruction priority hierarchy, and no autonomy level configuration.

**Why It Matters:** Research 10 identifies that GraphBot's DAG decomposition creates a novel safety surface: individually benign subtasks that compose into harmful outcomes (a "composition attack"). OWASP Top 10 for LLMs ranks Excessive Agency as #6 and Prompt Injection as #1 -- both directly applicable. AgentHarm (ICLR 2025) shows leading LLMs are "surprisingly compliant" with malicious requests, and jailbroken agents maintain coherent multi-step harmful behavior. The MAST taxonomy (NeurIPS 2025) documents 41-87% failure rates across 7 SOTA multi-agent frameworks.

**Informing Papers:**
- "Fully Autonomous AI Agents Should Not be Developed" (2025): Composition risk
- ToolEmu (ICLR 2024): LM-emulated sandbox for pre-flight safety
- OWASP Top 10 for LLMs (2025): Excessive Agency, Prompt Injection
- AgentHarm (ICLR 2025): Agents comply with malicious multi-step plans
- AutoDefense (2024): Multi-agent defense reduces attack success from 55.7% to 7.9%

**Estimated Complexity:** MEDIUM for Tier 1, LARGE for full implementation. Tier 1 (must-have) is achievable with rule-based checks and configurable limits.

**Recommended Approach:** Implement in three tiers:
- **Tier 1 (pre-deployment):** DAG-level intent classifier (analyze full decomposition before execution). Recursion depth and fan-out hard limits (already partially present: max_depth=3, max_children=5). Shell sandboxing via subprocess restrictions. Output sanitization between DAG levels. Instruction priority: system > user > decomposition > leaf > external data.
- **Tier 2 (production):** Configurable autonomy levels (5-level framework). Per-node risk scoring. NeMo Guardrails or Llama Guard integration. Execution audit trail.
- **Tier 3 (mature):** Multi-model cross-validation for safety-critical paths. Transactional execution with rollback. Constitutional AI principles in decomposition prompts.

**Classification:** Architecture change (Tier 1 requires interceptors in the orchestrator and executor) + new features (Tiers 2-3).

---

### Gap 6: No Memory Evolution (Consolidation, Forgetting, Abstraction)

**Description:** The knowledge graph grows monotonically. Every execution adds Task nodes, ExecutionTree nodes, and Pattern nodes. Nothing is ever consolidated, abstracted, or forgotten. Over time, context assembly will degrade as the graph grows unbounded: 2-hop traversal returns increasingly irrelevant results, entity resolution loads more candidates into memory, and pattern matching scans more patterns.

**Why It Matters:** The "Memory in the Age of AI Agents" survey (2025) identifies that systems without consolidation degrade over time. ACT-R activation models and Ebbinghaus forgetting curves are well-understood mechanisms for principled memory management. HippoRAG shows that Personalized PageRank (which naturally prioritizes well-connected, frequently-accessed nodes) outperforms flat retrieval by 10-30x. GraphRAG shows community detection enables global reasoning that 2-hop traversal cannot support.

**Informing Papers:**
- "Memory in the Age of AI Agents" (2025): Formation-Evolution-Retrieval lifecycle
- ACT-R activation model (HAI 2024): Frequency + recency-based activation
- MemoryBank / Ebbinghaus (AAAI 2024): Temporal decay with reinforcement
- HippoRAG (NeurIPS 2024): PPR-based graph retrieval
- GraphRAG (Microsoft 2024): Community detection and pre-computed summaries

**Estimated Complexity:** MEDIUM. Start with simple forgetting (activation decay + pruning), then add consolidation.

**Recommended Approach:**
1. Add `activation_score` and `access_timestamps` fields to graph nodes (schema change).
2. Implement ACT-R base-level activation: `activation = ln(sum(t_i^(-d)))` where t_i are times since each access.
3. Run a periodic background process that: (a) updates activation scores, (b) prunes entries below threshold, (c) merges duplicate entities via EntityResolver.
4. Replace 2-hop context assembly with PPR-seeded retrieval for multi-hop queries.
5. Implement community detection for global queries (use Kuzu's graph analytics or export to NetworkX).

**Classification:** Architecture change for schema and retrieval. New feature for the consolidation background process.

---

### Gap 7: No Model Cascading Beyond Simple Rotation

**Description:** ModelRouter selects models based solely on a static complexity-to-model map (5 tiers). All leaf nodes at the same complexity level use the same model. Provider rotation is sequential fallback on failure -- there is no cost-aware routing, no confidence-based escalation, no per-task model selection, and no learned routing.

**Why It Matters:** FrugalGPT (2023) achieves 98% cost reduction by learning per-query which model in a cascade to use. RouteLLM (ICLR 2025) reduces costs by 2x+ with lightweight router classifiers. GraphBot's DAG structure is ideally suited for cascading -- each leaf node is an independent routing decision. The current single-tier-per-complexity approach wastes resources: many complexity-3 tasks could be handled by a complexity-1 model with the right context.

**Informing Papers:**
- FrugalGPT (2023): Per-query model cascade with 98% cost reduction
- RouteLLM (ICLR 2025): Lightweight router models trained on preference data
- TALE (ACL 2025): Token-budget-aware reasoning reduces output tokens by 67%
- MixTuRe (ICLR 2024): Consistency-based confidence for cascade decisions
- C3PO (NeurIPS 2025): Probabilistic cost constraints for cascade optimization

**Estimated Complexity:** SMALL for basic cascading, MEDIUM for learned routing.

**Recommended Approach:**
1. Implement cascade logic in ModelRouter: try cheapest model first, check output confidence (via self-consistency or knowledge graph consistency), escalate to stronger model if confidence is low.
2. Add token budget directives to leaf node prompts: "Answer in under N tokens" based on estimated complexity (TALE pattern).
3. Integrate prompt compression (LLMLingua-2) as a pre-processing step on assembled context.
4. Track per-model success rates per task type in the knowledge graph. Use accumulated data to train a lightweight RouteLLM-style classifier.

**Classification:** New feature on existing architecture. ModelRouter's interface stays the same; internal logic becomes cascade-aware.

---

### Gap 8: No Long-Horizon Persistent Goals

**Description:** GraphBot processes each message as an independent request. There is no concept of persistent goals that span multiple messages, sessions, or days. The orchestrator has no task queue, no goal tracking, and no mechanism to resume interrupted work.

**Why It Matters:** CRADLE (ICML 2025) demonstrates agents operating across diverse software environments via persistent skill accumulation. MemGPT shows that LLM-driven context management enables multi-session tasks. HiAgent shows that hierarchical working memory doubles success rates on long-horizon tasks. Real-world tasks -- "research topic X over the next week", "monitor and fix CI failures as they appear", "gradually refactor module Y" -- require persistent state across invocations.

**Informing Papers:**
- Voyager (TMLR 2024): Lifelong learning with ever-growing skill library
- MemGPT / Letta (2023): Two-tier memory for multi-session persistence
- CRADLE (ICML 2025): General computer control with persistent skill curation
- HiAgent (ACL 2025): Hierarchical working memory for long-horizon tasks
- Inner Monologue (CoRL 2023): Closed-loop feedback for goal-directed execution

**Estimated Complexity:** LARGE. Requires a persistent task queue, goal decomposition across sessions, progress tracking in the knowledge graph, and resumption logic in the orchestrator.

**Recommended Approach:**
1. Add a Goal node type to the graph schema with status tracking (active, paused, completed, abandoned).
2. Implement a goal decomposition step that breaks long-horizon goals into session-sized sub-goals.
3. Store sub-goal progress in the knowledge graph. On each new message, check for active goals and route to the appropriate sub-goal.
4. Implement goal-level context assembly: when resuming, retrieve the goal's history and current state from the graph.
5. Add a cron-triggered "goal check" that evaluates progress on active goals and triggers next steps.

**Classification:** Architecture change. The Orchestrator currently processes single messages; it must become goal-aware.

---

### Gap 9: No Dynamic Tool Creation

**Description:** GraphBot's tool registry is static: file, web, and shell tools are hardcoded. When a task requires a capability that does not exist (e.g., "parse this CSV and compute statistics", "generate a chart"), the system falls back to LLM reasoning without tool support.

**Why It Matters:** LATM (ICLR 2024) demonstrates that LLMs can create reusable Python tools that amortize expensive generation across many uses. Voyager's skill library shows that an ever-growing tool set creates compound improvement. CREATOR (EMNLP 2023) shows that separating tool creation from tool use improves both. GraphBot's pattern cache already stores reusable execution templates; extending it to store reusable code tools is a natural evolution.

**Informing Papers:**
- LATM (ICLR 2024): LLMs as Tool Makers -- tool creation + lightweight tool use
- CREATOR (EMNLP 2023): Create-Decide-Execute-Reflect cycle for tool creation
- Voyager (TMLR 2024): Compositional skill library as persistent code
- TOOLMAKER (ACL 2025): Converting research artifacts into agent tools
- CodeAct (ICML 2024): Executable code as unified action space

**Estimated Complexity:** MEDIUM. Requires code generation, sandboxed execution, validation, and storage in the knowledge graph.

**Recommended Approach:**
1. When a leaf node fails because no tool exists for the required capability, trigger a "tool creation" sub-DAG: generate Python function, test it, store it.
2. Store created tools as Skill nodes in the knowledge graph with: function code, docstring, input/output schema, test cases, success rate.
3. During decomposition, include available custom tools in the decomposer's context so it can route to them.
4. Validate created tools by executing test cases in a sandbox before registering them.
5. Implement tool versioning: if a tool fails in production, create a new version rather than modifying the existing one.

**Classification:** New feature on existing architecture. The tool registry gains a dynamic registration mechanism; the knowledge graph stores tool definitions.

---

### Gap 10: No Semantic Caching

**Description:** GraphBot's pattern cache uses regex + Levenshtein matching on trigger templates. Semantically similar but syntactically different queries ("What is the weather in Amsterdam?" vs. "Amsterdam weather forecast") may not match, causing redundant decomposition and execution.

**Why It Matters:** GPTCache demonstrates 2-10x response speed improvement on cache hits with 60-70% hit rates. SCALM achieves 77% token savings. GraphBot's existing pattern cache achieves 30%+ token reduction; embedding-based similarity could push this to 60-70%. At GraphBot's scale (thousands of tasks), the cumulative savings are substantial.

**Informing Papers:**
- GPTCache (EMNLP 2023): Sentence-transformer embeddings for semantic matching
- MeanCache (IEEE IPDPS 2025): Context-chain aware cache keys
- SCALM (IEEE IWQoS 2024): Clustering-based cache with 77% token savings
- GPT Semantic Cache (2024): Redis-based in-memory semantic embeddings

**Estimated Complexity:** SMALL. Replace Levenshtein matching with embedding similarity. Use a lightweight embedding model (e.g., `all-MiniLM-L6-v2`) for pattern trigger encoding.

**Recommended Approach:**
1. At pattern save time, compute and store a sentence embedding of the trigger template.
2. At match time, compute embedding of incoming query and find nearest neighbors by cosine similarity.
3. Include context in the cache key (MeanCache pattern): same query with different graph context should not match.
4. Set similarity threshold empirically (start at 0.85, tune based on false positive rate).
5. Consider Redis-backed embedding index for O(1) lookup at scale.

**Classification:** New feature on existing architecture. PatternMatcher gains an embedding-based matching path alongside the existing regex/Levenshtein path.

---

## Part 4: Gap Classification Summary

| Gap | Type | Complexity | Impact |
|-----|------|-----------|--------|
| 1. No verification pipeline | New feature on existing architecture | MEDIUM | CRITICAL -- every output is unverified |
| 2. No recursive decomposition | Architecture change | LARGE | HIGH -- limits task complexity |
| 3. No failure reflection | New feature on existing architecture | SMALL | HIGH -- wastes half the learning signal |
| 4. No browser/computer use | New feature on existing architecture | MEDIUM | HIGH -- blocks entire task categories |
| 5. No safety architecture | Architecture change + new features | MEDIUM-LARGE | CRITICAL -- blocks deployment |
| 6. No memory evolution | Architecture change + new features | MEDIUM | MEDIUM -- degrades over time |
| 7. No model cascading | New feature on existing architecture | SMALL-MEDIUM | MEDIUM -- wastes cost |
| 8. No persistent goals | Architecture change | LARGE | MEDIUM -- blocks long-horizon tasks |
| 9. No dynamic tool creation | New feature on existing architecture | MEDIUM | MEDIUM -- limits adaptability |
| 10. No semantic caching | New feature on existing architecture | SMALL | MEDIUM -- suboptimal cache hit rate |

---

## Part 5: Can the Current DAG + Graph Design Support Full Autonomy?

**Short answer: The design is sound. The implementation needs significant extension, but NOT structural replacement.**

The evidence for this conclusion:

1. **Independent academic validation.** "Beyond Entangled Planning" (2026) formalizes exactly GraphBot's architecture (DAG-based sub-goal decomposition with isolated contexts) and shows 82% token reduction. DynTaskMAS (ICAPS 2025) reports near-linear scaling with the same architecture. MacNet (ICLR 2025) validates that DAGs outperform regular topologies for multi-agent coordination.

2. **The DAG is the right execution abstraction.** GoT (AAAI 2024) proves DAGs are strictly more expressive than trees (which are strictly more expressive than chains). Every major planning paradigm (ToT, GoT, ReAcTree, TDAG, DynTaskMAS) converges on DAG-like structures. GraphBot already has this.

3. **The knowledge graph is the right memory abstraction.** Graphiti, HippoRAG, GraphRAG, Think-on-Graph, and Mem0 all validate graph-structured memory over flat vector stores. GraphBot's Kuzu-based temporal knowledge graph is architecturally correct. It needs better retrieval (PPR, community detection) and evolution (consolidation, forgetting), but the foundation is right.

4. **The gaps are extensions, not replacements.** All 10 gaps can be addressed by adding to the current architecture:
   - Verification = interceptors between DAG nodes
   - Recursive decomposition = mutation support in DAGExecutor
   - Failure reflection = new post-execution step in GraphUpdater
   - Browser use = new tool in ToolRegistry
   - Safety = middleware layers in Orchestrator and Executor
   - Memory evolution = background process on existing graph
   - Model cascading = enhanced logic in ModelRouter
   - Persistent goals = new node type and orchestration layer
   - Tool creation = dynamic registration in ToolRegistry
   - Semantic caching = enhanced matching in PatternMatcher

5. **What WOULD require structural change.** Two specific scenarios would strain the current design:
   - **True multi-agent systems** (agents with persistent identity, state, and inter-agent communication) would require an agent abstraction layer above the DAG executor. Currently, DAG nodes are stateless function calls. Multi-agent coordination (per MacNet, GPTSwarm) requires nodes to maintain state across DAG executions.
   - **Real-time streaming execution** (continuous processing of incoming data, not request-response) would require replacing the current batch-oriented DAG executor with an event-driven architecture. The Nanobot event bus provides some of this, but the core engine is not event-driven.

**Bottom line:** GraphBot's DAG + graph architecture can support full autonomous agent capability through extension. The 10 gaps identified above are the specific extensions needed. No core architectural component needs to be replaced -- but several need to be significantly enhanced, particularly the DAGExecutor (mutation support), GraphStore (retrieval algorithms), and Orchestrator (goal awareness, safety middleware).

---

## Part 6: Phase 15+ Roadmap

Based on gap ranking, dependency ordering, and estimated complexity, here is the recommended build order.

### Phase 15: Foundation for Autonomy (Gaps 3, 5-Tier1, 10)

**Rationale:** These are the smallest-effort, highest-ROI changes. They make the system safer, smarter, and more efficient without touching core architecture.

| Task | Gap | Effort | Description |
|------|-----|--------|-------------|
| T15.1 | Gap 3 | 2-3 days | Failure reflection: post-execution LLM call on failures, store as Memory nodes with REFLECTION_OF edges. Retrieve during future decomposition. |
| T15.2 | Gap 3 | 1 day | Success-rate tracking: add success_count/failure_count to PatternNode. Weight pattern matching by success rate. |
| T15.3 | Gap 5 | 2-3 days | Safety Tier 1: DAG-level intent classifier (analyze full plan before execution), recursion hard limits, shell sandboxing (allowlist commands), output sanitization between levels. |
| T15.4 | Gap 10 | 2-3 days | Semantic caching: add embedding-based similarity to PatternMatcher using sentence-transformers. Context-aware cache keys. |
| T15.5 | Gap 7 | 2-3 days | Basic cascade: try cheapest model first per leaf, escalate on low-confidence output. Token budget directives in leaf prompts (TALE pattern). |

**Acceptance:** All tests pass. Failure reflections appear in the knowledge graph. Pattern matching uses embeddings. Safety checks block obviously harmful decompositions. Cascade reduces average cost per task by 30%+.

### Phase 16: Verification and Self-Correction (Gap 1)

**Rationale:** With failure reflection in place, the system now learns from mistakes. Verification prevents mistakes from reaching users in the first place.

| Task | Gap | Effort | Description |
|------|-----|--------|-------------|
| T16.1 | Gap 1 | 3-4 days | Layer 1 verification: format/type checking on all node outputs. Schema validation for structured outputs. |
| T16.2 | Gap 1 | 3-4 days | Layer 2 verification: self-consistency wrapper. Fan-out to 3 parallel executions, confidence-weighted selection (CISC). |
| T16.3 | Gap 1 | 3-4 days | Layer 3 verification: CRITIC-style knowledge graph verification. Generate verification queries, check against graph, revise if inconsistent. |
| T16.4 | Gap 1 | 2-3 days | Configurable verification levels per node. Default: Layer 1 for all, Layer 2 for complexity >= 3, Layer 3 opt-in. |

**Acceptance:** Verified outputs are measurably more accurate than unverified on a test suite. Verification overhead is configurable and bounded.

### Phase 17: Dynamic Execution (Gap 2)

**Rationale:** With verification catching errors, recursive decomposition becomes safe -- failed nodes get re-decomposed and verified rather than silently returning bad results.

| Task | Gap | Effort | Description |
|------|-----|--------|-------------|
| T17.1 | Gap 2 | 4-5 days | Mutable DAG: allow DAGExecutor to replace a failed leaf node with a sub-DAG (re-decomposition). |
| T17.2 | Gap 2 | 3-4 days | Lazy node expansion: nodes marked as `expandable` are decomposed only when reached in topological order. |
| T17.3 | Gap 2 | 3-4 days | Control flow nodes: conditional (if/else based on predecessor output), loop (retry with modified context). |
| T17.4 | Gap 2 | 2-3 days | Intermediate result feedback: after each wave of completed nodes, optionally re-plan remaining unexecuted nodes with accumulated results. |

**Acceptance:** DAG executor can dynamically expand nodes. A task that previously required manual depth-3 decomposition succeeds with depth-1 initial decomposition + lazy expansion. Control flow nodes work in benchmark tasks.

### Phase 18: Browser and Tool Expansion (Gaps 4, 9)

**Rationale:** With dynamic execution and verification in place, new tool types can be integrated safely and their outputs verified.

| Task | Gap | Effort | Description |
|------|-----|--------|-------------|
| T18.1 | Gap 4 | 4-5 days | Browser-Use integration: new BrowserTool in tools_gb/, Planner-Grounder DAG pattern, navigation sequence caching in knowledge graph. |
| T18.2 | Gap 4 | 2-3 days | Policy guards for web actions: consent checking, boundary enforcement, action audit logging (per ST-WebAgentBench). |
| T18.3 | Gap 9 | 4-5 days | Dynamic tool creation: on capability gap, generate Python function, test in sandbox, register in ToolRegistry, store as Skill node. |
| T18.4 | Gap 9 | 2-3 days | Tool quality tracking: success/failure counts per tool, automatic deprecation of tools with <50% success rate. |

**Acceptance:** GraphBot can navigate to a website, extract data, and fill forms via Browser-Use. Dynamically created tools persist across sessions and are reused. Tool success rates are tracked and displayed.

### Phase 19: Memory Intelligence (Gap 6)

**Rationale:** With more data flowing through the system (browser traces, created tools, failure reflections), memory evolution becomes critical to prevent retrieval degradation.

| Task | Gap | Effort | Description |
|------|-----|--------|-------------|
| T19.1 | Gap 6 | 3-4 days | ACT-R activation model: add activation_score to graph nodes, compute on retrieval, boost on access. Background decay process. |
| T19.2 | Gap 6 | 3-4 days | Memory consolidation: periodic merge of duplicate entities, summary generation for clusters of related memories. |
| T19.3 | Gap 6 | 3-4 days | PPR-based retrieval: replace 2-hop traversal with Personalized PageRank for multi-hop context assembly. |
| T19.4 | Gap 6 | 2-3 days | Forgetting: prune entries below activation threshold. Archive (not delete) to cold storage for potential recovery. |

**Acceptance:** Knowledge graph size stabilizes rather than growing unboundedly. Context assembly quality improves on multi-hop queries. Entity resolution performance remains constant as graph grows.

### Phase 20: Persistent Goals and Safety Hardening (Gaps 8, 5-Tier2/3)

**Rationale:** With the core engine mature, add long-horizon capability and production-grade safety.

| Task | Gap | Effort | Description |
|------|-----|--------|-------------|
| T20.1 | Gap 8 | 5-6 days | Goal node type, goal decomposition into session-sized sub-goals, progress tracking, resumption logic. |
| T20.2 | Gap 8 | 3-4 days | Cron-triggered goal evaluation: check progress on active goals, trigger next steps automatically. |
| T20.3 | Gap 5 | 3-4 days | Safety Tier 2: autonomy level configuration, per-node risk scoring, Llama Guard integration. |
| T20.4 | Gap 5 | 3-4 days | Safety Tier 3: multi-model cross-validation, transactional execution with rollback, constitutional principles in decomposition. |

**Acceptance:** GraphBot can accept a multi-day goal, decompose it into daily sub-goals, execute incrementally, and report progress. Safety evaluation suite passes with <5% policy violation rate.

---

## Part 7: Summary

GraphBot's recursive DAG execution engine with temporal knowledge graph is architecturally validated by at least 15 independent research papers published at top venues (NeurIPS, ICLR, ICML, ACL, AAAI). The core design -- decompose complex tasks into simple subtasks, execute in parallel on cheap models with rich graph context, learn from execution history -- is the direction the field has converged on.

The 10 gaps identified in this analysis are the specific extensions needed to move from a research prototype to a capable autonomous agent. They are ordered by impact and dependency:

1. **Learn from failures** (Gap 3) -- trivially implementable, compounds everything else
2. **Verify outputs** (Gap 1) -- prevents errors from propagating
3. **Decompose dynamically** (Gap 2) -- handles tasks the static decomposer cannot
4. **Add safety** (Gap 5) -- required for any deployment
5. **Add browser use** (Gap 4) -- unlocks new task categories
6. **Evolve memory** (Gap 6) -- prevents long-term degradation
7. **Route models smartly** (Gap 7) -- reduces cost further
8. **Persist goals** (Gap 8) -- enables long-horizon tasks
9. **Create tools dynamically** (Gap 9) -- enables open-ended capability
10. **Cache semantically** (Gap 10) -- improves cache hit rates

None of these require replacing the core architecture. All build on the DAG executor, knowledge graph, and pattern cache that already exist. The foundation is solid; the superstructure needs to be built.
