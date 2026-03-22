# GraphBot Research Index

**Compiled:** 2026-03-22
**Total entries across all files:** 250 (includes papers, frameworks, benchmarks, and surveys)
**Estimated unique papers/systems:** 190+ (some appear in multiple files for cross-referencing)

This directory contains 10 research survey files covering the academic and industry landscape relevant to GraphBot's architecture. Each file was researched on 2026-03-22 and covers papers from 2022--2026.

---

## File Index

### [01 - Planning and Decomposition](01-planning-and-decomposition.md)

**Entries:** 24 | **Sections:** Foundational Reasoning, Task Decomposition, Hierarchical/Multi-Agent Planning, Plan Refinement, External Formalisms, DAG-Specific Frameworks, Embodied Planning, Surveys

Covers Tree of Thoughts, Graph of Thoughts, Least-to-Most and Decomposed Prompting, ADaPT (as-needed decomposition), ReAcTree, CoAct, SELFGOAL, ReWOO, Reflexion, ISR-LLM, LLM+P, DynTaskMAS, and planning surveys. Validates that DAG-based execution is well-supported by literature, that separation of planning and execution is consensus best practice, and that recursive/dynamic decomposition outperforms static approaches. DynTaskMAS achieves 21-33% execution time reduction and 88% resource utilization, setting concrete targets for GraphBot.

**Most important finding for GraphBot:** ADaPT's "as-needed decomposition" -- attempt execution first, decompose only on failure -- is the key mechanism for GraphBot's recursive DAG expansion and directly prevents over-decomposition of simple tasks.

---

### [02 - Tool Use and Function Calling](02-tool-use-and-function-calling.md)

**Entries:** 24 | **Sections:** Foundational Tool-Augmented LLMs, Tool Orchestration/Composition, Function Calling/API Integration, Dynamic Tool Creation, Frameworks/Benchmarks, Tool Safety, Surveys

Covers Toolformer, Gorilla, ToolLLM/ToolBench (16K+ APIs), HuggingGPT, Chameleon, ToolChain* (A* search), ToolTree (MCTS), ToolACE, APIGen (three-stage verification), NexusRaven, RestGPT, LATM (tool maker/user split), CREATOR, WorkflowLLM (106K workflow samples), TaskWeaver, ToolSword (safety), and ToolkenGPT. Demonstrates that retrieval-augmented tool selection scales to massive registries, small models (7-8B) match GPT-4 on function calling with good data, and three-stage verification (format, execution, semantic) is non-negotiable for reliable tool use.

**Most important finding for GraphBot:** LATM's tool-maker/tool-user split -- use a powerful model to create reusable tools, then execute with cheap models -- directly validates GraphBot's cost architecture of expensive planning with cheap leaf execution.

---

### [03 - Memory and Knowledge Systems](03-memory-and-knowledge.md)

**Entries:** 22 | **Sections:** Foundational Memory Architectures, RAG and Retrieval, Knowledge Graphs, Reflexion/Experience-Based Learning, Memory Consolidation/Forgetting, Surveys

Covers Generative Agents, MemGPT, CoALA cognitive architecture, A-MEM (Zettelkasten-inspired), Self-RAG, GraphRAG, HippoRAG (Personalized PageRank), GNN-RAG, Think-on-Graph, Graphiti/Zep (temporal KG), Reflexion, ExpeL, Voyager skill library, LATS (MCTS + reflection), MemoryBank (Ebbinghaus forgetting), ACT-R memory activation, R3Mem, and Mem0. Validates GraphBot's temporal knowledge graph as the correct memory architecture, with graph-native retrieval (PPR, GNN) consistently outperforming vector similarity. Memory evolution (consolidation + forgetting) is identified as critical to prevent retrieval degradation over time.

**Most important finding for GraphBot:** Graphiti/Zep is the single most relevant system -- a temporal knowledge graph engine that validates GraphBot's core design with production-grade bi-temporal modeling, incremental updates, and hybrid retrieval combining graph traversal and semantic search.

---

### [04 - Self-Correction and Verification](04-self-correction-and-verification.md)

**Entries:** 25 | **Sections:** Self-Debugging/Self-Refinement, CRITIC/Chain-of-Verification, LLM-as-Judge, Limits of Self-Correction, Constitutional AI, Verification/Validation, Reflexion/Introspection

Covers Self-Refine, Self-Debugging (rubber duck debugging), CRITIC (tool-interactive critique), Chain-of-Verification, Self-Consistency, Tree of Thoughts, LLM-as-Judge surveys, multiagent debate for verification, SCoRe (RL-trained self-correction), Constitutional AI, Let's Verify Step by Step (process supervision), Math-Shepherd, Clover (formal verification), CISC (confidence-weighted consistency), and RISE. The central finding is that self-correction without external feedback is unreliable for reasoning, but self-correction with external feedback (tools, knowledge graph, code execution) is consistently effective. Proposes a five-layer verification architecture from format checking to human-in-the-loop.

**Most important finding for GraphBot:** "LLMs Cannot Self-Correct Reasoning Yet" (Huang et al., ICLR 2024) -- every correction loop in GraphBot must include an external feedback source; the knowledge graph is the decisive advantage that makes correction effective where intrinsic self-correction fails.

---

### [05 - Multi-Agent Systems](05-multi-agent-systems.md)

**Entries:** 23 | **Sections:** Foundational Frameworks, Agent Debate, Agent Societies/Coordination, Role-Based/Cooperative Systems, Failure Analysis/Scaling, Communication/Protocols

Covers AutoGen, MetaGPT, ChatDev, CAMEL, CrewAI, multi-agent debate (MAD) and its critique, MacNet (1000+ agent DAGs), GPTSwarm (agents as optimizable graphs), ProAgent, EvoAgent, DyLAN, AFlow (MCTS workflow optimization), failure taxonomy (MAST, 41-87% failure rates), Magentic-One, AgentVerse, MindSearch, and AgentsNet. MacNet and GPTSwarm independently validate that DAGs are the correct formalism for multi-agent systems. The MAST failure taxonomy reveals 41-87% failure rates across 7 SOTA frameworks, establishing that failure detection and mitigation must be built in from day one.

**Most important finding for GraphBot:** MacNet (ICLR 2025) discovers a "collaborative scaling law" showing that multi-agent performance follows logistic growth, with irregular DAG topologies outperforming regular ones -- this validates GraphBot's dynamic topology generation and identifies the point of diminishing returns for adding agents.

---

### [06 - Browser and Computer Use](06-browser-and-computer-use.md)

**Entries:** 25 | **Sections:** Core Benchmarks, Autonomous Web Agents, Computer Use/GUI Automation, Visual Grounding, Commercial Systems, Emerging Approaches/Safety

Covers Mind2Web, WebArena, VisualWebArena, AssistantBench, WorkArena/BrowserGym, ST-WebAgentBench (safety), SeeAct, WebVoyager, Agent Q (340% improvement via self-play), Agent Workflow Memory (51% improvement from reusing workflows), OSWorld, OpenAI CUA/Operator, Claude Computer Use (15% to 72% on OSWorld), Google Mariner, CogAgent, UFO, UGround, BrowseComp, and Browser-Use. SeeAct's planning/grounding separation is identified as the key architectural pattern, with the grounding gap (not planning) as the primary bottleneck. Agent Q demonstrates dramatic improvement through MCTS + self-play, and Agent Workflow Memory validates that GraphBot's temporal knowledge graph can drive 24-51% improvements by reusing learned web navigation patterns.

**Most important finding for GraphBot:** Agent Workflow Memory shows 24.6% improvement on Mind2Web and 51.1% on WebArena by inducing and reusing learned workflows from past executions -- this is exactly what GraphBot's temporal knowledge graph enables, making it a core differentiator for web navigation.

---

### [07 - Long-Horizon Execution](07-long-horizon-execution.md)

**Entries:** 24 | **Sections:** Persistent Agents, Goal Decomposition Over Hours/Days, Self-Improving Agents/Reward Shaping, Continuous Learning, Agent Persistence/State Management

Covers Voyager, DEPS, GITM (100% Minecraft tech tree completion), JARVIS-1, CRADLE, Plan-and-Act, Task-Decoupled Planning (82% token reduction), HiAgent, TMS, Reflexion, ExpeL, ADAS (automated agent design), Agent Q, Self-Rewarding models, SAGE, RAGEN, STaR, Trial and Error (ETO), MemGPT/Letta, SPRING, Inner Monologue, and metacognitive learning. Multiple independent groups converge on GraphBot's architecture (hierarchical DAG decomposition, isolated context per sub-task, memory-augmented planning). The pattern cache is validated as the most strategic asset by Voyager's skill library, JARVIS-1's memory, and SAGE's skill-integrated rewards.

**Most important finding for GraphBot:** Task-Decoupled Planning (TDP, 2026) independently validates GraphBot's architecture -- DAG-based sub-goal decomposition with isolated contexts achieves 82% token reduction while improving robustness, providing direct academic support for GraphBot's core design.

---

### [08 - Real-World Agent Frameworks](08-real-world-agent-frameworks.md)

**Entries:** 28 | **Sections:** Autonomous Coding Agents, Early Autonomous Agents (2023 Wave), Orchestration Frameworks, Production Coding Tools, Comparison Matrices

Covers OpenHands/OpenDevin, SWE-Agent (Agent-Computer Interfaces), Agentless, Devin, Manus AI ($2-3B Meta acquisition), CodeAct, AutoGPT, BabyAGI, AgentGPT, LangGraph, DSPy, Microsoft Agent Framework (Semantic Kernel + AutoGen), CrewAI, Smolagents, Claude Code, Cursor ($500M ARR), Aider, and GitHub Copilot Coding Agent. Includes detailed architecture and cost comparison matrices. Every successful framework separates planning from execution. Code-as-action outperforms JSON-as-action by 20-30%. GraphBot is unique in combining automatic DAG decomposition, temporal knowledge graph, and free model optimization -- no other framework has all three.

**Most important finding for GraphBot:** Across all 28 systems surveyed, GraphBot is the only framework that combines automatic DAG decomposition, temporal knowledge graph memory, and free-model optimization -- confirming its architecturally distinct position in the landscape.

---

### [09 - Cost Optimization and Model Routing](09-cost-optimization-and-model-routing.md)

**Entries:** 26 | **Sections:** FrugalGPT/Cost-Efficient Strategies, Model Cascading/Routing, Speculative Decoding, Semantic Caching/Prompt Compression, KV Cache Optimization, Model Selection/Multi-Model Orchestration, Free-Tier Maximization

Covers FrugalGPT (98% cost reduction), TALE (67% output token reduction), RouteLLM (2x+ cost reduction), cascade routing theory, MixTuRe, C3PO (probabilistic cost constraints), CASCADIA, Mixture-of-Agents, speculative decoding (Leviathan et al., Medusa, EAGLE), GPTCache, LLMLingua/LLMLingua-2 (up to 20x prompt compression), vLLM/PagedAttention, CacheGen, Gorilla, Distilling Step-by-Step, and free-tier provider analysis. Proposes a compounding cost reduction stack that reduces effective cost to 2-8% of a naive single-model baseline. Validates that GraphBot's core architecture (decompose complex into simple for cheap models) aligns with multiple independent research threads.

**Most important finding for GraphBot:** The compounding cost reduction stack (model routing + semantic caching + prompt compression + token budgets + free-tier routing) can reduce effective cost to 2-8% of baseline, validating GraphBot's entire architectural bet on cost efficiency.

---

### [10 - Safety and Alignment](10-safety-and-alignment.md)

**Entries:** 29 | **Sections:** Agent Safety/Sandboxing, Permission Models/Access Control, Human-in-the-Loop Control, Alignment for Autonomous Agents, Agent Security/Attack Surfaces, Safety Evaluation Benchmarks, Guardrails/Safety Filtering, Risk Assessment/Governance Frameworks, Responsible Deployment

Covers ToolEmu (emulated sandbox), fault-tolerant transactional sandboxing, HAICOSYSTEM (3x more risks in multi-turn), multilayer security frameworks, autonomy levels, AURA risk assessment, the argument against fully autonomous agents, Constitutional AI, sleeper agents, alignment faking, legal alignment, AgentDojo, AgentHarm (LLMs "surprisingly compliant" with malicious requests), prompt injection defenses, AutoDefense, NeMo Guardrails, Llama Guard, NIST AI RMF, MAESTRO threat modeling, TRiSM, OWASP Top 10 for LLMs, and RedCode. Identifies GraphBot's unique safety surface: composition attacks (benign subtasks combining into harmful outcomes), injection propagation through DAG levels, recursive explosion, and cross-model trust gaps.

**Most important finding for GraphBot:** "Individually benign subtasks can compose into harmful outcomes" -- GraphBot's DAG decomposition creates a novel attack surface where per-node safety checks are insufficient; a DAG-level intent classifier analyzing the full decomposition plan is the single most critical safety requirement before any public deployment.

---

## Cross-Cutting Themes

Eight patterns appear consistently across three or more research areas:

### 1. DAG/Graph Structure as the Universal Formalism

Files: 01, 03, 05, 06, 07, 08

Graph of Thoughts (01), GPTSwarm and MacNet (05), Agent Workflow Memory (06), Task-Decoupled Planning (07), and LangGraph (08) all converge independently on directed graphs as the correct formalism for agent workflows. GraphBot's DAG architecture is not speculative -- it is the consensus abstraction emerging from multiple research communities.

### 2. Separation of Planning and Execution

Files: 01, 02, 07, 08

Every successful framework separates these concerns: Plan-and-Solve and ReWOO (01), HuggingGPT (02), Plan-and-Act (07), and every production system in the framework survey (08). Unified plan-and-execute approaches consistently underperform specialized planner/executor pairs.

### 3. Small Models with Rich Context Beat Large Models

Files: 02, 03, 07, 09

ToolACE and APIGen (02, 7-8B matching GPT-4 on tool calling), GITM (07, text-based knowledge outperforming vision-based RL), Distilling Step-by-Step (09, 770M outperforming 540B PaLM), and GraphBot's own core thesis all validate that context quality matters more than model size. The temporal knowledge graph is the mechanism that makes this work.

### 4. External Feedback is Essential for Reliable Self-Correction

Files: 03, 04, 07

CRITIC (04), the "Cannot Self-Correct Yet" finding (04), Reflexion (03, 07), and tool-grounded verification all converge on the same result: self-correction without external feedback degrades performance, while self-correction with external signals (knowledge graph, code execution, tool outputs) is consistently effective. GraphBot's knowledge graph is its decisive advantage here.

### 5. Learning from Failures is as Important as Learning from Successes

Files: 03, 05, 07

Reflexion (03), ExpeL (03), ETO/Trial-and-Error (07), MAST failure taxonomy (05), and Agent Q (07) all demonstrate that agents improve dramatically when they learn from failures. Storing failure patterns, verbal reflections on errors, and negative examples in the knowledge graph is identified as the single highest-impact improvement for long-horizon execution quality.

### 6. Memory Evolution Prevents Degradation

Files: 03, 06, 07

Memory consolidation (03), Ebbinghaus/ACT-R forgetting (03), Agent Workflow Memory reuse (06), and skill library curation (07) all address the same problem: unbounded memory accumulation degrades retrieval quality. GraphBot must implement consolidation (merge redundant entries), forgetting (prune low-activation entries), and abstraction (generate summary nodes from clusters) to maintain knowledge graph quality over time.

### 7. Cost Optimization Through Architecture, Not Just Model Choice

Files: 01, 02, 07, 09

ReWOO's 5x token efficiency from plan-then-execute (01), LATM's tool-maker/user split (02), TDP's 82% token reduction from context isolation (07), and the full cost reduction stack (09) show that architectural decisions dominate model selection for cost optimization. GraphBot's DAG decomposition is itself the primary cost optimization mechanism.

### 8. Safety Requires Composition-Level Analysis

Files: 04, 05, 10

The "Cannot Self-Correct" finding (04), MAST failure taxonomy showing 41-87% failure rates (05), composition attacks (10), and the OWASP Top 10 for LLMs (10) converge on the conclusion that per-component safety checks are insufficient for multi-step agent systems. GraphBot must implement DAG-level safety evaluation that analyzes the combined intent and risk of all nodes, not just individual leaf nodes.

---

## Additional Files

| File | Description |
|------|-------------|
| [SUMMARY.md](SUMMARY.md) | Phase 9 research summary covering web scraping, file operations, shell execution, and visualization UX |
| [gap-analysis.md](gap-analysis.md) | Honest codebase audit comparing GraphBot to OpenClaw, identifying the 5 highest-impact fixes |
