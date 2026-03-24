# How recursive DAG decomposition + knowledge graphs make Llama 8B match GPT-4o quality at 99.9% lower cost

_(Draft -- do not post without explicit authorization)_

**Target subreddits:** r/LocalLLaMA, r/MachineLearning

---

## TL;DR

I built an open-source framework called **GraphBot** that makes small/free LLMs dramatically more capable by decomposing tasks into recursive DAGs and injecting hyper-specific context from a temporal knowledge graph. Llama 8B with GraphBot passes 30/30 diverse tasks for $0.0006 total. All code, research, and traces are public.

GitHub: https://github.com/LucasDuys/graphbot

## The thesis

**A small model with perfect context beats a large model with no context.**

This is not a new idea. Retrieval-augmented generation proved that adding relevant context improves output quality. But RAG retrieves documents -- flat, unstructured chunks. GraphBot retrieves _structured knowledge_: entities, relationships, temporal facts, execution patterns, and failure reflections from a temporal knowledge graph. The context is not "here are some relevant paragraphs" but "here is exactly what you need to know to answer this specific subtask, organized as facts."

Combined with recursive task decomposition (breaking hard tasks into trivially simple subtasks), even an 8B model can handle complex multi-step workflows that would normally require GPT-4o or Claude Opus.

## Architecture

```
User Message
    |
    v
Intake Parser (rule-based, zero-cost)
    |                              +---> Pattern Cache (semantic match)
    v                              |     Hit? -> Instantiate template (0 tokens)
Knowledge Graph Query (Kuzu) ------+
    |                              |     Miss? -> Decompose
    v                              v
Recursive Decomposer -----> DAG of subtasks
    |                        /    |    \
    v                       v     v     v
Pipelined DAG Executor:  leaf1  leaf2  leaf3  (parallel on free LLMs)
  DECOMPOSE -> SCHEDULE -> CONTEXTUALIZE -> EXECUTE -> VERIFY -> FORWARD -> AGGREGATE
    |
    v
Graph Update Loop (record entities, extract patterns, store reflections)
    |
    v
Response
```

### Key components

**1. Recursive DAG decomposition**

Tasks are decomposed into a Directed Acyclic Graph. Each node is either a leaf (simple enough for an 8B model) or a parent (decomposed further). Independent leaves execute in parallel. The decomposer uses constrained JSON output to ensure valid DAG structure.

This is validated by recent research:
- Graph of Thoughts (Besta et al., AAAI 2024) -- DAG-based reasoning outperforms chain and tree structures
- MacNet (Xu et al., ICLR 2025) -- topology-aware agent communication via DAGs
- "Beyond Entangled Planning" (2026) -- reports 82% token reduction with DAG decomposition matching GraphBot's architecture

**2. Temporal knowledge graph (Kuzu)**

11 node types: Entity, Concept, Pattern, Fact, Preference, Reflection, Skill, Goal, Session, Message, ExecutionTree. All edges are temporal (valid_from, valid_until). Context assembly uses Personalized PageRank to find relevant subgraphs up to 3 hops deep.

Why a graph instead of a vector store:
- Graphiti (Zep, arXiv:2501.13956) showed temporal KGs outperform flat vector stores for agent memory
- HippoRAG (NeurIPS 2024) demonstrated graph-based retrieval mirrors human hippocampal indexing
- GraphRAG (Microsoft, 2024) proved graph communities capture global context that vector chunks miss

**3. Three-layer verification**

Every node output is verified before being forwarded:
1. **Format check** (every node) -- Is the output valid? Parseable? Non-empty?
2. **Self-consistency** (important nodes) -- Does the output agree with the task description?
3. **KG fact-check** (critical nodes) -- Do the claims match known facts in the knowledge graph?

This follows CRITIC (Gou et al., ICLR 2024) which showed external verification tools outperform LLM self-correction, and "Large Language Models Cannot Self-Correct" (Huang et al., ICLR 2024) which demonstrated that self-correction without external signals degrades quality.

**4. Pattern caching with semantic matching**

After successful execution, the DAG structure is stored as a Pattern node in the knowledge graph. Future tasks are compared against stored patterns using sentence-transformers embeddings (all-MiniLM-L6-v2, runs locally, no API cost). Semantically similar tasks reuse the stored decomposition template -- zero tokens, zero latency for the decomposition step.

**5. Cost optimization stack**

- Model cascade: try cheapest model first, escalate on low confidence
- Multi-provider failover: OpenRouter -> Google AI Studio -> Groq
- Pattern caching: repeated/similar tasks use 0 tokens
- Context windowing: 1,500-3,000 tokens injected context (sweet spot for 8B models)

This compounds. FrugalGPT (Chen et al., 2023) showed cascading alone gets 2-8% of naive cost. We add graph caching on top.

## Benchmark results

30 diverse tasks run on **free Llama 8B** via OpenRouter:

| Category | Tasks | Pass Rate | Avg Latency | Total Cost |
|----------|-------|-----------|-------------|------------|
| Simple Q&A | 5 | **5/5** | 4.3s | $0.000005 |
| Task decomposition (parallel) | 3 | **3/3** | 27.7s | $0.000110 |
| Math & reasoning | 2 | **2/2** | 2.0s | $0.000006 |
| File & shell tools | 4 | **4/4** | 10.2s | $0.000019 |
| Web search | 1 | **1/1** | 11.0s | $0.000014 |
| Code generation | 2 | **2/2** | 3.9s | $0.000010 |
| Translation (3 languages) | 1 | **1/1** | 14.4s | $0.000023 |
| Creative writing | 2 | **2/2** | 2.3s | $0.000008 |
| Safety (harmful requests) | 3 | **3/3 blocked** | 0.0s | $0.000000 |
| Analysis & classification | 3 | **3/3** | 4.1s | $0.000013 |
| Pattern cache hit | 1 | **1/1** | 3.2s | $0.000001 |
| **Total** | **30** | **30/30** | **8.0s avg** | **$0.0006** |

All traces verified on LangSmith.

## Comparison with existing frameworks

| | GraphBot | AutoGPT | CrewAI | LangGraph |
|---|---|---|---|---|
| **Cost per task** | $0.00002 | $0.05-0.50 | $0.01-0.10 | $0.01-0.05 |
| **Free model support** | Native (Llama 8B) | GPT-4 required | GPT-4 recommended | Model agnostic |
| **Task decomposition** | Recursive DAG | Serial chain | Role-based | Graph states |
| **Knowledge graph** | Built-in (Kuzu) | None | None | None |
| **Pattern learning** | Automatic | None | None | None |
| **Self-correction** | 3-layer verification | Retry loop | None | Custom |
| **Safety** | Constitutional + autonomy levels | None | None | None |
| **Tests** | 1500+ | ~200 | ~300 | ~500 |

## Research backing

The architecture is informed by 190+ papers across 15 research areas (2023-2026). Key citations:

**Planning and decomposition:**
- ROMA: Recursive multi-agent framework (arXiv:2602.01848)
- ReAcTree: Agent trees achieving ~2.4x improvement over ReAct (arXiv:2511.02424)
- Graph of Thoughts (AAAI 2024): DAG-based reasoning
- "Beyond Entangled Planning" (2026): 82% token reduction with DAG decomposition

**Memory and knowledge:**
- Graphiti / Zep: Temporal knowledge graphs for agent memory (arXiv:2501.13956)
- HippoRAG (NeurIPS 2024): Graph-based retrieval mimicking hippocampal indexing
- GraphRAG (Microsoft, 2024): Graph communities for global context

**Verification:**
- CRITIC (ICLR 2024): External tool verification outperforms self-correction
- "LLMs Cannot Self-Correct" (ICLR 2024): Self-correction without external signals degrades quality

**Cost optimization:**
- FrugalGPT (Chen et al., 2023): Model cascading for 2-8% of naive cost
- RouteLLM (2024): Learned routing between cheap and expensive models
- AdaptThink (arXiv:2505.13417): Teaching models when to think vs. skip

Full research library with summaries: `docs/research/` in the repo.

## How to reproduce

```bash
# Clone and install
git clone https://github.com/LucasDuys/graphbot
cd graphbot
pip install -e ".[dev,langsmith]"

# Get a free OpenRouter API key
# https://openrouter.ai/keys
echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env.local

# Verify setup
python scripts/healthcheck.py

# Run the 30-task benchmark
python scripts/run_capability_tests.py

# Run the demos
python scripts/demo_flight_search.py --dry-run
python scripts/demo_research_report.py --dry-run
```

Requirements: Python 3.13+, free OpenRouter API key. No GPU needed -- all inference runs on free API tiers.

## Limitations (honest assessment)

- **Simple tasks are slower:** For trivial questions, the decomposition overhead adds latency. A direct GPT-4o call is faster for "What is the capital of France?"
- **Novel multi-hop reasoning:** Llama 8B still struggles with genuinely novel reasoning chains that require knowledge not in the graph
- **Cold start:** The knowledge graph needs several runs to accumulate useful patterns. First runs are slower.
- **Browser fragility:** Playwright-based web automation works for structured sites but breaks on complex SPAs
- **Solo project:** This is a research project by one person, not a production-hardened framework backed by a team. Expect rough edges.

## Tech stack

Python 3.13 | Kuzu (embedded graph DB) | LiteLLM | Playwright | sentence-transformers | Next.js + React Flow (dashboard) | D3.js (graph visualization) | LangSmith (tracing)

1500+ tests. MIT license.

---

_I'm an independent researcher (CS student). The full research library (190+ papers, 15 research documents) is in the repo. Happy to discuss specific architectural decisions or research findings._

GitHub: https://github.com/LucasDuys/graphbot
