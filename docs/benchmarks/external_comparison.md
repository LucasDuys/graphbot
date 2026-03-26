# GraphBot vs External Agent Frameworks: Benchmark Comparison

**Last updated:** March 2026
**Methodology:** Published metrics from official leaderboards, blog posts, and papers. GraphBot numbers from our 15-task benchmark (`benchmarks/single_call_validation.json`). All external numbers cited with sources.

---

## Table of Contents

1. [GraphBot Current Performance](#1-graphbot-current-performance)
2. [OpenHands Comparison](#2-openhands-comparison)
3. [CrewAI Comparison](#3-crewai-comparison)
4. [AutoGPT Comparison](#4-autogpt-comparison)
5. [LangGraph Comparison](#5-langgraph-comparison)
6. [Head-to-Head Summary](#6-head-to-head-summary)
7. [Where GraphBot Wins](#7-where-graphbot-wins)
8. [Where Others Win](#8-where-others-win)
9. [Honest Limitations](#9-honest-limitations)
10. [Sources](#10-sources)

---

## 1. GraphBot Current Performance

From our 15-task benchmark (easy Q&A, hard multi-step, tool-dependent tasks), judged by GPT-4o-mini on a 1-5 quality scale:

| Mode | Avg Quality | Total Cost | Avg Latency |
|------|-------------|------------|-------------|
| **Single-Call + Context** | **3.87/5** | **$0.004** | **20.4s** |
| Decomposition | 3.73/5 | $0.005 | 381.5s |
| GPT-4o Direct | 3.40/5 | $0.022 | 3.2s |

**By difficulty (single-call mode):**

| Task Type | Quality | Cost | Latency |
|-----------|---------|------|---------|
| Easy (trivial Q&A) | 5.00/5 | $0.000005 | 2.1s |
| Hard (multi-step) | 3.60/5 | $0.000703 | 39.9s |
| Tool (file/shell/web) | 3.00/5 | $0.000089 | 19.2s |

**Additional results:**
- 30-task capability benchmark: 30/30 pass rate, $0.0006 total cost
- 14/14 adversarial attacks blocked in 0ms
- 10/10 stress test tasks passed
- 1900+ tests in the test suite
- Model: Free Llama 8B via OpenRouter

---

## 2. OpenHands Comparison

[OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin) is an open-source AI software engineering agent. It introduced the [OpenHands Index](https://openhands.dev/blog/openhands-index) in January 2026, benchmarking 9 LLMs across 5 task domains.

### Published Metrics

| Benchmark | OpenHands Score | Notes |
|-----------|----------------|-------|
| SWE-bench Verified (Claude Opus 4.5) | ~80.9% resolve rate | Top-tier with frontier model [1] |
| SWE-bench Verified (Claude 3.7 Sonnet) | 43.2% resolve rate | More affordable model tier [2] |
| SWE-bench Live | ~19.3% (best) | Harder, more realistic benchmark [3] |
| GAIA (information gathering) | State-of-the-art | OpenHands-Versa: +9.1 pts over prior best [4] |

### Cost Profile

OpenHands relies on paid frontier models for competitive performance:

- **Claude Sonnet 4.6**: $3/M input tokens, $15/M output tokens [5]
- **Claude Opus 4.5**: $15/M input tokens, $75/M output tokens [5]
- **Estimated cost per SWE-bench task**: $0.10-$2.00+ depending on model and task complexity (multi-turn agent loop with tool use)
- **OpenHands Cloud**: Previously charged 2x API costs; now offers subscription tiers [6]

### Head-to-Head: GraphBot vs OpenHands

| Dimension | GraphBot | OpenHands |
|-----------|----------|-----------|
| **Primary strength** | General tasks at near-zero cost | Software engineering tasks |
| **Cost per task** | $0.00027 avg ($0.004 / 15 tasks) | $0.10-$2.00+ (frontier model API) |
| **Model requirement** | Free Llama 8B | Claude Opus/Sonnet (paid) |
| **Code generation quality** | 3.00/5 (tool tasks) | 80.9% SWE-bench (not comparable) |
| **Information gathering** | 3.87/5 overall | GAIA state-of-the-art |
| **Safety guardrails** | Constitutional + pre-decomposition | Sandboxed execution |
| **Knowledge graph** | Built-in (Kuzu, 11 node types) | None |
| **Pattern learning** | Automatic (zero-token cache hits) | None |

**Verdict:** OpenHands dominates on code-specific tasks when paired with frontier models. GraphBot wins on cost (500-7000x cheaper per task) and on general-purpose tasks where knowledge graph context adds value. These are fundamentally different tools: OpenHands is a coding agent, GraphBot is a general-purpose task pipeline.

---

## 3. CrewAI Comparison

[CrewAI](https://github.com/crewAIInc/crewAI) is a multi-agent orchestration framework using role-based agent collaboration. It has 80K+ GitHub stars and is widely used for business automation.

### Published Metrics

| Metric | CrewAI | Source |
|--------|--------|--------|
| Execution speed vs LangGraph | 5.76x faster on QA tasks | [7] |
| Token efficiency | 15-20% fewer tokens than LangGraph (sequential) | [7] |
| Memory usage | 200-300 MB per 3-agent crew | [7] |
| Model requirement | GPT-4 recommended, supports others | [8] |

CrewAI does not publish standardized benchmark scores (SWE-bench, GAIA). Their metrics focus on comparative framework performance rather than absolute task quality.

### Estimated Cost Profile

CrewAI tasks typically use GPT-4 or GPT-4o:
- **GPT-4o**: $2.50/M input, $10/M output tokens
- **Estimated cost per task**: $0.01-$0.10 (multi-agent conversations multiply token usage)
- Role-based design means 2-5 agents per task, each making LLM calls

### Head-to-Head: GraphBot vs CrewAI

| Dimension | GraphBot | CrewAI |
|-----------|----------|--------|
| **Architecture** | DAG with smart routing | Role-based multi-agent |
| **Cost per task** | $0.00027 avg | $0.01-$0.10 est. |
| **Free model support** | Native (Llama 8B) | GPT-4 recommended |
| **Task decomposition** | Recursive DAG (auto) | Role assignment (manual) |
| **Knowledge graph** | Built-in | None |
| **Pattern learning** | Automatic | None |
| **Community** | Small (new project) | Large (80K+ stars) |
| **Ecosystem** | Self-contained | Extensive integrations |
| **Best for** | Cost-sensitive, general tasks | Business process automation |

**Verdict:** CrewAI has a far larger ecosystem and community. GraphBot is 37-370x cheaper per task and works with free models. CrewAI's role-based design is better for business workflows; GraphBot's DAG execution is better for structured task decomposition.

---

## 4. AutoGPT Comparison

[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) was one of the first autonomous AI agent projects (160K+ GitHub stars). It pioneered the concept of LLM task loops but has been surpassed by more structured approaches.

### Published Metrics

AutoGPT does not maintain a public benchmark leaderboard. Available data:

| Metric | AutoGPT | Source |
|--------|---------|--------|
| SWE-bench performance | Not competitive with modern agents | [9] |
| Task completion | Serial chain execution | Architecture limitation |
| Model requirement | GPT-4 required for reliable operation | [10] |
| Cost per task | $0.05-$0.50 (GPT-4 loops, often 10-30+ calls) | Community reports |
| Test coverage | ~200 tests | Repository analysis |

AutoGPT's loop-based architecture (plan -> execute -> evaluate -> repeat) often leads to:
- Excessive API calls (10-30+ per task)
- Loop divergence on complex tasks
- High cost from uncontrolled iteration

### Head-to-Head: GraphBot vs AutoGPT

| Dimension | GraphBot | AutoGPT |
|-----------|----------|---------|
| **Architecture** | DAG with verification | Serial loop |
| **Cost per task** | $0.00027 avg | $0.05-$0.50 est. |
| **Free model support** | Native | GPT-4 required |
| **Loop control** | DAG structure prevents divergence | Can loop indefinitely |
| **Knowledge graph** | Built-in | None |
| **Safety** | Constitutional + composition detection | None built-in |
| **Community** | Small | Very large (160K+ stars) |
| **Maturity** | Active development | Pivoted to "AutoGPT Platform" |

**Verdict:** AutoGPT pioneered the space but its serial loop architecture has fundamental limitations. GraphBot's DAG-based approach avoids the loop divergence problem and is 185-1850x cheaper. AutoGPT has a massive community but the project has pivoted away from its original autonomous agent design.

---

## 5. LangGraph Comparison

[LangGraph](https://github.com/langchain-ai/langgraph) is LangChain's agent orchestration framework built on state machines. It is now the recommended approach for building agents in the LangChain ecosystem.

### Published Metrics

| Metric | LangGraph | Source |
|--------|-----------|--------|
| Latency | Lowest among LangChain family | [11] |
| Overhead per query | ~14ms framework overhead | [11] |
| Token efficiency | Baseline (CrewAI uses 15-20% fewer) | [7] |
| Model support | Model agnostic | [11] |
| Production deployments | Klarna: 2.5M conversations, 80% faster resolution | [12] |

LangGraph does not publish standardized benchmark scores. Its value proposition is flexibility and production-readiness rather than out-of-the-box task performance.

### Estimated Cost Profile

LangGraph is a framework, so cost depends entirely on the model used:
- **With GPT-4o**: $0.01-$0.05 per task (typical agent loop)
- **With free models**: Possible but less common, no published benchmarks
- **Framework overhead**: Minimal (14ms per query)

### Head-to-Head: GraphBot vs LangGraph

| Dimension | GraphBot | LangGraph |
|-----------|----------|-----------|
| **Architecture** | Opinionated DAG pipeline | Flexible state machine |
| **Cost per task** | $0.00027 avg | $0.01-$0.05 (model dependent) |
| **Free model support** | Native + optimized | Possible but not optimized |
| **Out-of-box features** | Full pipeline (safety, KG, decomposition) | Build-your-own |
| **Knowledge graph** | Built-in | None (add your own) |
| **Checkpointing** | Graph-based memory | Built-in state checkpoints |
| **Flexibility** | Opinionated (less flexible) | Highly flexible |
| **Production scale** | Research project | Production-proven (Klarna) |
| **Community** | Small | Very large (LangChain ecosystem) |

**Verdict:** LangGraph is a flexible framework for building custom agents; GraphBot is an opinionated pipeline that works out of the box. LangGraph wins on flexibility, production maturity, and ecosystem. GraphBot wins on cost and built-in intelligence (knowledge graph, pattern learning, safety). LangGraph requires you to build what GraphBot provides by default.

---

## 6. Head-to-Head Summary

| Framework | Task Quality | Cost/Task | Latency | Free Tier? | Safety Guards | Knowledge Graph | Pattern Learning |
|-----------|-------------|-----------|---------|------------|---------------|-----------------|------------------|
| **GraphBot** | 3.87/5 | **$0.00027** | 20.4s | **Yes (native)** | **Constitutional + composition** | **Yes (Kuzu)** | **Yes** |
| **OpenHands** | 80.9% SWE-bench* | $0.10-$2.00 | Minutes | No | Sandboxed | No | No |
| **CrewAI** | Not published | $0.01-$0.10 | Varies | No | No | No | No |
| **AutoGPT** | Not published | $0.05-$0.50 | Varies | No | No | No | No |
| **LangGraph** | Not published | $0.01-$0.05 | Low overhead | Partial | No | No | No |

*Quality metrics are not directly comparable. GraphBot uses a general 1-5 scale across diverse tasks. OpenHands uses SWE-bench (code-only). Others do not publish standardized scores.*

### Cost Comparison (Normalized)

| Framework | Cost for 15 general tasks | Multiplier vs GraphBot |
|-----------|--------------------------|----------------------|
| **GraphBot** | $0.004 | **1x (baseline)** |
| LangGraph + GPT-4o | ~$0.15-$0.75 | 37-187x |
| CrewAI + GPT-4o | ~$0.15-$1.50 | 37-375x |
| AutoGPT + GPT-4 | ~$0.75-$7.50 | 187-1875x |
| OpenHands + Claude Opus | ~$1.50-$30.00 | 375-7500x |

---

## 7. Where GraphBot Wins

### Near-Zero Cost
$0.004 for 15 tasks ($0.00027 per task average). This is 37-7500x cheaper than alternatives. The difference comes from:
- Free Llama 8B via OpenRouter (zero model cost)
- Single-call mode avoids multi-turn agent loops
- Pattern cache eliminates repeated computation (0 tokens for cached tasks)

### Constitutional Safety
Pre-decomposition blocking catches harmful requests in 0ms with zero LLM calls. 14/14 adversarial attacks blocked. No other framework in this comparison has built-in constitutional safety.

### Knowledge Graph Context
Kuzu-backed graph with 11 node types provides context that pure LLM calls cannot. Personalized PageRank retrieval finds connections 3+ hops deep. This is architecturally unique among the compared frameworks.

### Works with Free Models
GraphBot is designed and optimized for free/cheap models. All benchmarks run on Llama 8B. Other frameworks require or strongly recommend paid frontier models.

### Pattern Learning
Automatic pattern extraction means repeated tasks use zero tokens. No other framework in this comparison learns from its own execution history.

---

## 8. Where Others Win

### Code Generation and Software Engineering (OpenHands)
OpenHands + Claude Opus achieves 80.9% on SWE-bench Verified. GraphBot scores 3.00/5 on tool tasks (which include code generation). For professional software engineering tasks, OpenHands with a frontier model is far more capable.

### Production Scale (LangGraph)
LangGraph powers production deployments like Klarna's 2.5M conversation system. GraphBot is a research project without production deployment evidence.

### Ecosystem and Community (All)
- AutoGPT: 160K+ GitHub stars
- CrewAI: 80K+ GitHub stars
- LangGraph: LangChain ecosystem (massive)
- GraphBot: New project, small community

### Flexibility (LangGraph)
LangGraph's state machine architecture lets you build any agent topology. GraphBot's opinionated DAG pipeline is less flexible.

### Broader Tool Ecosystem (CrewAI, LangGraph)
CrewAI and LangGraph have extensive third-party integrations, tool libraries, and plugin ecosystems. GraphBot has built-in tools (file, web, shell, browser) but a smaller integration surface.

### Frontier Model Performance (OpenHands)
When cost is not a concern, OpenHands + Claude Opus or GPT-5 produces significantly higher quality results on complex tasks. GraphBot's free Llama 8B has a knowledge ceiling that graph context can structure but not overcome.

---

## 9. Honest Limitations

### What our benchmarks do NOT prove

1. **Quality scores are not directly comparable.** Our 1-5 GPT-4o-mini judge scale is not the same as SWE-bench resolve rates or GAIA accuracy. We compare cost and architecture honestly, but quality comparisons across different benchmarks are inherently approximate.

2. **Our benchmark is small.** 15 tasks vs SWE-bench's 500+ verified instances. Our results are directionally useful but not statistically robust at scale.

3. **We have not run GAIA or SWE-bench.** We cannot make direct claims about GraphBot's performance on these benchmarks. Running them would require significant infrastructure work and is planned for future phases.

4. **Cost estimates for other frameworks are approximate.** We use published pricing and community reports, not controlled experiments with identical tasks.

5. **Latency comparison favors others.** GraphBot's 20.4s average is significantly slower than GPT-4o direct (3.2s) and LangGraph's minimal overhead. The pipeline adds real latency.

6. **Tool task quality is our weakest area.** 3.00/5 on tool tasks suggests the tool execution pipeline needs improvement, which is acknowledged in our roadmap.

---

## 10. Sources

1. [SWE-bench Verified Leaderboard](https://www.swebench.com/) -- Claude Opus 4.5 at 80.9% resolve rate
2. [OpenHands SWE-bench Evaluation](https://openhands.dev/blog/evaluation-of-llms-as-coding-agents-on-swe-bench-at-30x-speed) -- Claude 3.7 Sonnet at 43.2%
3. [SWE-bench Live Leaderboard](https://swe-bench-live.github.io/) -- Best score ~19.3%
4. [OpenHands Index](https://openhands.dev/blog/openhands-index) -- GAIA and multi-benchmark evaluation
5. [Claude API Pricing](https://platform.claude.com/docs/en/about-claude/pricing) -- Current token pricing
6. [OpenHands Cloud Pricing](https://openhands.dev/blog/all-new-openhands-cloud-more-model-choice-lower-prices-slicker-design) -- Pricing model changes
7. [CrewAI vs LangGraph Performance](https://www.secondtalent.com/resources/crewai-vs-autogen-usage-performance-features-and-popularity-in/) -- 5.76x speed advantage, token efficiency
8. [CrewAI Framework Review](https://latenode.com/blog/ai-frameworks-technical-infrastructure/crewai-framework/crewai-framework-2025-complete-review-of-the-open-source-multi-agent-ai-platform) -- Architecture and capabilities
9. [AI Computer-Use Benchmarks Guide 2025-2026](https://o-mega.ai/articles/the-2025-2026-guide-to-ai-computer-use-benchmarks-and-top-ai-agents) -- AutoGPT surpassed by structured approaches
10. [AutoGPT Benchmarks Repository](https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks) -- Benchmark infrastructure
11. [LangGraph Performance Analysis](https://www.alphabold.com/langchain-vs-langgraph/) -- Latency and overhead metrics
12. [LangGraph at Klarna](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more) -- Production deployment metrics
13. [Benchmarking Multi-Agent Architectures](https://blog.langchain.com/benchmarking-multi-agent-architectures/) -- LangChain's own analysis
14. [SWE-bench Pro: Why 46% Beats 81%](https://www.morphllm.com/swe-bench-pro) -- Benchmark validity concerns
