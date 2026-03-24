"""Demo: research report pipeline via Orchestrator.

Runs the task "Research the top 5 AI agent frameworks in 2026 and write a
comparison report" through the full Orchestrator pipeline (decomposition ->
parallel web search -> synthesis).  Saves the output report to
demos/research_report.md and prints pipeline stats.

Usage:
    python scripts/demo_research_report.py             # live mode (calls LLMs)
    python scripts/demo_research_report.py --dry-run   # mocked responses, no API calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env.local before any imports that read env vars.
_ENV_FILE = _PROJECT_ROOT / ".env.local"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from core_gb.orchestrator import Orchestrator
from core_gb.types import CompletionResult, ExecutionResult, TaskNode
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter

logger = logging.getLogger(__name__)

DEMO_TASK: str = (
    "Research the top 5 AI agent frameworks in 2026 and write a comparison report"
)

DEMOS_DIR: Path = _PROJECT_ROOT / "demos"
OUTPUT_PATH: Path = DEMOS_DIR / "research_report.md"


# ---------------------------------------------------------------------------
# Pipeline stats
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    """Collected metrics from a single pipeline run."""

    task: str
    success: bool
    nodes_executed: int
    latency_s: float
    total_cost: float
    total_tokens: int
    tools_used: int
    llm_calls: int
    model_used: str
    errors: tuple[str, ...]

    def display(self) -> str:
        """Return a human-readable stats block."""
        lines: list[str] = [
            "",
            "=" * 60,
            "Pipeline Stats",
            "=" * 60,
            f"  Task:           {self.task[:70]}",
            f"  Success:        {self.success}",
            f"  Nodes executed: {self.nodes_executed}",
            f"  Latency:        {self.latency_s:.2f}s",
            f"  Total cost:     ${self.total_cost:.6f}",
            f"  Total tokens:   {self.total_tokens}",
            f"  Tools used:     {self.tools_used}",
            f"  LLM calls:      {self.llm_calls}",
            f"  Model:          {self.model_used or 'n/a'}",
        ]
        if self.errors:
            lines.append(f"  Errors:         {', '.join(self.errors)}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _extract_stats(
    task: str, result: ExecutionResult, elapsed_s: float
) -> PipelineStats:
    """Extract pipeline stats from an ExecutionResult."""
    return PipelineStats(
        task=task,
        success=result.success,
        nodes_executed=result.total_nodes,
        latency_s=elapsed_s,
        total_cost=result.total_cost,
        total_tokens=result.total_tokens,
        tools_used=result.tools_used,
        llm_calls=result.llm_calls,
        model_used=result.model_used,
        errors=result.errors,
    )


# ---------------------------------------------------------------------------
# Dry-run mock infrastructure
# ---------------------------------------------------------------------------

_MOCK_DECOMPOSITION: dict[str, Any] = {
    "nodes": [
        {
            "id": "root",
            "description": "Research top 5 AI agent frameworks and write comparison report",
            "domain": "synthesis",
            "task_type": "THINK",
            "complexity": 3,
            "depends_on": [],
            "provides": [],
            "consumes": [],
            "is_atomic": False,
            "children": ["research", "synthesize"],
        },
        {
            "id": "research",
            "description": "Research all 5 AI agent frameworks in parallel",
            "domain": "web",
            "task_type": "RETRIEVE",
            "complexity": 2,
            "depends_on": [],
            "provides": [],
            "consumes": [],
            "is_atomic": False,
            "children": ["search1", "search2", "search3", "search4", "search5"],
        },
        {
            "id": "search1",
            "description": "Search the web for LangGraph AI agent framework features and capabilities in 2026",
            "domain": "web",
            "task_type": "RETRIEVE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["langgraph_info"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
            "tool_method": "web_search",
            "tool_params": {"query": "LangGraph AI agent framework 2026"},
        },
        {
            "id": "search2",
            "description": "Search the web for CrewAI agent framework features and capabilities in 2026",
            "domain": "web",
            "task_type": "RETRIEVE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["crewai_info"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
            "tool_method": "web_search",
            "tool_params": {"query": "CrewAI agent framework 2026"},
        },
        {
            "id": "search3",
            "description": "Search the web for AutoGen agent framework features and capabilities in 2026",
            "domain": "web",
            "task_type": "RETRIEVE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["autogen_info"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
            "tool_method": "web_search",
            "tool_params": {"query": "AutoGen agent framework 2026"},
        },
        {
            "id": "search4",
            "description": "Search the web for OpenAI Agents SDK features and capabilities in 2026",
            "domain": "web",
            "task_type": "RETRIEVE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["openai_agents_info"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
            "tool_method": "web_search",
            "tool_params": {"query": "OpenAI Agents SDK framework 2026"},
        },
        {
            "id": "search5",
            "description": "Search the web for Anthropic Claude Agent SDK features and capabilities in 2026",
            "domain": "web",
            "task_type": "RETRIEVE",
            "complexity": 1,
            "depends_on": [],
            "provides": ["claude_agents_info"],
            "consumes": [],
            "is_atomic": True,
            "children": [],
            "tool_method": "web_search",
            "tool_params": {"query": "Anthropic Claude Agent SDK 2026"},
        },
        {
            "id": "synthesize",
            "description": "Synthesize research into a structured comparison report of the top 5 AI agent frameworks",
            "domain": "synthesis",
            "task_type": "WRITE",
            "complexity": 2,
            "depends_on": ["search1", "search2", "search3", "search4", "search5"],
            "provides": ["report"],
            "consumes": [
                "langgraph_info",
                "crewai_info",
                "autogen_info",
                "openai_agents_info",
                "claude_agents_info",
            ],
            "is_atomic": True,
            "children": [],
        },
    ],
    "output_template": {
        "aggregation_type": "template_fill",
        "template": (
            "# Top 5 AI Agent Frameworks in 2026 -- Comparison Report\n\n"
            "## 1. LangGraph\n{langgraph_info}\n\n"
            "## 2. CrewAI\n{crewai_info}\n\n"
            "## 3. AutoGen\n{autogen_info}\n\n"
            "## 4. OpenAI Agents SDK\n{openai_agents_info}\n\n"
            "## 5. Anthropic Claude Agent SDK\n{claude_agents_info}\n\n"
            "## Summary\n{report}"
        ),
        "slot_definitions": {
            "langgraph_info": "LangGraph framework overview",
            "crewai_info": "CrewAI framework overview",
            "autogen_info": "AutoGen framework overview",
            "openai_agents_info": "OpenAI Agents SDK overview",
            "claude_agents_info": "Claude Agent SDK overview",
            "report": "Comparative synthesis",
        },
    },
}

_MOCK_SEARCH_RESULTS: dict[str, str] = {
    "langgraph": (
        "LangGraph is a library by LangChain for building stateful, multi-actor "
        "applications with LLMs as graphs. In 2026 it supports cyclic workflows, "
        "streaming, persistence, and human-in-the-loop patterns. Key strengths: "
        "tight LangChain integration, mature ecosystem, strong community. "
        "Weaknesses: steeper learning curve, opinionated graph abstraction."
    ),
    "crewai": (
        "CrewAI enables role-based multi-agent collaboration. Agents are assigned "
        "roles, goals, and backstories. In 2026 it supports hierarchical crews, "
        "tool delegation, and memory. Key strengths: intuitive role metaphor, "
        "rapid prototyping, built-in task delegation. Weaknesses: less flexible "
        "for non-role-based workflows, smaller plugin ecosystem."
    ),
    "autogen": (
        "AutoGen by Microsoft Research provides a framework for building multi-agent "
        "conversational systems. In 2026 it features nested conversations, code "
        "execution sandboxes, and teachable agents. Key strengths: strong research "
        "backing, flexible conversation patterns, code execution. Weaknesses: "
        "complex setup for simple use cases, documentation gaps."
    ),
    "openai_agents": (
        "OpenAI Agents SDK (formerly Swarm) offers a lightweight framework for "
        "building agent systems with handoff patterns. In 2026 it integrates "
        "natively with GPT models, supports tool calling, and provides built-in "
        "tracing. Key strengths: simplicity, native OpenAI integration, production "
        "ready. Weaknesses: vendor lock-in, limited to OpenAI models."
    ),
    "claude_agents": (
        "Anthropic Claude Agent SDK provides a framework for building agents with "
        "Claude models. In 2026 it features computer use, MCP tool integration, "
        "and extended thinking. Key strengths: strong safety features, tool use "
        "capabilities, long context. Weaknesses: Anthropic-only models, newer "
        "ecosystem with fewer community resources."
    ),
}

_MOCK_SYNTHESIS: str = (
    "# Top 5 AI Agent Frameworks in 2026 -- Comparison Report\n\n"
    "## 1. LangGraph\n\n"
    f"{_MOCK_SEARCH_RESULTS['langgraph']}\n\n"
    "## 2. CrewAI\n\n"
    f"{_MOCK_SEARCH_RESULTS['crewai']}\n\n"
    "## 3. AutoGen\n\n"
    f"{_MOCK_SEARCH_RESULTS['autogen']}\n\n"
    "## 4. OpenAI Agents SDK\n\n"
    f"{_MOCK_SEARCH_RESULTS['openai_agents']}\n\n"
    "## 5. Anthropic Claude Agent SDK\n\n"
    f"{_MOCK_SEARCH_RESULTS['claude_agents']}\n\n"
    "## Comparative Summary\n\n"
    "| Framework | Best For | Model Support | Maturity |\n"
    "|-----------|----------|---------------|----------|\n"
    "| LangGraph | Complex stateful workflows | Any via LangChain | High |\n"
    "| CrewAI | Role-based collaboration | Any | Medium |\n"
    "| AutoGen | Research + multi-agent chat | Any | Medium |\n"
    "| OpenAI Agents SDK | Production OpenAI apps | OpenAI only | High |\n"
    "| Claude Agent SDK | Safety-first agents | Anthropic only | Growing |\n\n"
    "**Recommendation:** Choose LangGraph for maximum flexibility, CrewAI for "
    "rapid prototyping with role-based agents, AutoGen for research-oriented "
    "multi-agent systems, OpenAI Agents SDK for production OpenAI deployments, "
    "and Claude Agent SDK for safety-critical applications.\n"
)


def _build_mock_completion(content: str, model: str = "mock/dry-run") -> CompletionResult:
    """Build a mock CompletionResult for dry-run mode."""
    return CompletionResult(
        content=content,
        model=model,
        tokens_in=200,
        tokens_out=300,
        latency_ms=50.0,
        cost=0.0,
    )


def _mock_router_factory() -> AsyncMock:
    """Create a mock ModelRouter that returns canned responses.

    The mock inspects the task description / messages to decide which canned
    response to return:
    - If messages contain the decomposition system prompt, return the mock DAG.
    - If the description mentions a specific framework, return that search result.
    - Otherwise return the synthesis report.
    """
    mock_router = AsyncMock(spec=ModelRouter)

    call_count: list[int] = [0]

    async def _route_side_effect(
        node: TaskNode,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> CompletionResult:
        call_count[0] += 1
        combined_text = " ".join(m.get("content", "") for m in messages).lower()

        # Decomposition call: system prompt contains the decomposition schema
        if "task decomposer" in combined_text:
            return _build_mock_completion(json.dumps(_MOCK_DECOMPOSITION))

        # Search calls: check for framework names in the task description
        desc = (node.description or "").lower()
        for key, text in _MOCK_SEARCH_RESULTS.items():
            # Match framework name fragments in the node description
            fragments = key.replace("_", " ").split()
            if any(frag in desc for frag in fragments):
                return _build_mock_completion(text)

        # Synthesis / aggregation call
        return _build_mock_completion(_MOCK_SYNTHESIS)

    mock_router.route = AsyncMock(side_effect=_route_side_effect)
    mock_router.call_count = call_count
    return mock_router


# ---------------------------------------------------------------------------
# Run functions
# ---------------------------------------------------------------------------

async def run_live() -> tuple[ExecutionResult, float]:
    """Run the research report task through the live Orchestrator pipeline."""
    db_path = str(_PROJECT_ROOT / "data" / "demo_research.db")
    store = GraphStore(db_path)
    store.initialize()

    provider = OpenRouterProvider()
    router = ModelRouter(provider=provider)
    orchestrator = Orchestrator(store, router)

    print(f"Task: {DEMO_TASK}")
    print("Running through Orchestrator pipeline (live)...")
    print()

    start = time.time()
    result = await orchestrator.process(DEMO_TASK)
    elapsed = time.time() - start

    store.close()
    return result, elapsed


async def run_dry() -> tuple[ExecutionResult, float]:
    """Run the research report task with mocked LLM responses (no API calls)."""
    store = GraphStore()  # in-memory
    store.initialize()

    mock_router = _mock_router_factory()

    # Patch OpenRouterProvider so Orchestrator does not try to validate an API key.
    with patch("core_gb.orchestrator.ModelRouter", return_value=mock_router):
        orchestrator = Orchestrator(store, mock_router)  # type: ignore[arg-type]

    print(f"Task: {DEMO_TASK}")
    print("Running through Orchestrator pipeline (dry-run, mocked responses)...")
    print()

    start = time.time()
    result = await orchestrator.process(DEMO_TASK)
    elapsed = time.time() - start

    store.close()
    return result, elapsed


def save_report(result: ExecutionResult) -> Path:
    """Save the output report to demos/research_report.md."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(result.output, encoding="utf-8")
    return OUTPUT_PATH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo: research report pipeline via Orchestrator",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use mocked LLM responses instead of live API calls",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging for pipeline stages",
    )
    return parser.parse_args()


async def async_main() -> None:
    """Async entry point."""
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.dry_run:
        result, elapsed = await run_dry()
    else:
        result, elapsed = await run_live()

    # Print output preview
    output_text = result.output or "(no output)"
    preview = output_text[:500]
    if len(output_text) > 500:
        preview += "\n... (truncated, full report saved to file)"
    print("--- Output Preview ---")
    print(preview)
    print("--- End Preview ---")

    # Pipeline stats
    stats = _extract_stats(DEMO_TASK, result, elapsed)
    print(stats.display())

    # Save report
    if result.success and result.output:
        saved_path = save_report(result)
        print(f"Report saved to: {saved_path}")
    else:
        print("Pipeline did not produce a successful output; report not saved.")
        if result.errors:
            print(f"Errors: {', '.join(result.errors)}")


def main() -> None:
    """Synchronous entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
