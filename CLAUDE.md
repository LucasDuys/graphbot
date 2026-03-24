# CLAUDE.md -- GraphBot Project Orchestration

## Identity

You are the lead architect and sole developer of **GraphBot**, a recursive execution engine powered by a temporal knowledge graph that enables free/cheap LLMs to match the capability of expensive frontier models.

## Project Vision

### The One-Sentence Pitch
A personal AI agent that decomposes any task into a recursive DAG of trivially simple subtasks, executes all independent leaves in parallel on free LLMs, and aggregates results back up -- with a temporal knowledge graph providing hyper-specific context at every level so that 8B models perform like 70B models.

### The Core Thesis
**A small model with perfect context beats a large model with no context.** The knowledge graph eliminates the need for expensive inference by pre-computing what the model needs to know. The recursive DAG eliminates the need for complex reasoning by decomposing until each leaf is trivially simple. The parallel execution eliminates latency.

## Architecture

```
User Message -> Intake Parser (rule-based, zero-cost)
  -> Knowledge Graph Query (Kuzu)
  -> Pattern Match? YES -> Instantiate Template (0 tokens)
                   NO  -> Recursive Decomposer (small model + graph context)
  -> Pipelined DAG Executor (7 concurrent stages)
     DECOMPOSE -> SCHEDULE -> CONTEXTUALIZE -> EXECUTE -> VERIFY -> FORWARD -> AGGREGATE
  -> Graph Update Loop (record task, update entities, extract patterns)
  -> Response
```

## Directory Structure

- `nanobot/` -- Base framework layer (channels, tools, config, session, CLI)
- `graph/` -- Kuzu graph store, schema, context assembly, entity resolution, patterns
- `core_gb/` -- GraphBot core: intake, decomposer, executor, scheduler, forwarder, aggregator
- `agents_gb/` -- Specialized sub-agents (file, web, code, comms, system, synthesis)
- `models/` -- LLM provider abstraction (Groq, Cerebras, OpenRouter, Ollama)
- `tools_gb/` -- GraphBot tool implementations (file, web, shell, browser, dynamic)
- `tests/` -- Unit, integration, and benchmark tests
- `scripts/` -- Seed graph, visualize, benchmark
- `docs/` -- Architecture documentation

## Code Quality Standards

- Type everything (full type hints)
- Test everything (write test before implementation)
- Document everything (docstrings on public interfaces)
- No dead code, no magic numbers
- Structured logging at all levels
- Proper error handling on every external call
- No emojis in code or output

## Commit Format

`[component] brief description`
Examples: `[graph] implement entity resolver`, `[core] add topological sort`

## Working Methodology

1. Check PROGRESS.md
2. Check PLANNING.md
3. Research if needed (log in RESEARCH.md)
4. Write test FIRST
5. Implement
6. Run test
7. Update PROGRESS.md
