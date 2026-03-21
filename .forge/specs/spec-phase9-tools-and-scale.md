---
domain: phase9-tools-and-scale
status: approved
created: 2026-03-21
complexity: complex
linked_repos: []
---

# Phase 9: Tool System + Real-World Scale Proof

## Overview

Build GraphBot's tool system (file ops, web scraping, shell execution) as leaf
executors within the existing DAG architecture. Each tool is just another atomic
node type that the decomposer can route to.

METHODOLOGY: Every sub-phase starts with deep research (multiple parallel agents
investigating best practices, papers, existing implementations). Research is
documented in docs/research/ BEFORE any code is written. Implementation is driven
from the research documentation.

## Requirements

### R001: Research Phase -- Web Scraping Best Practices
Deep research before building anything web-related.
**Acceptance Criteria:**
- [ ] docs/research/web-scraping.md written with findings from 3+ parallel research agents
- [ ] Topics covered: Playwright vs httpx vs Firecrawl, token-efficient page representation (accessibility trees, readability extraction, structured markdown), anti-bot handling, rate limiting, robots.txt compliance
- [ ] Specific investigation: how OpenClaw/Cline handles web content (Semantic Snapshots / accessibility trees)
- [ ] Specific investigation: how to convert HTML to minimal token representation (Jina Reader, Readability, Mozilla's readability-lxml, Trafilatura)
- [ ] Specific investigation: Playwright MCP server capabilities and tool interface
- [ ] Recommended approach documented with rationale
- [ ] Token budget analysis: raw HTML vs cleaned text vs structured extraction (measure token savings)

### R002: Research Phase -- File Operations Best Practices
Deep research before building file tools.
**Acceptance Criteria:**
- [ ] docs/research/file-operations.md written
- [ ] Topics: safe file editing (diff-based vs full rewrite), path sandboxing, encoding handling, large file streaming, binary file detection
- [ ] How OpenClaw handles file edits (search-and-replace vs full rewrite)
- [ ] How to represent file content efficiently for small models (line numbers, context windows, truncation)
- [ ] Recommended approach documented

### R003: Research Phase -- Shell Execution Best Practices
Deep research before building shell tools.
**Acceptance Criteria:**
- [ ] docs/research/shell-execution.md written
- [ ] Topics: command sandboxing, timeout handling, output capture (stdout/stderr), working directory management, environment isolation
- [ ] Security: command injection prevention, allowlist/blocklist patterns
- [ ] How to represent command output efficiently for small models
- [ ] Windows + Unix compatibility considerations
- [ ] Recommended approach documented

### R004: Research Phase -- Non-Technical Visualization
Research how to make the DAG visualization understandable to non-technical users.
**Acceptance Criteria:**
- [ ] docs/research/visualization-ux.md written
- [ ] Topics: progressive disclosure (show simple view, expand to technical), animation pacing, status storytelling
- [ ] How Vercel, Linear, Raycast communicate technical processes to broad audiences
- [ ] Color psychology for status communication (beyond red/green)
- [ ] How to narrate what GraphBot is doing in plain English alongside the DAG
- [ ] Recommended UI enhancements documented

### R005: Web Tools -- Playwright Integration
Build web scraping/browsing as leaf executor tools.
**Acceptance Criteria:**
- [ ] `tools_gb/web.py` with WebTool class
- [ ] `web_fetch(url) -> str` -- fetch page, extract clean text (minimal tokens)
- [ ] `web_search(query) -> list[dict]` -- search via DuckDuckGo (already in Nanobot deps: ddgs)
- [ ] `web_scrape(url, selector?) -> str` -- extract structured data from page
- [ ] Content cleaning: HTML -> clean markdown/text using research-recommended approach
- [ ] Token budget: cleaned output must be <30% of raw HTML token count
- [ ] Playwright MCP integration for JS-rendered pages
- [ ] Rate limiting: max 10 requests/minute per domain
- [ ] robots.txt respect (fetch and check before scraping)
- [ ] Test: fetch a static page, verify clean output
- [ ] Test: search returns structured results
- [ ] Test: token reduction measured (raw HTML vs cleaned)

### R006: File Tools
Build file operations as leaf executor tools.
**Acceptance Criteria:**
- [ ] `tools_gb/file.py` with FileTool class
- [ ] `file_read(path) -> str` -- read file content with line numbers
- [ ] `file_write(path, content) -> str` -- write/create file
- [ ] `file_edit(path, old_text, new_text) -> str` -- search-and-replace edit
- [ ] `file_list(directory, pattern?) -> list[str]` -- glob listing
- [ ] `file_search(directory, query) -> list[dict]` -- grep-like content search
- [ ] Path sandboxing: all paths resolved relative to allowed root (configurable)
- [ ] Large file handling: truncation with "... truncated" for files > 10K lines
- [ ] Binary file detection: skip binary files gracefully
- [ ] Test: full CRUD cycle on temp files
- [ ] Test: path traversal blocked (../../etc/passwd rejected)
- [ ] Test: large file truncated correctly

### R007: Shell Tools
Build shell execution as leaf executor tools.
**Acceptance Criteria:**
- [ ] `tools_gb/shell.py` with ShellTool class
- [ ] `shell_run(command, timeout=30) -> dict` -- run command, return {stdout, stderr, exit_code}
- [ ] Timeout enforcement (default 30s, configurable)
- [ ] Output truncation: max 5000 chars stdout, 2000 chars stderr
- [ ] Working directory: configurable, defaults to project root
- [ ] Command blocklist: prevent destructive commands (rm -rf /, format, etc.)
- [ ] Environment isolation: no access to secret env vars (filter OPENROUTER_API_KEY etc from env)
- [ ] Windows + Unix compatibility (subprocess with shell=True on Windows, shell=False on Unix)
- [ ] Test: run simple command (echo, dir/ls)
- [ ] Test: timeout kills long-running process
- [ ] Test: blocked command rejected
- [ ] Test: output truncated for long output

### R008: Tool Registry + Domain Routing
Wire tools into the decomposer and DAG executor.
**Acceptance Criteria:**
- [ ] `tools_gb/registry.py` with ToolRegistry class
- [ ] Registry maps Domain enum values to tool instances: FILE -> FileTool, WEB -> WebTool, CODE -> ShellTool
- [ ] DAGExecutor leaf execution: if leaf.domain has a registered tool, use the tool instead of LLM
- [ ] Tool results wrapped in ExecutionResult for consistent pipeline
- [ ] Decomposer aware of available tools: tool descriptions injected into decomposition prompt
- [ ] Test: leaf with domain=FILE routes to FileTool
- [ ] Test: leaf with domain=WEB routes to WebTool
- [ ] Test: leaf with domain=SYSTEM still routes to LLM (no tool override)

### R009: Real-World Use Case Tests
Prove the system works on actual tasks, not just knowledge questions.
**Acceptance Criteria:**
- [ ] `benchmarks/real_tasks.json` with 10 real-world tasks:
  - "List all Python files in the graphbot project and count lines of code"
  - "Search the web for 'TU Eindhoven computer science' and summarize the top 3 results"
  - "Read CLAUDE.md and list all architecture decision records"
  - "Find all TODO comments in the codebase"
  - "Search for 'best graph databases 2026' and compare the top 3"
  - "Create a summary of all test files in the project"
  - "What are the current free LLM API options? Search and compare"
  - "Read pyproject.toml and list all dependencies with their versions"
  - "Search for Eindhoven tech companies and list 5 with their websites"
  - "Analyze the git log and summarize recent changes"
- [ ] `scripts/run_real_tasks.py` executes all 10 with real tools + real models
- [ ] Each task: success/fail, nodes, tokens, latency, cost, output quality
- [ ] At least 8/10 tasks complete successfully
- [ ] Results documented in benchmarks/REAL_TASKS_RESULTS.md

### R010: Enhanced UI -- Non-Technical Narrative
Make the visualization tell a story for non-technical viewers.
**Acceptance Criteria:**
- [ ] Status bar shows plain-English narration: "Breaking task into 3 parts..." -> "Searching the web..." -> "Reading files..." -> "Combining results..."
- [ ] Each node shows a human-readable action label, not technical description
- [ ] Progress animation: pulsing glow on running nodes, checkmark animation on completion
- [ ] Output panel shows results incrementally as nodes complete (not all at once)
- [ ] Timing breakdown visible: "Decomposition: 1.2s | Execution: 3.4s | Total: 4.6s"
- [ ] Cost shown prominently: "$0.0003 total" with comparison "vs ~$0.05 on GPT-4"

## Future Considerations (NOT Phase 9)
- MCP server registry for extensible tools
- Code execution sandbox (Docker/subprocess isolation)
- Multi-file editing transactions (atomic commits)
- Browser automation for interactive tasks (login, form filling)
- Channel integration (Telegram, Discord, Slack)
