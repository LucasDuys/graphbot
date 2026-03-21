# Research Summary -- Phase 9 Tool System

Research completed 2026-03-21 by 4 parallel agents. Key findings below.

## Web Scraping (docs/research/web-scraping.md)

**Recommended 3-tier approach:**
1. readability-lxml (already in deps, F1=0.922, fast, no JS) -- default for static pages
2. Jina Reader API (JS support, automatic cleanup) -- fallback for JS-heavy
3. Playwright MCP -- interactive workflows only (too expensive for scraping: 3.8K-50K tokens/page)

**Token reduction:** Raw HTML 223K tokens -> cleaned text 500-1500 tokens (20-50x reduction)
**Search:** DuckDuckGo via ddgs (already in deps), rate limit 1 req/sec
**Key finding:** Semantic markdown achieves 35% accuracy improvement in RAG with 40% token reduction vs HTML

## File Operations (docs/research/file-operations.md)

**Finding: Nanobot's file tools are production-grade.** Already have:
- ReadFileTool: pagination, 128K char budget, image support
- WriteFileTool: auto mkdir, UTF-8
- EditFileTool: search-and-replace with fuzzy fallback + diff reporting
- ListDirTool: recursive, ignore boilerplate

**Path sandboxing:** Already implemented via resolve() + relative_to()
**Recommended:** Build tools_gb/ wrappers around Nanobot's tools rather than from scratch. Add: file_search (grep), encoding detection, atomic writes.

## Shell Execution (docs/research/shell-execution.md)

**Finding: Nanobot's ExecTool is solid.** Already has:
- Async subprocess, timeout management (default 60s, max 600s)
- Output truncation (head+tail, 10K char limit)
- Command blocklist (rm -rf, format, shutdown, fork bombs)
- SSRF protection, path traversal detection
- Environment filtering

**Recommended:** Wrap ExecTool for tools_gb/ interface. Add: resource limits, streaming output, better error context.
**Cross-platform:** Use pathlib everywhere, detect platform for shell selection.

## Visualization UX (docs/research/visualization-ux.md)

**Key patterns from Vercel/Linear/GitHub/ChatGPT:**
1. Progressive disclosure: summary first, expand for details
2. Plain-English narration alongside DAG: "Breaking question into 3 parts..."
3. Animation timing: 200-300ms for transitions, never >500ms
4. Accessible colors: OKLCH + shape/icon (not just color)
5. Show cost transparently: "$0.0003 (vs $0.05 on GPT-4)"

**Animation vocabulary:**
- Pulse/glow = "active/working" (2.5s cycle)
- Slide-in = "just appeared" (250ms)
- Checkmark = "success" (150ms)
- Bounce = "unblocked" (100ms)

## Implementation Impact

The research changes our implementation approach:
- **Web tools:** Use readability-lxml (already installed!) + ddgs. Jina as fallback. Skip BeautifulSoup.
- **File tools:** Wrap Nanobot's existing tools. Don't rebuild from scratch.
- **Shell tools:** Wrap Nanobot's ExecTool. Don't rebuild from scratch.
- **UI:** Add narration layer + progressive disclosure. Not just DAG nodes.
