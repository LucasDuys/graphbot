---
domain: phase10a-prompt-quality
status: approved
created: 2026-03-22
complexity: medium
linked_repos: []
---

# Phase 10A: Prompt Quality -- Fix Tool Routing

## Overview

The decomposer produces task trees but assigns generic domains (system/synthesis)
instead of tool-specific ones (file/web/code). Research Anthropic's and others'
best practices for structured output from LLMs, then fix the prompt so tools get used.

METHODOLOGY: Deep research first (Anthropic docs, structured output papers, tool-use
prompting guides), document findings, THEN implement from documentation.

## Requirements

### R001: Research -- Structured Output + Tool Routing Prompts
Deep research before any prompt changes.
**Acceptance Criteria:**
- [ ] docs/research/prompt-engineering.md written with findings from research agents
- [ ] Topics: Anthropic's official tool-use prompting guide, structured JSON output best practices, domain classification in prompts, few-shot example design for tool routing
- [ ] Specific: how Claude/Llama/Qwen handle response_format=json, what makes JSON output reliable
- [ ] Specific: how to make LLMs choose the RIGHT tool/domain (tool descriptions, enum constraints, negative examples)
- [ ] Specific: Anthropic's "thinking" approach -- chain-of-thought before structured output
- [ ] Actionable recommendations documented

### R002: Domain Override Fallback
If the decomposer assigns wrong domain, detect tool keywords in leaf descriptions and override.
**Acceptance Criteria:**
- [ ] `_infer_domain_from_description(description: str) -> Domain` in decomposer.py or orchestrator
- [ ] Checks for file-related words (read, write, list, .py, .md, directory, etc.) -> Domain.FILE
- [ ] Checks for web-related words (search, fetch, url, http, browse, web, online) -> Domain.WEB
- [ ] Checks for shell-related words (run, execute, git, pytest, command, pip) -> Domain.CODE
- [ ] Applied as post-processing after decomposition: if leaf is "system"/"synthesis" but description has tool keywords, override domain
- [ ] Test: "Read README.md" gets domain FILE even if decomposer said SYSTEM
- [ ] Test: "Search the web for X" gets domain WEB
- [ ] Test: "Run git log" gets domain CODE
- [ ] Test: "What is 2+2" stays SYSTEM (no override)

### R003: Improved Decomposition Prompt
Update prompt based on research findings.
**Acceptance Criteria:**
- [ ] Prompt includes explicit tool descriptions: "domain 'file' has tools: file_read, file_list, file_search. Use this domain when the task involves reading, listing, or searching files."
- [ ] Prompt includes negative examples: "Do NOT use domain 'system' for file reading tasks. Use 'file' instead."
- [ ] Few-shot examples updated to cover all tool domains (file, web, code)
- [ ] Test: decompose "List Python files in the project" -> at least one leaf has domain FILE
- [ ] Test: decompose "Search web for X" -> at least one leaf has domain WEB
- [ ] Benchmark: re-run 10 real tasks, tool usage rate > 50% (up from 0%)

### R004: SimpleExecutor Tool Awareness
When SimpleExecutor receives a task that could use a tool, check tool registry first.
**Acceptance Criteria:**
- [ ] SimpleExecutor accepts optional ToolRegistry parameter
- [ ] Before calling LLM, check if task description suggests a tool (reuse _infer_domain logic)
- [ ] If tool domain detected, use tool instead of LLM (0 tokens, 0 cost)
- [ ] This catches tool tasks that bypass decomposition (simple path)
- [ ] Test: SimpleExecutor with tool registry, task "list files in ." -> uses FileTool, not LLM
