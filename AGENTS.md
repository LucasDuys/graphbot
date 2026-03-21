# AGENTS.md -- Sub-Agent Definitions

Each agent is a prompt template + tool binding + model selection rule dispatched by the router when a TaskNode matches its domain.

## File Agent
- **Role**: File system operations (search, read, write, move, organize)
- **Model**: Llama 3.1 8B (Groq free). Escalate to 70B for content generation.
- **Tools**: read_file, write_file, search_files, move_file, list_dir, file_info
- **Output**: `{"action": "...", "result": "...", "success": true/false}`

## Web Agent
- **Role**: Browse, search, scrape, API calls, structured extraction
- **Model**: Llama 3.1 8B. Qwen 2.5 7B for complex extraction.
- **Tools**: web_search, web_fetch, semantic_snapshot
- **Output**: `{"query": "...", "findings": [...], "sources": [...], "success": true/false}`

## Code Agent
- **Role**: Generate, debug, explain, refactor code
- **Model**: Qwen 2.5 Coder 7B. Llama 3.1 70B for complex refactoring.
- **Tools**: read_file, write_file, run_command, search_files
- **Output**: `{"code": "...", "language": "...", "filepath": "...", "action": "create|modify|explain", "success": true/false}`

## Communication Agent
- **Role**: Draft messages, summarize threads, compose emails
- **Model**: Llama 3.1 8B. 70B for sensitive/high-stakes communication.
- **Tools**: send_message, read_messages, search_messages
- **Output**: `{"message": "...", "platform": "...", "recipient": "...", "tone": "...", "success": true/false}`

## System Agent
- **Role**: Shell commands, installations, cron, system monitoring
- **Model**: Llama 3.1 8B. 70B for complex scripting.
- **Tools**: run_command (sandboxed), cron_add, cron_remove, process_list, system_info
- **Output**: `{"command": "...", "stdout": "...", "stderr": "...", "exit_code": 0, "success": true/false}`

## Synthesis Agent
- **Role**: Cross-domain reasoning, comparison, recommendation. Only agent that routinely uses larger models.
- **Model**: Llama 3.1 70B (Groq free). DeepSeek R1 for very complex analysis.
- **Tools**: All read-only tools from other agents.
- **Output**: `{"synthesis": "...", "recommendation": "...", "uncertainties": [...], "confidence": 0.0-1.0, "success": true/false}`
