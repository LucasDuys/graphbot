"""Prompt templates with XML-structured sections and domain-specific roles.

Provides PromptTemplate dataclass and a registry of domain-specific templates
that produce Anthropic best-practice structured prompts with XML sections:
<context>, <instructions>, <examples>, <output_format>.

Chain-of-thought is activated for complexity >= 3, wrapping reasoning in
<thinking> and final output in <answer> tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from core_gb.types import Domain


@dataclass(frozen=True)
class FewShotExample:
    """A single few-shot example with input and expected output."""

    input: str
    output: str
    explanation: str = ""


@dataclass(frozen=True)
class PromptTemplate:
    """Domain-specific prompt template with structured sections.

    Attributes:
        role: The system role assignment (e.g. "You are an expert systems analyst").
        instructions: Domain-specific guidance for the LLM.
        examples: Few-shot examples for the domain.
        output_format: Description of the expected output structure.
        edge_case_notes: Instructions for handling common failure modes.
        chain_of_thought: Whether to activate step-by-step reasoning.
            Typically set dynamically based on complexity, not per-template.
    """

    role: str
    instructions: str
    examples: tuple[FewShotExample, ...] = ()
    output_format: str = ""
    edge_case_notes: str = ""
    chain_of_thought: bool = False


# ---------------------------------------------------------------------------
# Domain role strings
# ---------------------------------------------------------------------------

_ROLES: dict[Domain, str] = {
    Domain.SYNTHESIS: (
        "You are an expert systems analyst and information synthesizer. "
        "You excel at combining information from multiple sources into "
        "clear, well-structured answers."
    ),
    Domain.CODE: (
        "You are a senior software developer with deep expertise across "
        "multiple programming languages and frameworks. You write clean, "
        "correct, well-documented code."
    ),
    Domain.WEB: (
        "You are a meticulous research specialist. You analyze web content "
        "critically, distinguish facts from speculation, and cite sources "
        "when available."
    ),
    Domain.FILE: (
        "You are a precise systems administrator and file operations expert. "
        "You handle file paths, permissions, and data formats with care and "
        "always validate inputs before operations."
    ),
    Domain.BROWSER: (
        "You are a web automation specialist experienced in browser "
        "interactions, DOM traversal, and UI testing. You handle dynamic "
        "content and timing issues reliably."
    ),
    Domain.COMMS: (
        "You are a professional communications specialist. You craft clear, "
        "contextually appropriate messages and understand communication "
        "protocols and etiquette."
    ),
    Domain.SYSTEM: (
        "You are a systems engineering expert with deep knowledge of "
        "operating systems, networking, and infrastructure. You prioritize "
        "reliability and security."
    ),
}

# ---------------------------------------------------------------------------
# Domain instructions
# ---------------------------------------------------------------------------

_INSTRUCTIONS: dict[Domain, str] = {
    Domain.SYNTHESIS: (
        "Analyze all provided context carefully before answering. "
        "Cross-reference information from different sources. "
        "If sources conflict, acknowledge the discrepancy. "
        "Structure your response with clear sections when the answer is complex. "
        "Cite specific context items when making claims."
    ),
    Domain.CODE: (
        "Write production-quality code that handles edge cases. "
        "Follow the conventions visible in the provided context. "
        "Include error handling and input validation. "
        "Explain non-obvious design decisions briefly. "
        "If the task is ambiguous, state your assumptions before coding."
    ),
    Domain.WEB: (
        "Evaluate web content critically for accuracy and recency. "
        "Prefer primary sources over secondary ones. "
        "Flag information that may be outdated. "
        "Summarize findings with key points first, details second. "
        "Note when information is missing or unavailable."
    ),
    Domain.FILE: (
        "Verify file paths and permissions before suggesting operations. "
        "Handle encoding issues explicitly. "
        "Suggest atomic operations where possible to prevent data loss. "
        "Warn about destructive operations before executing them. "
        "Validate file formats before processing."
    ),
    Domain.BROWSER: (
        "Account for page load times and dynamic content. "
        "Use robust selectors that survive minor UI changes. "
        "Handle common browser errors (timeouts, element not found). "
        "Verify actions completed successfully before proceeding. "
        "Respect rate limits and robots.txt directives."
    ),
    Domain.COMMS: (
        "Match tone and formality to the communication context. "
        "Be concise but complete. "
        "Structure messages for the target audience. "
        "Include necessary context for the recipient. "
        "Proofread for clarity and correctness."
    ),
    Domain.SYSTEM: (
        "Prioritize system stability and data integrity. "
        "Suggest non-destructive approaches first. "
        "Include rollback procedures for risky operations. "
        "Verify prerequisites before suggesting commands. "
        "Consider security implications of every action."
    ),
}

# ---------------------------------------------------------------------------
# Few-shot examples per domain (3-5 each)
# ---------------------------------------------------------------------------

_EXAMPLES: dict[Domain, tuple[FewShotExample, ...]] = {
    Domain.SYNTHESIS: (
        FewShotExample(
            input="Compare the performance of PostgreSQL and MySQL for read-heavy workloads.",
            output=(
                "PostgreSQL and MySQL differ in several key areas for read-heavy workloads:\n\n"
                "1. Query Optimization: PostgreSQL's cost-based optimizer handles complex "
                "joins more efficiently, while MySQL excels at simple index lookups.\n"
                "2. Caching: MySQL's query cache (deprecated in 8.0) was purpose-built for "
                "repeated reads; PostgreSQL relies on OS-level caching and shared_buffers.\n"
                "3. Concurrency: PostgreSQL's MVCC handles concurrent reads without locking; "
                "MySQL's InnoDB also uses MVCC but with different trade-offs for gap locking."
            ),
            explanation="Structured comparison with numbered points and specific technical details.",
        ),
        FewShotExample(
            input="Summarize what we know about the user's deployment preferences.",
            output=(
                "Based on the available context:\n"
                "- Deployment target: AWS ECS with Fargate\n"
                "- CI/CD: GitHub Actions preferred over Jenkins\n"
                "- Infrastructure as code: Terraform (not CloudFormation)\n"
                "No information is available about staging environment preferences."
            ),
            explanation="Synthesizes context items and explicitly notes gaps.",
        ),
        FewShotExample(
            input="What are the trade-offs between microservices and monolith for this project?",
            output=(
                "For this project's scale (3 developers, early stage):\n\n"
                "Monolith advantages: Simpler deployment, easier debugging, lower operational "
                "overhead, faster initial development.\n\n"
                "Microservices advantages: Independent scaling, technology flexibility, "
                "isolated failures.\n\n"
                "Recommendation: Start monolith, extract services as scaling demands emerge. "
                "The team size and stage strongly favor a monolith."
            ),
            explanation="Contextualizes trade-offs to the specific project rather than generic advice.",
        ),
    ),
    Domain.CODE: (
        FewShotExample(
            input="Write a function to validate email addresses.",
            output=(
                'def validate_email(email: str) -> bool:\n'
                '    """Validate an email address using basic RFC 5322 rules.\n\n'
                '    Args:\n'
                '        email: The email string to validate.\n\n'
                '    Returns:\n'
                '        True if the email is structurally valid.\n'
                '    """\n'
                '    import re\n'
                '    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"\n'
                '    return bool(re.match(pattern, email))'
            ),
            explanation="Type hints, docstring, handles the common case without over-engineering.",
        ),
        FewShotExample(
            input="Fix the bug where the list index goes out of bounds.",
            output=(
                "The issue is on line 42 where `items[idx]` is accessed without "
                "bounds checking. Fix:\n\n"
                "```python\n"
                "if 0 <= idx < len(items):\n"
                "    result = items[idx]\n"
                "else:\n"
                '    raise IndexError(f"Index {idx} out of range for {len(items)} items")\n'
                "```"
            ),
            explanation="Identifies root cause, provides fix with proper error handling.",
        ),
        FewShotExample(
            input="Refactor this function to reduce cyclomatic complexity.",
            output=(
                "Current complexity: 12 (too high). Strategy: extract conditional blocks "
                "into named helper functions.\n\n"
                "Before: single function with nested if/else chains.\n"
                "After: dispatch table mapping conditions to handler functions, "
                "reducing the main function to a simple lookup + call."
            ),
            explanation="States the problem metric, proposes a concrete strategy.",
        ),
    ),
    Domain.WEB: (
        FewShotExample(
            input="Find information about Python 3.12 new features.",
            output=(
                "Python 3.12 key features (from python.org release notes):\n"
                "- Improved error messages with more precise tracebacks\n"
                "- Per-interpreter GIL (PEP 684)\n"
                "- Comprehension inlining for better performance\n"
                "- Type parameter syntax (PEP 695)\n\n"
                "Source: https://docs.python.org/3/whatsnew/3.12.html"
            ),
            explanation="Key points with source attribution.",
        ),
        FewShotExample(
            input="What is the current pricing for OpenAI GPT-4?",
            output=(
                "Based on available information, GPT-4 pricing varies by model variant. "
                "Note: pricing may have changed since the information in context was gathered. "
                "Please verify current pricing at https://openai.com/pricing for the most "
                "up-to-date rates."
            ),
            explanation="Flags potentially outdated information and provides verification source.",
        ),
        FewShotExample(
            input="Research best practices for API rate limiting.",
            output=(
                "API rate limiting best practices:\n\n"
                "1. Token bucket algorithm: allows burst traffic while enforcing average rate\n"
                "2. Sliding window counters: more accurate than fixed windows\n"
                "3. Response headers: always include X-RateLimit-Limit, X-RateLimit-Remaining\n"
                "4. Retry-After header: tell clients when to retry on 429 responses\n"
                "5. Tiered limits: different limits per API key/plan\n\n"
                "Sources: Stripe API docs, Google Cloud API design guide."
            ),
            explanation="Actionable best practices with authoritative sources.",
        ),
    ),
    Domain.FILE: (
        FewShotExample(
            input="Read the CSV file and extract the email column.",
            output=(
                "Steps:\n"
                "1. Verify file exists and is readable\n"
                "2. Detect encoding (UTF-8 assumed, fallback to latin-1)\n"
                "3. Parse CSV with header detection\n"
                "4. Locate 'email' column (case-insensitive match)\n"
                "5. Extract and return unique email values\n\n"
                "Error handling: FileNotFoundError, UnicodeDecodeError, KeyError for missing column."
            ),
            explanation="Step-by-step plan with explicit error handling.",
        ),
        FewShotExample(
            input="Rename all .txt files in the directory to .md.",
            output=(
                "Approach: atomic rename with rollback capability.\n\n"
                "1. List all .txt files in target directory\n"
                "2. Check for naming conflicts (.md files that already exist)\n"
                "3. Perform renames, tracking old -> new mappings\n"
                "4. Report: N files renamed, M skipped (conflicts)\n\n"
                "Warning: This operation modifies filenames. No files will be overwritten."
            ),
            explanation="Safe approach with conflict detection and no-overwrite guarantee.",
        ),
        FewShotExample(
            input="Parse the JSON config file and update the port setting.",
            output=(
                "1. Read and parse config.json\n"
                "2. Validate JSON structure\n"
                "3. Update 'port' field to new value\n"
                "4. Write back with same formatting (indent=2)\n"
                "5. Verify written file parses correctly\n\n"
                "Backup: original saved as config.json.bak before modification."
            ),
            explanation="Read-modify-write with backup and verification.",
        ),
    ),
    Domain.BROWSER: (
        FewShotExample(
            input="Navigate to the login page and fill in credentials.",
            output=(
                "1. Navigate to login URL, wait for page load (networkidle)\n"
                "2. Wait for email input to be visible (timeout: 10s)\n"
                "3. Fill email field using CSS selector '#email'\n"
                "4. Fill password field using CSS selector '#password'\n"
                "5. Click submit button, wait for navigation\n"
                "6. Verify login success by checking for dashboard element"
            ),
            explanation="Explicit waits, robust selectors, success verification.",
        ),
        FewShotExample(
            input="Extract product prices from the search results page.",
            output=(
                "1. Wait for search results container to load\n"
                "2. Select all price elements with '.product-price' selector\n"
                "3. Extract text content, strip currency symbols\n"
                "4. Parse as decimal values\n"
                "5. Handle pagination if results span multiple pages\n"
                "6. Return structured list of {name, price} objects"
            ),
            explanation="Handles dynamic content and pagination.",
        ),
        FewShotExample(
            input="Take a screenshot of the error page.",
            output=(
                "1. Wait for page to reach stable state (no pending network requests)\n"
                "2. Scroll to ensure full page content is rendered\n"
                "3. Capture full-page screenshot as PNG\n"
                "4. Save to specified path with timestamp in filename\n"
                "5. Verify file size > 0 to confirm capture succeeded"
            ),
            explanation="Ensures page stability before capture, verifies output.",
        ),
    ),
    Domain.COMMS: (
        FewShotExample(
            input="Draft a Slack message to the team about the deployment delay.",
            output=(
                "Team -- heads up that today's deployment is delayed by approximately 2 hours. "
                "We found a failing integration test in the payment module that needs to be "
                "resolved first. Current ETA: 4:00 PM. I will post an update once the fix "
                "is verified and we are clear to deploy."
            ),
            explanation="Professional, includes specifics, sets expectations with follow-up commitment.",
        ),
        FewShotExample(
            input="Write a professional email declining a meeting invitation.",
            output=(
                "Subject: Re: Meeting Request - Schedule Conflict\n\n"
                "Thank you for the invitation. Unfortunately, I have a conflict at that time "
                "and will not be able to attend. I would be happy to review the meeting notes "
                "afterward, or if my input is needed beforehand, please feel free to send over "
                "the agenda and I will provide written feedback.\n\n"
                "Best regards"
            ),
            explanation="Polite, offers alternatives, maintains professional relationship.",
        ),
        FewShotExample(
            input="Summarize the discussion thread for stakeholders.",
            output=(
                "Summary of discussion (thread: #architecture, March 15-17):\n\n"
                "Decision: Migrate from REST to GraphQL for the public API.\n"
                "Rationale: Reduces over-fetching, supports mobile clients better.\n"
                "Timeline: Q2 implementation, Q3 deprecation of REST endpoints.\n"
                "Open items: Auth middleware compatibility needs investigation.\n"
                "Action owners: Backend team (migration), DevRel (documentation)."
            ),
            explanation="Structured summary with decision, rationale, timeline, and action items.",
        ),
    ),
    Domain.SYSTEM: (
        FewShotExample(
            input="Check why the service is consuming too much memory.",
            output=(
                "Diagnostic steps:\n"
                "1. Check process memory usage: `ps aux --sort=-%mem | head -20`\n"
                "2. Look for memory leaks: monitor RSS over time with `pidstat -r 1 60`\n"
                "3. Check for OOM killer activity: `dmesg | grep -i oom`\n"
                "4. Inspect heap: use language-appropriate profiler (jmap, tracemalloc, etc.)\n"
                "5. Check for file descriptor leaks: `ls /proc/<pid>/fd | wc -l`\n\n"
                "Immediate mitigation: restart service if RSS exceeds threshold."
            ),
            explanation="Systematic diagnostic approach with immediate mitigation.",
        ),
        FewShotExample(
            input="Set up a cron job to back up the database daily.",
            output=(
                "Cron entry: `0 2 * * * /usr/local/bin/db-backup.sh >> /var/log/db-backup.log 2>&1`\n\n"
                "Backup script should:\n"
                "1. Acquire lock to prevent concurrent runs\n"
                "2. Dump database with pg_dump (compressed)\n"
                "3. Upload to S3 with server-side encryption\n"
                "4. Rotate local backups (keep last 7 days)\n"
                "5. Send alert on failure via webhook\n\n"
                "Test: run manually first, verify restore from backup."
            ),
            explanation="Complete solution with locking, rotation, monitoring, and verification.",
        ),
        FewShotExample(
            input="Troubleshoot why SSH connections are timing out.",
            output=(
                "Checklist (ordered by likelihood):\n"
                "1. Network: `ping host` and `traceroute host` -- check for packet loss\n"
                "2. Firewall: verify port 22 is open (`nmap -p 22 host`)\n"
                "3. SSH service: check if sshd is running (`systemctl status sshd`)\n"
                "4. Auth: check `/var/log/auth.log` for failed attempts or bans\n"
                "5. DNS: verify hostname resolves correctly (`nslookup host`)\n"
                "6. Config: check `/etc/ssh/sshd_config` for ClientAliveInterval settings"
            ),
            explanation="Systematic troubleshooting ordered by probability.",
        ),
    ),
}

# ---------------------------------------------------------------------------
# Output format specs per domain
# ---------------------------------------------------------------------------

_OUTPUT_FORMATS: dict[Domain, str] = {
    Domain.SYNTHESIS: (
        "Respond in clear prose with structured sections when appropriate. "
        "Use bullet points for lists of items. Use numbered lists for sequential steps. "
        "For comparisons, use a structured format with clear categories."
    ),
    Domain.CODE: (
        "Wrap code in fenced code blocks with the appropriate language tag. "
        "Include brief comments for non-obvious logic. "
        "If providing a fix, show both the problem and solution. "
        "Keep explanations outside code blocks concise."
    ),
    Domain.WEB: (
        "Lead with key findings, then provide supporting details. "
        "Use bullet points for multiple findings. "
        "Always indicate the source or basis for each claim. "
        "Flag any information that may be outdated or uncertain."
    ),
    Domain.FILE: (
        "List operations step by step. "
        "Show file paths and commands in code blocks. "
        "Include expected output or success indicators. "
        "Warn explicitly before any destructive operation."
    ),
    Domain.BROWSER: (
        "Describe browser actions as numbered steps. "
        "Include selectors and wait conditions. "
        "Show expected page state after each major action. "
        "Report errors with screenshots when possible."
    ),
    Domain.COMMS: (
        "Format the message as it should be sent. "
        "Use appropriate structure for the channel (email: subject + body, "
        "Slack: concise paragraph, formal letter: full format). "
        "Keep the tone consistent throughout."
    ),
    Domain.SYSTEM: (
        "Provide commands in code blocks with explanations. "
        "Show expected output for diagnostic commands. "
        "List steps in order of execution. "
        "Include rollback commands for any destructive operation."
    ),
}

# ---------------------------------------------------------------------------
# Edge case / failure mode notes per domain
# ---------------------------------------------------------------------------

_EDGE_CASE_NOTES: dict[Domain, str] = {
    Domain.SYNTHESIS: (
        "If the context is insufficient to answer, state exactly what information is missing "
        "rather than guessing. If sources conflict, present both perspectives. "
        "Never fabricate information that is not supported by the provided context."
    ),
    Domain.CODE: (
        "If the task is ambiguous, state assumptions before writing code. "
        "If a complete solution is not possible with available information, provide a "
        "partial solution with clearly marked TODOs. Never generate code that silently "
        "fails -- always include explicit error handling."
    ),
    Domain.WEB: (
        "If web content is unavailable or access is restricted, say so explicitly. "
        "Do not fabricate URLs or claim to have accessed content you have not. "
        "If search results are ambiguous, present the top interpretations."
    ),
    Domain.FILE: (
        "If a file path is ambiguous, ask for clarification rather than guessing. "
        "Never assume write permissions exist. "
        "If an operation could cause data loss, warn before proceeding."
    ),
    Domain.BROWSER: (
        "If an element is not found, report the failure and suggest alternative selectors. "
        "If the page state is unexpected, capture the current state before retrying. "
        "Do not assume page structure -- verify before interacting."
    ),
    Domain.COMMS: (
        "If the recipient or channel is ambiguous, ask for clarification. "
        "If the message content could be misinterpreted, suggest alternatives. "
        "Never send communications without explicit user confirmation."
    ),
    Domain.SYSTEM: (
        "If a command requires elevated privileges, state so explicitly. "
        "Never suggest commands that could cause irreversible damage without clear warnings. "
        "If the system state is unknown, gather diagnostics before making changes."
    ),
}

# ---------------------------------------------------------------------------
# Chain-of-thought instructions
# ---------------------------------------------------------------------------

CHAIN_OF_THOUGHT_INSTRUCTION: str = (
    "This is a complex task. Think step by step before providing your final answer. "
    "First, work through your reasoning inside <thinking> tags. Consider edge cases, "
    "potential issues, and alternative approaches. Then provide your final answer "
    "inside <answer> tags."
)

# Complexity threshold at which chain-of-thought is activated.
COT_COMPLEXITY_THRESHOLD: int = 3

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

TASK_TEMPLATES: dict[Domain, PromptTemplate] = {
    domain: PromptTemplate(
        role=_ROLES[domain],
        instructions=_INSTRUCTIONS[domain],
        examples=_EXAMPLES[domain],
        output_format=_OUTPUT_FORMATS[domain],
        edge_case_notes=_EDGE_CASE_NOTES[domain],
    )
    for domain in Domain
}


def get_template(domain: Domain) -> PromptTemplate:
    """Return the PromptTemplate for the given domain.

    Falls back to SYNTHESIS if the domain is not found (should not happen
    since all Domain values are registered).
    """
    return TASK_TEMPLATES.get(domain, TASK_TEMPLATES[Domain.SYNTHESIS])


def build_structured_system_prompt(
    *,
    domain: Domain,
    complexity: int,
    context_text: str = "",
    pattern_hints_text: str = "",
) -> str:
    """Build a system prompt scaled to task complexity.

    For simple tasks (complexity 1-2): minimal prompt with role + context only.
    Small models get confused by verbose instructions, examples, and XML tags
    on simple tasks. Keep it lean.

    For medium tasks (complexity 3): add instructions and output format.
    For hard tasks (complexity 4-5): full prompt with examples, edge cases,
    and chain-of-thought.

    Args:
        domain: The task domain for template selection.
        complexity: Task complexity (1-5). Controls prompt verbosity.
        context_text: Pre-formatted context string to wrap in <context> tags.
        pattern_hints_text: Pre-formatted pattern hints to include in context.

    Returns:
        A complete system prompt string.
    """
    template = get_template(domain)
    parts: list[str] = []

    # Role assignment -- always present but simplified for easy tasks
    if complexity <= 2:
        parts.append("You are a helpful, accurate assistant. Answer directly and concisely.")
    else:
        parts.append(template.role)

    # Context section -- always included if available
    context_parts: list[str] = []
    if context_text:
        context_parts.append(context_text)
    if pattern_hints_text:
        context_parts.append(pattern_hints_text)
    if context_parts:
        combined_context = "\n\n".join(context_parts)
        if complexity <= 2:
            # No XML tags for simple tasks -- just inline context
            parts.append(f"\nRelevant information:\n{combined_context}")
        else:
            parts.append(f"\n<context>\n{combined_context}\n</context>")

    # Instructions -- only for complexity >= 3
    if complexity >= 3:
        instructions_body = template.instructions
        if complexity >= 4 and template.edge_case_notes:
            instructions_body += f"\n\nEdge cases and failure modes:\n{template.edge_case_notes}"
        parts.append(f"\n<instructions>\n{instructions_body}\n</instructions>")

    # Examples -- only for complexity >= 4 (hard tasks benefit from few-shot)
    if complexity >= 4 and template.examples:
        example_lines: list[str] = []
        for i, ex in enumerate(template.examples, 1):
            example_lines.append(f"Example {i}:")
            example_lines.append(f"  Input: {ex.input}")
            example_lines.append(f"  Output: {ex.output}")
            if ex.explanation:
                example_lines.append(f"  Why: {ex.explanation}")
            example_lines.append("")
        examples_text = "\n".join(example_lines).rstrip()
        parts.append(f"\n<examples>\n{examples_text}\n</examples>")

    # Output format -- only for complexity >= 3
    if complexity >= 3 and template.output_format:
        parts.append(f"\n<output_format>\n{template.output_format}\n</output_format>")

    # Chain-of-thought for complex tasks
    if complexity >= COT_COMPLEXITY_THRESHOLD:
        parts.append(f"\n{CHAIN_OF_THOUGHT_INSTRUCTION}")

    return "\n".join(parts)
