"""Code editing agent -- read, analyze, edit, test loop."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from core_gb.types import CompletionResult, ExecutionResult, TaskNode

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class CodeEditAgent:
    """Multi-turn agent that reads, analyzes, edits, and tests code.

    Implements a retry loop:
    1. Read file to understand context
    2. Ask LLM to generate search-and-replace edit
    3. Apply edit via FileTool.edit()
    4. Run tests to verify
    5. If tests fail and retries remain, inject error context and retry from step 2
    """

    def __init__(self, file_tool: Any, shell_tool: Any, router: Any) -> None:
        self._file = file_tool
        self._shell = shell_tool
        self._router = router

    async def edit(
        self,
        instruction: str,
        file_path: str,
        test_command: str | None = None,
    ) -> ExecutionResult:
        """Execute a code edit with test verification loop."""
        start = time.perf_counter()
        root_id = str(uuid.uuid4())

        # Step 1: Read file
        read_result = self._file.read(file_path)
        if not read_result["success"]:
            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                root_id=root_id,
                output="",
                success=False,
                total_nodes=1,
                total_tokens=0,
                total_latency_ms=elapsed,
                total_cost=0.0,
                errors=(f"Cannot read file: {read_result.get('error', '')}",),
            )

        file_content = read_result["content"]
        last_error: str | None = None
        total_tokens = 0
        total_cost = 0.0

        for attempt in range(MAX_RETRIES):
            # Step 2: Ask LLM for edit
            messages = self._build_edit_prompt(instruction, file_path, file_content, last_error)

            try:
                route_node = TaskNode(id="edit", description=instruction, complexity=3)
                completion: CompletionResult = await self._router.route(
                    route_node,
                    messages,
                    response_format={"type": "json_object"},
                )
                total_tokens += completion.tokens_in + completion.tokens_out
                total_cost += completion.cost
            except Exception as exc:
                last_error = str(exc)
                continue

            # Step 3: Parse edit from response
            edit = self._parse_edit(completion.content)
            if edit is None:
                last_error = f"Could not parse edit from LLM response: {completion.content[:200]}"
                continue

            # Step 4: Apply edit
            edit_result = self._file.edit(file_path, edit["old_text"], edit["new_text"])
            if not edit_result["success"]:
                last_error = f"Edit failed: {edit_result.get('error', '')}"
                continue

            # Step 5: Run tests (if provided)
            if test_command:
                test_result = await self._shell.run(test_command, timeout=60)
                if not test_result["success"]:
                    last_error = (
                        f"Tests failed (attempt {attempt + 1}):\n"
                        f"{test_result.get('stdout', '')[:500]}\n"
                        f"{test_result.get('stderr', '')[:500]}"
                    )
                    # Re-read file for next attempt (it was modified)
                    re_read = self._file.read(file_path)
                    if re_read["success"]:
                        file_content = re_read["content"]
                    continue

            # Success
            elapsed = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                root_id=root_id,
                output=f"Edit applied to {file_path}: {edit['old_text'][:50]}... -> {edit['new_text'][:50]}...",
                success=True,
                total_nodes=1,
                total_tokens=total_tokens,
                total_latency_ms=elapsed,
                total_cost=total_cost,
                model_used="code_agent",
            )

        # All retries exhausted
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            root_id=root_id,
            output="",
            success=False,
            total_nodes=1,
            total_tokens=total_tokens,
            total_latency_ms=elapsed,
            total_cost=total_cost,
            errors=(f"Edit failed after {MAX_RETRIES} attempts. Last error: {last_error}",),
        )

    def _build_edit_prompt(
        self,
        instruction: str,
        file_path: str,
        content: str,
        previous_error: str | None,
    ) -> list[dict[str, str]]:
        """Build prompt for LLM to generate a search-and-replace edit."""
        system = (
            f'You are a code editor. Generate a precise search-and-replace edit.\n\n'
            f'<file path="{file_path}">\n{content[:3000]}\n</file>\n\n'
            f'Return a JSON object with exactly two keys:\n'
            f'- "old_text": the EXACT text to find in the file (must match character-for-character)\n'
            f'- "new_text": the replacement text\n\n'
            f'Rules:\n'
            f'- old_text must exist EXACTLY in the file (copy it precisely)\n'
            f'- Include enough context in old_text to uniquely identify the location\n'
            f'- Keep edits small and focused\n'
            f'- Preserve indentation and formatting'
        )

        user = instruction
        if previous_error:
            user = f"{instruction}\n\nPrevious attempt failed:\n{previous_error}\n\nPlease fix the issue."

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _parse_edit(content: str) -> dict[str, str] | None:
        """Parse old_text/new_text from LLM JSON response."""
        try:
            data = json.loads(content)
            if "old_text" in data and "new_text" in data:
                return {"old_text": str(data["old_text"]), "new_text": str(data["new_text"])}
        except Exception:
            pass

        # Try json_repair as fallback
        try:
            import json_repair
            data = json_repair.loads(content)
            if isinstance(data, dict) and "old_text" in data and "new_text" in data:
                return {"old_text": str(data["old_text"]), "new_text": str(data["new_text"])}
        except Exception:
            pass

        return None
