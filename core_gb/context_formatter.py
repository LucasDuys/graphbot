"""Context formatter with activation ranking for single-call prompts.

Takes EnrichedContext from the context enrichment pipeline and formats it
into structured prompt sections ordered by activation score. Integrates
with TokenBudget to trim sections that exceed the available budget.

Returns a list of message dicts ready for the LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core_gb.context_enrichment import EnrichedContext
from core_gb.prompt_templates import build_structured_system_prompt
from core_gb.token_budget import TokenBudget
from core_gb.types import Domain, Pattern

_DEFAULT_PREAMBLE: str = "You are a helpful assistant."


@dataclass(frozen=True)
class SectionDef:
    """A formatted context section with its activation score for ranking.

    Attributes:
        name: Section identifier (entities, memories, reflections, etc.).
        text: The fully formatted section text.
        activation_tokens: Token estimate from EnrichedContext, used as
            activation score for ranking (higher = more relevant).
    """

    name: str
    text: str
    activation_tokens: int


class ContextFormatter:
    """Formats EnrichedContext into activation-ranked prompt sections.

    Sections are ordered by activation score (token weight from the
    enrichment pipeline) so the most relevant context appears first
    in the prompt. Sections that exceed the token budget are dropped.

    Args:
        token_budget: Optional TokenBudget instance for trimming.
            If None, a default budget with generous limits is used.
        system_preamble: Opening line for the system message.
            Defaults to "You are a helpful assistant."
    """

    def __init__(
        self,
        *,
        token_budget: TokenBudget | None = None,
        system_preamble: str = _DEFAULT_PREAMBLE,
        domain: Domain = Domain.SYNTHESIS,
        complexity: int = 1,
    ) -> None:
        self._budget = token_budget or TokenBudget(max_tokens=100_000)
        self._preamble = system_preamble
        self._domain = domain
        self._complexity = complexity

    def format(
        self,
        enriched: EnrichedContext,
        *,
        task: str,
        domain: Domain | None = None,
        complexity: int | None = None,
    ) -> list[dict[str, str]]:
        """Format enriched context into LLM-ready message dicts.

        Assembles sections from the enriched context, ranks them by
        activation score, trims to the token budget, and builds the
        final message list with XML-structured system prompt.

        Args:
            enriched: The enriched context from ContextEnricher.
            task: The user task / question to place as the final message.
            domain: Override domain for prompt template selection.
                Falls back to the instance default if not provided.
            complexity: Override complexity for chain-of-thought activation.
                Falls back to the instance default if not provided.

        Returns:
            A list of message dicts with "role" and "content" keys,
            ordered as: system message, conversation turns, user message.
        """
        effective_domain = domain or self._domain
        effective_complexity = complexity if complexity is not None else self._complexity

        # 1. Format all non-empty sections
        raw_sections = self._format_sections(enriched)

        # 2. Rank sections by activation score
        ranked = self._rank_sections(enriched)

        # 3. Trim to budget
        trimmed = self._trim_to_budget(ranked)

        # 4. Assemble context text from ranked, trimmed sections
        context_parts: list[str] = []
        for section_def in trimmed:
            context_parts.append(section_def.text)
        context_text = "\n\n".join(context_parts) if context_parts else ""

        # 5. Build XML-structured system message
        system_content = build_structured_system_prompt(
            domain=effective_domain,
            complexity=effective_complexity,
            context_text=context_text,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # 6. Add conversation turns as separate messages
        if enriched.conversation_turns:
            for turn in enriched.conversation_turns:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        # 7. Add the user task as the final message
        messages.append({"role": "user", "content": task})

        return messages

    @staticmethod
    def format_as_xml_document(
        index: int,
        source: str,
        content: str,
    ) -> str:
        """Format a piece of context as an XML document element.

        Args:
            index: Document index (1-based).
            source: Source identifier (e.g., "knowledge_graph", "memory").
            content: The document content.

        Returns:
            XML-formatted document string.
        """
        return (
            f'<document index="{index}">'
            f"<source>{source}</source>"
            f"<content>{content}</content>"
            f"</document>"
        )

    def _format_sections(
        self,
        enriched: EnrichedContext,
    ) -> dict[str, str]:
        """Format each non-empty section of the enriched context.

        Returns a dict mapping section name to formatted text. Empty
        sections (no data) are excluded.

        Entity and relationship context is formatted as XML document
        elements for structured retrieval.
        """
        sections: dict[str, str] = {}
        doc_index = 1

        # Entities -- formatted as XML documents
        if enriched.entities:
            docs: list[str] = []
            for entity in enriched.entities:
                etype = entity.get("type", "")
                name = entity.get("name", "")
                details = entity.get("details", "")
                content = f"{etype}: {name} -- {details}"
                docs.append(self.format_as_xml_document(
                    doc_index, "knowledge_graph", content,
                ))
                doc_index += 1
            sections["entities"] = "\n".join(docs)

        # Relationships -- formatted as XML documents
        if enriched.relationship_descriptions:
            docs = []
            for desc in enriched.relationship_descriptions:
                docs.append(self.format_as_xml_document(
                    doc_index, "knowledge_graph", desc,
                ))
                doc_index += 1
            sections["relationships"] = "\n".join(docs)

        # Community summaries -- formatted as XML documents
        if enriched.community_summaries:
            docs = []
            for summary in enriched.community_summaries:
                docs.append(self.format_as_xml_document(
                    doc_index, "knowledge_graph", summary,
                ))
                doc_index += 1
            sections["communities"] = "\n".join(docs)

        # Memories -- formatted as XML documents
        if enriched.memories:
            docs = []
            for memory in enriched.memories:
                docs.append(self.format_as_xml_document(
                    doc_index, "memory", memory,
                ))
                doc_index += 1
            sections["memories"] = "\n".join(docs)

        # Reflections
        if enriched.reflections:
            lines: list[str] = ["Previous attempts at similar tasks failed because:"]
            for refl in enriched.reflections:
                task_desc = refl.get("task_description", "")
                what_failed = refl.get("what_failed", "")
                why = refl.get("why", "")
                what_to_try = refl.get("what_to_try", "")
                lines.append(
                    f"- Task: {task_desc} | Failed: {what_failed} | "
                    f"Why: {why} | Try: {what_to_try}"
                )
            sections["reflections"] = "\n".join(lines)

        # Patterns
        if enriched.patterns:
            lines = ["Similar tasks have been answered like this:"]
            for pattern in enriched.patterns:
                detail = f'- "{pattern.trigger}": {pattern.description}'
                if pattern.success_count > 0:
                    detail += f" ({pattern.success_count} successes)"
                lines.append(detail)
            sections["patterns"] = "\n".join(lines)

        # Conversation
        if enriched.conversation_turns:
            lines = ["Recent conversation:"]
            for turn in enriched.conversation_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                display_role = role.capitalize()
                lines.append(f"{display_role}: {content}")
            sections["conversation"] = "\n".join(lines)

        return sections

    def _rank_sections(
        self,
        enriched: EnrichedContext,
    ) -> list[SectionDef]:
        """Rank formatted sections by activation score (token weight).

        The activation score is the per-section token estimate from the
        enrichment pipeline. Higher token counts indicate more relevant
        content was retrieved, so those sections rank first.

        Only non-empty sections are included.

        Returns:
            List of SectionDef ordered by activation_tokens descending.
        """
        sections = self._format_sections(enriched)

        activation_map: dict[str, int] = {
            "entities": enriched.entity_tokens,
            "relationships": enriched.relationship_tokens,
            "communities": enriched.community_tokens,
            "memories": enriched.memory_tokens,
            "reflections": enriched.reflection_tokens,
            "patterns": enriched.pattern_tokens,
            "conversation": enriched.conversation_tokens,
        }

        defs: list[SectionDef] = []
        for name, text in sections.items():
            defs.append(SectionDef(
                name=name,
                text=text,
                activation_tokens=activation_map.get(name, 0),
            ))

        # Sort by activation score descending (most relevant first)
        defs.sort(key=lambda s: s.activation_tokens, reverse=True)
        return defs

    def _trim_to_budget(
        self,
        ranked_sections: list[SectionDef],
    ) -> list[SectionDef]:
        """Trim ranked sections to fit within the token budget.

        Iterates through sections in activation-ranked order and includes
        each section only if it fits within the remaining budget. Sections
        that do not fit are dropped entirely.

        The conversation section is excluded from trimming here because
        conversation turns are added as separate messages (not in the
        system prompt content that gets budgeted).

        Args:
            ranked_sections: Sections ordered by activation score.

        Returns:
            Subset of sections that fit within the budget, in the same order.
        """
        budget = self._budget.available_budget
        result: list[SectionDef] = []
        used: int = 0

        for section in ranked_sections:
            # Conversation is handled separately as individual messages,
            # but we still include the summary in the system prompt if it fits
            cost = self._budget.estimate_tokens(section.text)
            if used + cost <= budget:
                result.append(section)
                used += cost

        return result
