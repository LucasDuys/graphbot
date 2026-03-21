"""IntakeParser -- rule-based, zero-cost intent classification.

Classifies user messages into domains, estimates complexity, and extracts
entities without any LLM calls. Pure pattern matching for zero-token cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core_gb.types import Domain


STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "against",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "its", "his", "her", "their", "our", "your", "my", "me", "him",
    "them", "us", "you", "it", "she", "he", "we", "they", "i",
    "also", "plus", "well", "get", "got", "let", "put", "say", "said",
    "tell", "told", "make", "made", "take", "give", "new", "old", "big",
    "small", "good", "bad", "first", "last", "long", "great", "little",
    "right", "still", "find", "know", "want", "think", "see", "look",
    "come", "go", "like", "use", "try", "ask", "work", "call", "keep",
    "provide", "set", "number", "part", "turn",
    "please", "hello", "hey", "thanks", "thank", "okay", "yes",
})

CONJUNCTIONS: tuple[str, ...] = ("and", "then", "also", "plus", "as well")


@dataclass(frozen=True)
class IntakeResult:
    domain: Domain
    complexity: int
    entities: tuple[str, ...]
    is_simple: bool
    raw_message: str


class IntakeParser:
    """Zero-cost, zero-token intent classification via pattern matching."""

    DOMAIN_KEYWORDS: dict[Domain, set[str]] = {
        Domain.FILE: {
            "file", "read", "write", "save", "open", "create", "delete",
            "rename", "move", "copy", "directory", "folder", "path",
            "readme", "todo", "log",
        },
        Domain.WEB: {
            "search", "browse", "fetch", "url", "http", "website",
            "weather", "news", "download", "api", "web",
        },
        Domain.CODE: {
            "code", "function", "class", "bug", "fix", "refactor", "test",
            "compile", "run", "debug", "implement", "script", "program",
            "python", "javascript",
        },
        Domain.COMMS: {
            "email", "message", "send", "notify", "slack", "chat", "reply",
            "forward", "telegram", "discord",
        },
        Domain.SYSTEM: {
            "calculate", "math", "convert", "time", "date", "schedule",
            "remind", "timer", "alarm", "what", "how", "explain",
        },
        Domain.SYNTHESIS: {
            "compare", "analyze", "summarize", "report", "review",
            "evaluate", "research", "plan", "design",
        },
    }

    def parse(self, message: str) -> IntakeResult:
        """Classify message intent without any LLM calls.

        Returns IntakeResult with domain, complexity estimate, extracted entities.
        """
        domain = self._classify_domain(message)
        complexity = self._estimate_complexity(message)
        entities = self._extract_entities(message)
        multi_domain = self._has_multi_domain_signals(message)
        # Tasks needing tools should always decompose (not be "simple")
        needs_tool = self._needs_tool(message)
        is_simple = complexity <= 2 and len(entities) <= 1 and not multi_domain and not needs_tool

        return IntakeResult(
            domain=domain,
            complexity=complexity,
            entities=tuple(entities),
            is_simple=is_simple,
            raw_message=message,
        )

    def _classify_domain(self, message: str) -> Domain:
        """Count keyword matches per domain, pick highest. Tie-break: SYNTHESIS."""
        lower = message.lower()
        words = set(re.findall(r"[a-z]+", lower))
        # Also check for substring matches (e.g. "http" in a URL)
        scores: dict[Domain, int] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in words:
                    score += 1
                elif kw in lower:
                    score += 1
            scores[domain] = score

        max_score = max(scores.values())
        if max_score == 0:
            return Domain.SYSTEM

        # Collect domains with max score
        tied = [d for d, s in scores.items() if s == max_score]
        if Domain.SYNTHESIS in tied:
            return Domain.SYNTHESIS
        return tied[0]

    def _estimate_complexity(self, message: str) -> int:
        """Heuristic complexity estimation from 1-5."""
        words = message.split()
        word_count = len(words)
        lower = message.lower()

        complexity = 1

        if word_count > 15:
            complexity += 1
        if word_count > 30:
            complexity += 1

        # Count unique conjunctions present
        unique_conjunctions = sum(
            1 for conj in CONJUNCTIONS if f" {conj} " in f" {lower} "
        )
        if unique_conjunctions > 0:
            complexity += unique_conjunctions

        # Comma-separated items
        comma_count = message.count(",")
        if comma_count >= 2:
            complexity += 1

        # Multiple questions
        question_count = message.count("?")
        if question_count >= 2:
            complexity += 1

        return min(complexity, 5)

    def _has_multi_domain_signals(self, message: str) -> bool:
        """Check if the message has keywords from multiple domains."""
        lower = message.lower()
        words = set(re.findall(r"[a-z]+", lower))
        domains_hit = 0
        for keywords in self.DOMAIN_KEYWORDS.values():
            for kw in keywords:
                if kw in words or kw in lower:
                    domains_hit += 1
                    break
        return domains_hit >= 2

    @staticmethod
    def _needs_tool(message: str) -> bool:
        """Check if the message requires tool access (file, web, shell)."""
        lower = message.lower()
        tool_signals = [
            # File operations
            "read ", "read the", "open ", "list files", "list all", "find files",
            "search for", "search the", ".py", ".md", ".json", ".toml", ".txt",
            "directory", "folder", "pyproject", "readme", "claude.md",
            # Web operations
            "search the web", "fetch", "scrape", "browse", "url ", "http",
            "look up", "find online",
            # Shell operations
            "run ", "run the", "execute", "git log", "git ", "pytest", "command",
            "terminal", "shell", "pip ",
        ]
        return any(signal in lower for signal in tool_signals)

    def _extract_entities(self, message: str) -> list[str]:
        """Extract capitalized words that are not common stop words and are 3+ chars."""
        if not message:
            return []
        # Split into words, strip punctuation from edges
        tokens = message.split()
        entities: list[str] = []
        seen: set[str] = set()

        for i, token in enumerate(tokens):
            # Strip punctuation
            clean = re.sub(r"[^a-zA-Z]", "", token)
            if len(clean) < 3:
                continue
            # Must start with uppercase
            if not clean[0].isupper():
                continue
            # Skip first word of the message (sentence-start capitalization)
            if i == 0:
                continue
            # Skip if lowercase form is a stop word
            if clean.lower() in STOP_WORDS:
                continue
            if clean not in seen:
                seen.add(clean)
                entities.append(clean)

        return entities
