"""Conversation memory -- per-chat message history stored in the knowledge graph.

Messages are persisted as Memory nodes with category="conversation" and
source_episode=chat_id, enabling per-chat isolation and graph-native storage.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from graph.store import GraphStore

logger = logging.getLogger(__name__)

_DEFAULT_MAX_MESSAGES: int = 10


class ConversationMemory:
    """Store and retrieve recent messages per chat_id using graph Memory nodes.

    Each message is stored as a Memory node with:
        - category = "conversation"
        - source_episode = chat_id
        - content = JSON-encoded {"role": ..., "content": ..., "seq": ...}

    The ``seq`` field is a monotonically increasing integer per chat_id,
    used to maintain message ordering without relying on timestamp precision.

    Args:
        store: Initialized GraphStore instance.
        max_messages: Maximum number of recent messages to return per chat.
            Defaults to 10.
    """

    def __init__(
        self,
        store: GraphStore,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
    ) -> None:
        self._store = store
        self._max_messages = max_messages
        # Track per-chat sequence numbers for ordering.
        self._seq_counters: dict[str, int] = {}

    def add_message(self, chat_id: str, role: str, content: str) -> str:
        """Store a message in the graph for the given chat_id.

        Args:
            chat_id: Unique identifier for the conversation (e.g. WhatsApp sender ID).
            role: Message role -- "user" or "assistant".
            content: The message text.

        Returns:
            The ID of the created Memory node.
        """
        seq = self._next_seq(chat_id)
        node_id = str(uuid.uuid4())

        payload = json.dumps({
            "role": role,
            "content": content,
            "seq": seq,
        }, ensure_ascii=False)

        self._store.create_node("Memory", {
            "id": node_id,
            "content": payload,
            "category": "conversation",
            "source_episode": chat_id,
            "confidence": 1.0,
        })

        logger.debug(
            "Stored conversation message: chat=%s role=%s seq=%d id=%s",
            chat_id, role, seq, node_id,
        )
        return node_id

    def get_history(self, chat_id: str) -> list[dict[str, str]]:
        """Retrieve the most recent messages for a chat_id.

        Returns a list of dicts with "role" and "content" keys, ordered
        chronologically (oldest first). Limited to max_messages.

        Args:
            chat_id: The conversation identifier.

        Returns:
            List of {"role": ..., "content": ...} dicts, oldest first.
        """
        rows = self._store.query(
            "MATCH (m:Memory) "
            "WHERE m.category = $cat AND m.source_episode = $ep "
            "RETURN m.content AS content",
            {"cat": "conversation", "ep": chat_id},
        )

        if not rows:
            return []

        # Parse and sort by sequence number.
        parsed: list[tuple[int, str, str]] = []
        for row in rows:
            raw = str(row.get("content", ""))
            try:
                data = json.loads(raw)
                seq = int(data.get("seq", 0))
                role = str(data.get("role", "user"))
                content = str(data.get("content", ""))
                parsed.append((seq, role, content))
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("Skipping malformed conversation node: %s", raw[:80])
                continue

        parsed.sort(key=lambda x: x[0])

        # Take only the most recent max_messages.
        if len(parsed) > self._max_messages:
            parsed = parsed[-self._max_messages:]

        return [{"role": role, "content": content} for _, role, content in parsed]

    def format_as_messages(self, chat_id: str) -> list[dict[str, str]]:
        """Format conversation history as a list of message dicts for LLM context.

        Returns the same format as get_history -- a list of
        {"role": ..., "content": ...} dicts suitable for direct inclusion
        in an LLM messages array.

        Args:
            chat_id: The conversation identifier.

        Returns:
            List of message dicts, oldest first. Empty list if no history.
        """
        return self.get_history(chat_id)

    def _next_seq(self, chat_id: str) -> int:
        """Return the next sequence number for a chat_id.

        On first call for a chat_id, scans existing graph nodes to find the
        current max sequence, then increments from there. Subsequent calls
        use the in-memory counter.
        """
        if chat_id not in self._seq_counters:
            # Bootstrap from existing graph data.
            max_seq = self._scan_max_seq(chat_id)
            self._seq_counters[chat_id] = max_seq

        self._seq_counters[chat_id] += 1
        return self._seq_counters[chat_id]

    def _scan_max_seq(self, chat_id: str) -> int:
        """Scan existing Memory nodes to find the maximum sequence number."""
        rows = self._store.query(
            "MATCH (m:Memory) "
            "WHERE m.category = $cat AND m.source_episode = $ep "
            "RETURN m.content AS content",
            {"cat": "conversation", "ep": chat_id},
        )

        max_seq = 0
        for row in rows:
            raw = str(row.get("content", ""))
            try:
                data = json.loads(raw)
                seq = int(data.get("seq", 0))
                if seq > max_seq:
                    max_seq = seq
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        return max_seq
