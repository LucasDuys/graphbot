"""Tests for ConversationMemory -- per-chat message storage and retrieval."""

from __future__ import annotations

import pytest

from core_gb.conversation import ConversationMemory
from graph.store import GraphStore


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


class TestConversationMemoryStore:
    """Storing messages in the graph as Memory nodes."""

    def test_add_message_creates_memory_node(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        conv.add_message("chat-001", "user", "Hello there")

        # Verify a Memory node with category=conversation exists.
        rows = store.query(
            "MATCH (m:Memory) WHERE m.category = $cat AND m.source_episode = $ep "
            "RETURN m.content AS content",
            {"cat": "conversation", "ep": "chat-001"},
        )
        assert len(rows) == 1
        assert "Hello there" in str(rows[0]["content"])
        store.close()

    def test_add_message_stores_role(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        conv.add_message("chat-001", "user", "What is Python?")

        rows = store.query(
            "MATCH (m:Memory) WHERE m.category = $cat AND m.source_episode = $ep "
            "RETURN m.content AS content",
            {"cat": "conversation", "ep": "chat-001"},
        )
        content = str(rows[0]["content"])
        assert "user" in content
        assert "What is Python?" in content
        store.close()

    def test_add_multiple_messages_same_chat(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        conv.add_message("chat-001", "user", "Hello")
        conv.add_message("chat-001", "assistant", "Hi there!")
        conv.add_message("chat-001", "user", "How are you?")

        rows = store.query(
            "MATCH (m:Memory) WHERE m.category = $cat AND m.source_episode = $ep "
            "RETURN m.content AS content",
            {"cat": "conversation", "ep": "chat-001"},
        )
        assert len(rows) == 3
        store.close()


class TestConversationMemoryRetrieve:
    """Retrieving recent messages for a chat_id."""

    def test_get_history_returns_messages_in_order(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        conv.add_message("chat-001", "user", "First message")
        conv.add_message("chat-001", "assistant", "First reply")
        conv.add_message("chat-001", "user", "Second message")

        history = conv.get_history("chat-001")
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "First message"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "First reply"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "Second message"
        store.close()

    def test_get_history_respects_max_messages(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store, max_messages=3)

        for i in range(5):
            conv.add_message("chat-001", "user", f"Message {i}")

        history = conv.get_history("chat-001")
        assert len(history) == 3
        # Should return the 3 most recent messages.
        assert history[0]["content"] == "Message 2"
        assert history[1]["content"] == "Message 3"
        assert history[2]["content"] == "Message 4"
        store.close()

    def test_get_history_default_max_is_10(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        for i in range(15):
            conv.add_message("chat-001", "user", f"Message {i}")

        history = conv.get_history("chat-001")
        assert len(history) == 10
        # Most recent 10 messages: 5 through 14.
        assert history[0]["content"] == "Message 5"
        assert history[-1]["content"] == "Message 14"
        store.close()

    def test_get_history_empty_for_unknown_chat(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        history = conv.get_history("nonexistent-chat")
        assert history == []
        store.close()


class TestConversationMemoryIsolation:
    """Different chat_ids have separate histories."""

    def test_different_chats_are_isolated(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        conv.add_message("alice", "user", "Alice's message")
        conv.add_message("bob", "user", "Bob's message")
        conv.add_message("alice", "user", "Alice again")

        alice_history = conv.get_history("alice")
        bob_history = conv.get_history("bob")

        assert len(alice_history) == 2
        assert len(bob_history) == 1
        assert alice_history[0]["content"] == "Alice's message"
        assert alice_history[1]["content"] == "Alice again"
        assert bob_history[0]["content"] == "Bob's message"
        store.close()

    def test_max_messages_applied_per_chat(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store, max_messages=2)

        for i in range(5):
            conv.add_message("alice", "user", f"Alice {i}")
        for i in range(3):
            conv.add_message("bob", "user", f"Bob {i}")

        alice_history = conv.get_history("alice")
        bob_history = conv.get_history("bob")

        assert len(alice_history) == 2
        assert len(bob_history) == 2
        # Alice: most recent 2
        assert alice_history[0]["content"] == "Alice 3"
        assert alice_history[1]["content"] == "Alice 4"
        # Bob: most recent 2
        assert bob_history[0]["content"] == "Bob 1"
        assert bob_history[1]["content"] == "Bob 2"
        store.close()


class TestConversationMemoryFormatContext:
    """Formatting conversation history for LLM context injection."""

    def test_format_as_messages_returns_dict_list(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        conv.add_message("chat-001", "user", "What is 2+2?")
        conv.add_message("chat-001", "assistant", "4")
        conv.add_message("chat-001", "user", "And 3+3?")

        messages = conv.format_as_messages("chat-001")
        assert len(messages) == 3
        assert all(isinstance(m, dict) for m in messages)
        assert all("role" in m and "content" in m for m in messages)
        assert messages[0] == {"role": "user", "content": "What is 2+2?"}
        assert messages[1] == {"role": "assistant", "content": "4"}
        assert messages[2] == {"role": "user", "content": "And 3+3?"}
        store.close()

    def test_format_as_messages_empty_for_unknown_chat(self) -> None:
        store = _make_store()
        conv = ConversationMemory(store)

        messages = conv.format_as_messages("nonexistent")
        assert messages == []
        store.close()
