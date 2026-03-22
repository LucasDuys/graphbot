"""Tests for GraphBot Telegram channel with mocked Telegram API."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import ExecutionResult
from nanobot.channels.graphbot_telegram import (
    GraphBotTelegramChannel,
    _count_graph_nodes,
    _format_footer,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    output: str = "Test response",
    total_cost: float = 0.0012,
    total_nodes: int = 3,
    success: bool = True,
) -> ExecutionResult:
    return ExecutionResult(
        root_id="test-root",
        output=output,
        success=success,
        total_nodes=total_nodes,
        total_cost=total_cost,
    )


def _make_update(text: str = "hello", chat_id: int = 42) -> MagicMock:
    """Build a minimal mocked telegram Update with message + user."""
    user = SimpleNamespace(first_name="Alice", id=1, username="alice")
    message = MagicMock()
    message.text = text
    message.chat_id = chat_id
    message.reply_text = AsyncMock()
    update = MagicMock()
    update.message = message
    update.effective_user = user
    return update


@pytest.fixture()
def mock_store() -> MagicMock:
    store = MagicMock()
    store.query.return_value = [{"cnt": 5}]
    store.initialize = MagicMock()
    store.close = MagicMock()
    return store


@pytest.fixture()
def mock_orchestrator() -> AsyncMock:
    orch = AsyncMock()
    orch.process.return_value = _make_result()
    return orch


@pytest.fixture()
def channel(mock_store: MagicMock, mock_orchestrator: AsyncMock) -> GraphBotTelegramChannel:
    return GraphBotTelegramChannel(
        token="test-token",
        store=mock_store,
        orchestrator=mock_orchestrator,
    )


# ---------------------------------------------------------------------------
# Unit tests: footer formatting
# ---------------------------------------------------------------------------

class TestFormatFooter:
    def test_footer_contains_node_count_and_cost(self, mock_store: MagicMock) -> None:
        result = _make_result(total_cost=0.0025)
        footer = _format_footer(result, mock_store)
        assert "nodes" in footer
        assert "$0.0025" in footer

    def test_footer_zero_cost(self, mock_store: MagicMock) -> None:
        result = _make_result(total_cost=0.0)
        footer = _format_footer(result, mock_store)
        assert "$0.0000" in footer

    def test_footer_node_count_from_store(self, mock_store: MagicMock) -> None:
        # Store returns 5 nodes per table; count iterates NODE_TYPES
        footer = _format_footer(_make_result(), mock_store)
        assert "Graph:" in footer


class TestCountGraphNodes:
    def test_counts_across_tables(self) -> None:
        store = MagicMock()
        store.query.return_value = [{"cnt": 3}]
        total = _count_graph_nodes(store)
        # Called once per NODE_TYPE, each returning 3
        assert total > 0
        assert store.query.call_count > 0

    def test_handles_query_error_gracefully(self) -> None:
        store = MagicMock()
        store.query.side_effect = RuntimeError("db error")
        total = _count_graph_nodes(store)
        assert total == 0


# ---------------------------------------------------------------------------
# Integration-style tests: message handling (mocked Telegram API)
# ---------------------------------------------------------------------------

class TestOnMessage:
    @pytest.mark.asyncio
    async def test_message_routed_to_orchestrator(
        self, channel: GraphBotTelegramChannel, mock_orchestrator: AsyncMock
    ) -> None:
        update = _make_update(text="What is Python?")
        await channel._on_message(update, MagicMock())

        mock_orchestrator.process.assert_awaited_once_with("What is Python?")

    @pytest.mark.asyncio
    async def test_response_includes_output_and_footer(
        self, channel: GraphBotTelegramChannel, mock_orchestrator: AsyncMock
    ) -> None:
        mock_orchestrator.process.return_value = _make_result(
            output="Python is a language", total_cost=0.001
        )
        update = _make_update(text="What is Python?")
        await channel._on_message(update, MagicMock())

        reply_text: str = update.message.reply_text.call_args[0][0]
        assert "Python is a language" in reply_text
        assert "nodes" in reply_text
        assert "$0.0010" in reply_text

    @pytest.mark.asyncio
    async def test_orchestrator_error_sends_error_message(
        self, channel: GraphBotTelegramChannel, mock_orchestrator: AsyncMock
    ) -> None:
        mock_orchestrator.process.side_effect = RuntimeError("LLM down")
        update = _make_update(text="break me")
        await channel._on_message(update, MagicMock())

        reply_text: str = update.message.reply_text.call_args[0][0]
        assert "error" in reply_text.lower()
        assert "LLM down" in reply_text

    @pytest.mark.asyncio
    async def test_empty_message_ignored(
        self, channel: GraphBotTelegramChannel, mock_orchestrator: AsyncMock
    ) -> None:
        update = _make_update()
        update.message.text = None
        await channel._on_message(update, MagicMock())

        mock_orchestrator.process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_update_message_ignored(
        self, channel: GraphBotTelegramChannel, mock_orchestrator: AsyncMock
    ) -> None:
        update = MagicMock()
        update.message = None
        await channel._on_message(update, MagicMock())

        mock_orchestrator.process.assert_not_awaited()


class TestOnStart:
    @pytest.mark.asyncio
    async def test_start_command_replies(self, channel: GraphBotTelegramChannel) -> None:
        update = _make_update()
        await channel._on_start(update, MagicMock())

        update.message.reply_text.assert_awaited_once()
        reply_text: str = update.message.reply_text.call_args[0][0]
        assert "GraphBot" in reply_text
        assert "Alice" in reply_text


class TestChannelLifecycle:
    def test_init_without_token_from_env(self) -> None:
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "env-tok"}, clear=False):
            ch = GraphBotTelegramChannel()
            assert ch._token == "env-tok"

    def test_init_explicit_token_overrides_env(self) -> None:
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "env-tok"}, clear=False):
            ch = GraphBotTelegramChannel(token="explicit-tok")
            assert ch._token == "explicit-tok"

    def test_persistent_db_path_default(self) -> None:
        ch = GraphBotTelegramChannel(token="t")
        assert "graphbot.db" in ch._db_path

    @pytest.mark.asyncio
    async def test_stop_closes_store(
        self, mock_store: MagicMock, mock_orchestrator: AsyncMock
    ) -> None:
        ch = GraphBotTelegramChannel(
            token="t", store=mock_store, orchestrator=mock_orchestrator
        )
        await ch.stop()
        mock_store.close.assert_called_once()
        assert ch._store is None
        assert ch._orchestrator is None

    @pytest.mark.asyncio
    async def test_start_without_token_logs_error(self) -> None:
        ch = GraphBotTelegramChannel(token="")
        # Should return immediately without raising
        await ch.start()
        assert not ch._running
