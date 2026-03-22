"""Tests for GraphBot WhatsApp channel with mocked WhatsApp bridge API."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_gb.types import ExecutionResult
from nanobot.channels.graphbot_whatsapp import (
    GraphBotWhatsAppChannel,
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


def _make_bridge_message(
    content: str = "hello",
    sender: str = "12345@s.whatsapp.net",
    message_id: str = "msg-001",
    pn: str = "",
) -> dict[str, str]:
    """Build a minimal bridge message payload."""
    return {
        "type": "message",
        "content": content,
        "sender": sender,
        "pn": pn,
        "id": message_id,
    }


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
def channel(mock_store: MagicMock, mock_orchestrator: AsyncMock) -> GraphBotWhatsAppChannel:
    ch = GraphBotWhatsAppChannel(
        bridge_url="ws://localhost:3001",
        bridge_token="test-token",
        store=mock_store,
        orchestrator=mock_orchestrator,
    )
    # Simulate an active WebSocket connection for send_text.
    ch._ws = AsyncMock()
    ch._connected = True
    return ch


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
        footer = _format_footer(_make_result(), mock_store)
        assert "Graph:" in footer


class TestCountGraphNodes:
    def test_counts_across_tables(self) -> None:
        store = MagicMock()
        store.query.return_value = [{"cnt": 3}]
        total = _count_graph_nodes(store)
        assert total > 0
        assert store.query.call_count > 0

    def test_handles_query_error_gracefully(self) -> None:
        store = MagicMock()
        store.query.side_effect = RuntimeError("db error")
        total = _count_graph_nodes(store)
        assert total == 0


# ---------------------------------------------------------------------------
# Integration-style tests: message handling (mocked WhatsApp bridge)
# ---------------------------------------------------------------------------

class TestOnMessage:
    @pytest.mark.asyncio
    async def test_message_routed_to_orchestrator(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        data = _make_bridge_message(content="What is Python?")
        await channel._on_message(data)

        mock_orchestrator.process.assert_awaited_once_with("What is Python?")

    @pytest.mark.asyncio
    async def test_response_includes_output_and_footer(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        mock_orchestrator.process.return_value = _make_result(
            output="Python is a language", total_cost=0.001
        )
        data = _make_bridge_message(content="What is Python?")
        await channel._on_message(data)

        # Verify the reply was sent via WebSocket.
        channel._ws.send.assert_awaited_once()
        sent_payload = json.loads(channel._ws.send.call_args[0][0])
        assert "Python is a language" in sent_payload["text"]
        assert "nodes" in sent_payload["text"]
        assert "$0.0010" in sent_payload["text"]

    @pytest.mark.asyncio
    async def test_orchestrator_error_sends_error_message(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        mock_orchestrator.process.side_effect = RuntimeError("LLM down")
        data = _make_bridge_message(content="break me")
        await channel._on_message(data)

        channel._ws.send.assert_awaited_once()
        sent_payload = json.loads(channel._ws.send.call_args[0][0])
        assert "error" in sent_payload["text"].lower()
        assert "LLM down" in sent_payload["text"]

    @pytest.mark.asyncio
    async def test_empty_content_ignored(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        data = _make_bridge_message(content="")
        await channel._on_message(data)

        mock_orchestrator.process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_sender_ignored(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        data = _make_bridge_message(content="hello", sender="", pn="")
        await channel._on_message(data)

        mock_orchestrator.process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_duplicate_message_id_deduplicated(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        data = _make_bridge_message(content="hello", message_id="dup-123")
        await channel._on_message(data)
        await channel._on_message(data)

        # Only processed once despite two deliveries.
        mock_orchestrator.process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pn_fallback_when_no_sender(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        """When sender is empty, pn (phone number) is used as chat_id."""
        data = _make_bridge_message(
            content="hello via pn",
            sender="",
            pn="31612345678@s.whatsapp.net",
            message_id="pn-msg-001",
        )
        await channel._on_message(data)

        channel._ws.send.assert_awaited_once()
        sent_payload = json.loads(channel._ws.send.call_args[0][0])
        assert sent_payload["to"] == "31612345678@s.whatsapp.net"

    @pytest.mark.asyncio
    async def test_reply_sent_to_correct_chat_id(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        data = _make_bridge_message(
            content="hello",
            sender="user@lid.whatsapp.net",
            message_id="chat-id-test",
        )
        await channel._on_message(data)

        sent_payload = json.loads(channel._ws.send.call_args[0][0])
        assert sent_payload["type"] == "send"
        assert sent_payload["to"] == "user@lid.whatsapp.net"


# ---------------------------------------------------------------------------
# Bridge message dispatch
# ---------------------------------------------------------------------------

class TestHandleBridgeMessage:
    @pytest.mark.asyncio
    async def test_dispatches_message_type(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        raw = json.dumps(_make_bridge_message(content="routed", message_id="dispatch-1"))
        await channel._handle_bridge_message(raw)

        mock_orchestrator.process.assert_awaited_once_with("routed")

    @pytest.mark.asyncio
    async def test_status_connected_sets_flag(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        channel._connected = False
        raw = json.dumps({"type": "status", "status": "connected"})
        await channel._handle_bridge_message(raw)

        assert channel._connected is True

    @pytest.mark.asyncio
    async def test_status_disconnected_clears_flag(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        channel._connected = True
        raw = json.dumps({"type": "status", "status": "disconnected"})
        await channel._handle_bridge_message(raw)

        assert channel._connected is False

    @pytest.mark.asyncio
    async def test_invalid_json_handled_gracefully(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        # Should not raise.
        await channel._handle_bridge_message("not json at all {{{")

    @pytest.mark.asyncio
    async def test_error_type_handled(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        raw = json.dumps({"type": "error", "error": "session expired"})
        # Should not raise.
        await channel._handle_bridge_message(raw)

    @pytest.mark.asyncio
    async def test_qr_type_handled(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        raw = json.dumps({"type": "qr", "data": "qr-data-here"})
        # Should not raise.
        await channel._handle_bridge_message(raw)


# ---------------------------------------------------------------------------
# Send text
# ---------------------------------------------------------------------------

class TestSendText:
    @pytest.mark.asyncio
    async def test_send_text_uses_websocket(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        await channel._send_text("user@lid.whatsapp.net", "Hello back")

        channel._ws.send.assert_awaited_once()
        sent = json.loads(channel._ws.send.call_args[0][0])
        assert sent["type"] == "send"
        assert sent["to"] == "user@lid.whatsapp.net"
        assert sent["text"] == "Hello back"

    @pytest.mark.asyncio
    async def test_send_text_noop_when_disconnected(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        channel._connected = False
        await channel._send_text("user@lid.whatsapp.net", "Hello")

        channel._ws.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_text_noop_when_no_ws(
        self, channel: GraphBotWhatsAppChannel
    ) -> None:
        channel._ws = None
        await channel._send_text("user@lid.whatsapp.net", "Hello")
        # No exception raised.


# ---------------------------------------------------------------------------
# Channel lifecycle
# ---------------------------------------------------------------------------

class TestChannelLifecycle:
    def test_init_bridge_url_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {"WHATSAPP_BRIDGE_URL": "ws://remote:9999"},
            clear=False,
        ):
            ch = GraphBotWhatsAppChannel()
            assert ch._bridge_url == "ws://remote:9999"

    def test_init_explicit_url_overrides_env(self) -> None:
        with patch.dict(
            "os.environ",
            {"WHATSAPP_BRIDGE_URL": "ws://env:1111"},
            clear=False,
        ):
            ch = GraphBotWhatsAppChannel(bridge_url="ws://explicit:2222")
            assert ch._bridge_url == "ws://explicit:2222"

    def test_init_bridge_token_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {"WHATSAPP_BRIDGE_TOKEN": "env-secret"},
            clear=False,
        ):
            ch = GraphBotWhatsAppChannel()
            assert ch._bridge_token == "env-secret"

    def test_init_default_bridge_url(self) -> None:
        ch = GraphBotWhatsAppChannel()
        assert ch._bridge_url == "ws://localhost:3001"

    def test_persistent_db_path_default(self) -> None:
        ch = GraphBotWhatsAppChannel()
        assert "graphbot.db" in ch._db_path

    @pytest.mark.asyncio
    async def test_stop_closes_store(
        self, mock_store: MagicMock, mock_orchestrator: AsyncMock
    ) -> None:
        ch = GraphBotWhatsAppChannel(
            bridge_url="ws://localhost:3001",
            store=mock_store,
            orchestrator=mock_orchestrator,
        )
        await ch.stop()
        mock_store.close.assert_called_once()
        assert ch._store is None
        assert ch._orchestrator is None

    @pytest.mark.asyncio
    async def test_stop_closes_websocket(self) -> None:
        ch = GraphBotWhatsAppChannel(bridge_url="ws://localhost:3001")
        ws_mock = AsyncMock()
        ch._ws = ws_mock
        ch._connected = True
        await ch.stop()
        ws_mock.close.assert_awaited_once()
        assert ch._ws is None
        assert ch._connected is False

    @pytest.mark.asyncio
    async def test_stop_without_ws_does_not_raise(self) -> None:
        ch = GraphBotWhatsAppChannel(bridge_url="ws://localhost:3001")
        # No ws set -- should not raise.
        await ch.stop()


# ---------------------------------------------------------------------------
# Deduplication buffer
# ---------------------------------------------------------------------------

class TestDeduplication:
    @pytest.mark.asyncio
    async def test_dedup_buffer_caps_at_1000(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        """Ensure the deduplication buffer does not grow unbounded."""
        for i in range(1050):
            data = _make_bridge_message(
                content=f"msg {i}",
                message_id=f"id-{i}",
            )
            await channel._on_message(data)

        assert len(channel._processed_message_ids) <= 1000

    @pytest.mark.asyncio
    async def test_message_without_id_still_processed(
        self, channel: GraphBotWhatsAppChannel, mock_orchestrator: AsyncMock
    ) -> None:
        """Messages with no id field are processed (not deduplicated)."""
        data = _make_bridge_message(content="no id msg", message_id="")
        await channel._on_message(data)
        await channel._on_message(data)

        # Processed both times since there is no id to deduplicate on.
        assert mock_orchestrator.process.await_count == 2
