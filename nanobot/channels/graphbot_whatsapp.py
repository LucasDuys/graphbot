"""GraphBot WhatsApp channel -- bridges WhatsApp messages to Orchestrator."""

from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loguru import logger

from core_gb.orchestrator import Orchestrator
from core_gb.types import ExecutionResult
from graph.schema import NODE_TYPES
from graph.store import GraphStore
from models.openrouter import OpenRouterProvider
from models.router import ModelRouter

# Persistent database path (project-root/data/graphbot.db).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = str(_PROJECT_ROOT / "data" / "graphbot.db")


def _load_env() -> None:
    """Load .env.local from project root into environment (idempotent)."""
    env_file = _PROJECT_ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _format_footer(result: ExecutionResult, store: GraphStore) -> str:
    """Build a compact stats footer: node count + cost."""
    node_count = _count_graph_nodes(store)
    cost_str = f"${result.total_cost:.4f}" if result.total_cost else "$0.0000"
    return f"\n\n---\nGraph: {node_count} nodes | Cost: {cost_str}"


def _count_graph_nodes(store: GraphStore) -> int:
    """Count total nodes across all tables in the knowledge graph."""
    total = 0
    for node_type in NODE_TYPES:
        try:
            rows = store.query(f"MATCH (n:{node_type.name}) RETURN count(n) AS cnt")
            if rows:
                total += int(rows[0].get("cnt", 0))
        except Exception:
            pass
    return total


class GraphBotWhatsAppChannel:
    """WhatsApp channel that routes messages through the GraphBot Orchestrator.

    Unlike the base WhatsAppChannel (which uses MessageBus), this channel
    directly invokes Orchestrator.process() and returns the result with a
    stats footer showing graph node count and inference cost.

    Connects to the same Node.js Baileys bridge as the base channel,
    communicating over WebSocket.

    Config (from .env.local):
        WHATSAPP_BRIDGE_URL   -- WebSocket URL (default: ws://localhost:3001)
        WHATSAPP_BRIDGE_TOKEN -- optional auth token for the bridge
    """

    def __init__(
        self,
        bridge_url: str | None = None,
        bridge_token: str | None = None,
        db_path: str = _DB_PATH,
        store: GraphStore | None = None,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        _load_env()
        self._bridge_url: str = (
            bridge_url or os.environ.get("WHATSAPP_BRIDGE_URL", "ws://localhost:3001")
        )
        self._bridge_token: str = (
            bridge_token or os.environ.get("WHATSAPP_BRIDGE_TOKEN", "")
        )
        self._db_path: str = db_path
        self._store: GraphStore | None = store
        self._orchestrator: Orchestrator | None = orchestrator
        self._ws: Any = None
        self._connected: bool = False
        self._running: bool = False
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()

    def _ensure_orchestrator(self) -> tuple[GraphStore, Orchestrator]:
        """Lazily initialise the persistent GraphStore and Orchestrator."""
        if self._store is not None and self._orchestrator is not None:
            return self._store, self._orchestrator

        store = GraphStore(self._db_path)
        store.initialize()
        self._store = store

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        provider = OpenRouterProvider(api_key=api_key)
        router = ModelRouter(provider)
        self._orchestrator = Orchestrator(store, router)
        return self._store, self._orchestrator

    async def start(self) -> None:
        """Start WhatsApp bridge connection and listen for messages."""
        if not self._bridge_url:
            logger.error("WHATSAPP_BRIDGE_URL not configured")
            return

        self._running = True
        self._ensure_orchestrator()

        import websockets

        logger.info("GraphBot WhatsApp connecting to bridge at {}", self._bridge_url)

        while self._running:
            try:
                async with websockets.connect(self._bridge_url) as ws:
                    self._ws = ws

                    if self._bridge_token:
                        await ws.send(json.dumps({
                            "type": "auth",
                            "token": self._bridge_token,
                        }))

                    self._connected = True
                    logger.info("GraphBot WhatsApp bridge connected")

                    async for raw_message in ws:
                        try:
                            await self._handle_bridge_message(raw_message)
                        except Exception as exc:
                            logger.error("Error handling bridge message: {}", exc)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._connected = False
                self._ws = None
                logger.warning("WhatsApp bridge connection error: {}", exc)

                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    async def stop(self) -> None:
        """Shut down WebSocket connection and close the graph store."""
        self._running = False
        self._connected = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._store is not None:
            self._store.close()
            self._store = None
            self._orchestrator = None

    async def _send_text(self, chat_id: str, text: str) -> None:
        """Send a text message back through the WhatsApp bridge."""
        if not self._ws or not self._connected:
            logger.warning("WhatsApp bridge not connected, cannot send reply")
            return

        payload = json.dumps({
            "type": "send",
            "to": chat_id,
            "text": text,
        }, ensure_ascii=False)

        try:
            await self._ws.send(payload)
        except Exception as exc:
            logger.error("Error sending WhatsApp reply: {}", exc)

    async def _handle_bridge_message(self, raw: str) -> None:
        """Dispatch an incoming bridge WebSocket frame."""
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from bridge: {}", raw[:100])
            return

        msg_type: str = data.get("type", "")

        if msg_type == "message":
            await self._on_message(data)
        elif msg_type == "status":
            status = data.get("status", "")
            logger.info("WhatsApp status: {}", status)
            if status == "connected":
                self._connected = True
            elif status == "disconnected":
                self._connected = False
        elif msg_type == "qr":
            logger.info("Scan QR code in the bridge terminal to connect WhatsApp")
        elif msg_type == "error":
            logger.error("WhatsApp bridge error: {}", data.get("error"))

    async def _on_message(self, data: dict[str, Any]) -> None:
        """Receive a text message, run through Orchestrator, reply with result + footer."""
        content: str = data.get("content", "")
        if not content:
            return

        # Deduplicate by message id (bridge may re-deliver).
        message_id: str = data.get("id", "")
        if message_id:
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

        # Derive chat_id (full LID for replies) and sender for logging.
        pn: str = data.get("pn", "")
        sender: str = data.get("sender", "")
        chat_id: str = sender if sender else pn

        if not chat_id:
            logger.warning("Message has no sender or pn, cannot reply")
            return

        sender_short: str = chat_id.split("@")[0] if "@" in chat_id else chat_id
        logger.info("GraphBot received from {}: {}...", sender_short, content[:80])

        store, orchestrator = self._ensure_orchestrator()

        try:
            result: ExecutionResult = await orchestrator.process(content, chat_id=chat_id)
            response = result.output + _format_footer(result, store)
        except Exception as exc:
            logger.error("Orchestrator error: {}", exc)
            response = f"An error occurred while processing your message: {exc}"

        await self._send_text(chat_id, response)
