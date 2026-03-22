"""GraphBot Telegram channel -- bridges Telegram messages to Orchestrator."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

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


class GraphBotTelegramChannel:
    """Telegram channel that routes messages through the GraphBot Orchestrator.

    Unlike the base TelegramChannel (which uses MessageBus), this channel
    directly invokes Orchestrator.process() and returns the result with a
    stats footer showing graph node count and inference cost.

    Config:
        TELEGRAM_BOT_TOKEN -- set in .env.local
    """

    def __init__(
        self,
        token: str | None = None,
        db_path: str = _DB_PATH,
        store: GraphStore | None = None,
        orchestrator: Orchestrator | None = None,
    ) -> None:
        _load_env()
        self._token: str = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._db_path: str = db_path
        self._store: GraphStore | None = store
        self._orchestrator: Orchestrator | None = orchestrator
        self._app: Application | None = None
        self._running: bool = False

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
        """Start Telegram long-polling and listen for messages."""
        if not self._token:
            logger.error("TELEGRAM_BOT_TOKEN not configured")
            return

        self._running = True
        self._ensure_orchestrator()

        builder = Application.builder().token(self._token)
        self._app = builder.build()

        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        await self._app.initialize()
        await self._app.start()

        bot_info = await self._app.bot.get_me()
        logger.info("GraphBot Telegram @{} connected", bot_info.username)

        await self._app.updater.start_polling(
            allowed_updates=["message"],
            drop_pending_updates=True,
        )

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Shut down polling and close the graph store."""
        self._running = False
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None
        if self._store is not None:
            self._store.close()
            self._store = None
            self._orchestrator = None

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        await update.message.reply_text(
            f"Hi {user.first_name}! I am GraphBot.\n"
            "Send me a message and I will process it through the knowledge graph."
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Receive a text message, run through Orchestrator, reply with result + footer."""
        if not update.message or not update.message.text:
            return

        message_text: str = update.message.text
        chat_id: int = update.message.chat_id
        logger.info("GraphBot received from chat {}: {}...", chat_id, message_text[:80])

        store, orchestrator = self._ensure_orchestrator()

        try:
            result: ExecutionResult = await orchestrator.process(message_text)
            response = result.output + _format_footer(result, store)
        except Exception as exc:
            logger.error("Orchestrator error: {}", exc)
            response = f"An error occurred while processing your message: {exc}"

        await update.message.reply_text(response)
