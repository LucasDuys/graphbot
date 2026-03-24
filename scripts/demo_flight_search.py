"""Flight search demo -- pipeline design and DAG definition.

Demonstrates a multi-step task pipeline:
  "Find cheap flights from Amsterdam to Barcelona next week and
   message me the results on WhatsApp"

DAG decomposition:
  1. Browser search  -- navigate to flight search, extract results
  2. Data extraction -- parse raw HTML text into structured flight data
  3. Formatting      -- render a clean message with prices and airlines
  4. WhatsApp send   -- deliver via WhatsApp bridge (or print to console)

Modes:
  --dry-run  (default)  Uses mocked browser data, prints to console.
  --live               Uses real Playwright browser + live WhatsApp bridge.

Output is always saved to demos/flight_search_results.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root + path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core_gb.types import Domain, FlowType, TaskNode
from tools_gb.browser_planner import BrowserAction, BrowserPlan, BrowserPlanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock flight data (realistic AMS -> BCN prices/airlines)
# ---------------------------------------------------------------------------

_NEXT_MONDAY = datetime.now() + timedelta(days=(7 - datetime.now().weekday()) % 7 or 7)


def _mock_flight_rows() -> list[dict[str, Any]]:
    """Return realistic mock flight search results for AMS -> BCN."""
    base_date = _NEXT_MONDAY
    return [
        {
            "airline": "Transavia",
            "flight_no": "HV5143",
            "departure": f"{base_date.strftime('%a %d %b')} 06:25",
            "arrival": f"{base_date.strftime('%a %d %b')} 09:00",
            "duration": "2h 35m",
            "stops": "Direct",
            "price_eur": 49,
        },
        {
            "airline": "Vueling",
            "flight_no": "VY8408",
            "departure": f"{base_date.strftime('%a %d %b')} 10:15",
            "arrival": f"{base_date.strftime('%a %d %b')} 12:55",
            "duration": "2h 40m",
            "stops": "Direct",
            "price_eur": 67,
        },
        {
            "airline": "KLM",
            "flight_no": "KL1677",
            "departure": f"{base_date.strftime('%a %d %b')} 14:30",
            "arrival": f"{base_date.strftime('%a %d %b')} 17:05",
            "duration": "2h 35m",
            "stops": "Direct",
            "price_eur": 112,
        },
        {
            "airline": "easyJet",
            "flight_no": "U27802",
            "departure": f"{base_date.strftime('%a %d %b')} 07:40",
            "arrival": f"{base_date.strftime('%a %d %b')} 10:20",
            "duration": "2h 40m",
            "stops": "Direct",
            "price_eur": 54,
        },
        {
            "airline": "Ryanair",
            "flight_no": "FR3812",
            "departure": f"{base_date.strftime('%a %d %b')} 19:55",
            "arrival": f"{base_date.strftime('%a %d %b')} 22:30",
            "duration": "2h 35m",
            "stops": "Direct",
            "price_eur": 39,
        },
        {
            "airline": "Lufthansa",
            "flight_no": "LH989",
            "departure": f"{base_date.strftime('%a %d %b')} 12:10",
            "arrival": f"{base_date.strftime('%a %d %b')} 16:50",
            "duration": "4h 40m",
            "stops": "1 stop (FRA)",
            "price_eur": 143,
        },
    ]


def _mock_raw_html_text() -> str:
    """Simulate the raw text that a browser extract_text call would return."""
    lines: list[str] = ["Flight results: Amsterdam (AMS) to Barcelona (BCN)", ""]
    for f in _mock_flight_rows():
        lines.append(
            f"{f['airline']} {f['flight_no']} | "
            f"{f['departure']} -> {f['arrival']} | "
            f"{f['duration']} | {f['stops']} | "
            f"EUR {f['price_eur']}"
        )
    lines.append("")
    lines.append("Prices are per person, one-way, including taxes.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock BrowserTool (Playwright mock)
# ---------------------------------------------------------------------------


class MockBrowserTool:
    """Drop-in replacement for BrowserTool that returns canned flight data.

    Follows the same async interface as tools_gb.browser.BrowserTool so the
    BrowserPlanner grounder can execute plans against it without changes.
    """

    def __init__(self) -> None:
        self._current_url: str = ""

    async def navigate(self, url: str) -> dict[str, Any]:
        logger.info("MockBrowser: navigate -> %s", url)
        self._current_url = url
        return {
            "success": True,
            "url": url,
            "title": "Google Flights - AMS to BCN",
        }

    async def extract_text(self, selector: str = "body") -> dict[str, Any]:
        logger.info("MockBrowser: extract_text (selector=%s)", selector)
        return {
            "success": True,
            "text": _mock_raw_html_text(),
            "url": self._current_url,
        }

    async def screenshot(self, path: str | None = None) -> dict[str, Any]:
        logger.info("MockBrowser: screenshot (path=%s)", path)
        return {
            "success": True,
            "data": b"<mock-png-bytes>",
            "path": path,
            "url": self._current_url,
        }

    async def click(self, selector: str) -> dict[str, Any]:
        logger.info("MockBrowser: click (selector=%s)", selector)
        return {"success": True, "selector": selector, "url": self._current_url}

    async def fill(self, selector: str, value: str) -> dict[str, Any]:
        logger.info("MockBrowser: fill (selector=%s)", selector)
        return {"success": True, "selector": selector, "url": self._current_url}

    async def close(self) -> None:
        logger.info("MockBrowser: close")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------


def build_flight_search_dag() -> list[TaskNode]:
    """Build the 4-node DAG for the flight search pipeline.

    DAG shape (sequential):
      browser_search -> data_extraction -> formatting -> whatsapp_send

    Returns:
        Ordered list of TaskNodes representing the pipeline.
    """
    root_id = "flight_search_root"

    node_search = TaskNode(
        id="flight_search_browser",
        description=(
            "Navigate to https://www.google.com/travel/flights and search for "
            "one-way flights from Amsterdam (AMS) to Barcelona (BCN) for next week. "
            "Extract the text of the results page."
        ),
        parent_id=root_id,
        is_atomic=True,
        domain=Domain.BROWSER,
        tool_method="browser_extract_text",
        tool_params={"url": "https://www.google.com/travel/flights", "selector": "body"},
        complexity=2,
    )

    node_extract = TaskNode(
        id="flight_search_extract",
        description=(
            "Parse the raw flight search results text into structured data: "
            "airline, flight number, departure, arrival, duration, stops, price."
        ),
        parent_id=root_id,
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        requires=["flight_search_browser"],
        consumes=["flight_search_browser"],
        complexity=1,
    )

    node_format = TaskNode(
        id="flight_search_format",
        description=(
            "Format the structured flight data into a clean, readable message "
            "suitable for WhatsApp delivery, sorted by price ascending."
        ),
        parent_id=root_id,
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        requires=["flight_search_extract"],
        consumes=["flight_search_extract"],
        complexity=1,
    )

    node_send = TaskNode(
        id="flight_search_send",
        description=(
            "Send the formatted flight results message via WhatsApp. "
            "Falls back to console output if the bridge is unavailable."
        ),
        parent_id=root_id,
        is_atomic=True,
        domain=Domain.COMMS,
        requires=["flight_search_format"],
        consumes=["flight_search_format"],
        complexity=1,
    )

    return [node_search, node_extract, node_format, node_send]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


async def stage_browser_search(live: bool) -> str:
    """Stage 1: Browser search -- navigate and extract raw flight text.

    Args:
        live: If True, use real Playwright browser. Otherwise use mock.

    Returns:
        Raw text extracted from the flight search results page.
    """
    planner = BrowserPlanner()
    plan = BrowserPlan(
        task="Search flights AMS to BCN",
        steps=[
            BrowserAction(
                action="navigate",
                params={"url": "https://www.google.com/travel/flights"},
                description="Open Google Flights",
            ),
            BrowserAction(
                action="extract_text",
                params={"selector": "body"},
                description="Extract flight results text",
            ),
        ],
    )

    if live:
        from tools_gb.browser import BrowserTool

        browser = BrowserTool(headless=True)
        try:
            result = await planner.ground(plan, browser)
        finally:
            await browser.close()
    else:
        browser = MockBrowserTool()
        result = await planner.ground(plan, browser)

    if not result["success"]:
        raise RuntimeError(f"Browser search failed: {result.get('error', 'unknown')}")

    raw_text: str = result.get("output", "")
    logger.info("Stage 1 complete: extracted %d chars of raw text", len(raw_text))
    return raw_text


def stage_data_extraction(raw_text: str, live: bool) -> list[dict[str, Any]]:
    """Stage 2: Parse raw text into structured flight records.

    In dry-run mode, returns the known mock data directly.
    In live mode, applies simple line parsing heuristics to the raw text.

    Args:
        raw_text: Raw text from the browser extraction stage.
        live: Whether we are running in live mode.

    Returns:
        List of flight dicts with keys: airline, flight_no, departure,
        arrival, duration, stops, price_eur.
    """
    if not live:
        # In mock mode we already know the exact structure.
        flights = _mock_flight_rows()
        logger.info("Stage 2 complete: extracted %d flights (mock)", len(flights))
        return flights

    # Live mode: best-effort line parsing of raw page text.
    flights: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if "|" not in line or "EUR" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        try:
            # Expected format: "Airline FLNO | dep -> arr | dur | stops | EUR price"
            airline_part = parts[0].split()
            airline = " ".join(airline_part[:-1]) if len(airline_part) > 1 else airline_part[0]
            flight_no = airline_part[-1] if len(airline_part) > 1 else ""
            time_parts = parts[1].split("->")
            departure = time_parts[0].strip() if len(time_parts) > 0 else ""
            arrival = time_parts[1].strip() if len(time_parts) > 1 else ""
            duration = parts[2]
            stops = parts[3]
            price_str = parts[4].replace("EUR", "").strip()
            price_eur = int(price_str)
            flights.append({
                "airline": airline,
                "flight_no": flight_no,
                "departure": departure,
                "arrival": arrival,
                "duration": duration,
                "stops": stops,
                "price_eur": price_eur,
            })
        except (ValueError, IndexError) as exc:
            logger.warning("Could not parse flight line: %s (%s)", line, exc)
            continue

    logger.info("Stage 2 complete: extracted %d flights (live)", len(flights))
    return flights


def stage_format_message(flights: list[dict[str, Any]]) -> str:
    """Stage 3: Format flight data into a clean WhatsApp-ready message.

    Sorts flights by price (cheapest first) and renders a readable summary.

    Args:
        flights: List of structured flight dicts.

    Returns:
        Formatted message string.
    """
    if not flights:
        return "No flights found for Amsterdam (AMS) to Barcelona (BCN) next week."

    sorted_flights = sorted(flights, key=lambda f: f["price_eur"])

    lines: list[str] = [
        "Cheap flights: Amsterdam (AMS) -> Barcelona (BCN)",
        f"Week of {_NEXT_MONDAY.strftime('%d %b %Y')}",
        "",
        f"{'#':<3} {'Airline':<12} {'Flight':<9} {'Departs':<22} {'Duration':<9} {'Stops':<14} {'Price':>6}",
        "-" * 80,
    ]

    for i, f in enumerate(sorted_flights, 1):
        lines.append(
            f"{i:<3} {f['airline']:<12} {f['flight_no']:<9} "
            f"{f['departure']:<22} {f['duration']:<9} {f['stops']:<14} "
            f"EUR {f['price_eur']:>3}"
        )

    lines.append("-" * 80)
    cheapest = sorted_flights[0]
    lines.append(
        f"Best deal: {cheapest['airline']} {cheapest['flight_no']} "
        f"at EUR {cheapest['price_eur']} ({cheapest['duration']}, {cheapest['stops']})"
    )
    lines.append("")
    lines.append("Prices are per person, one-way, including taxes.")

    message = "\n".join(lines)
    logger.info("Stage 3 complete: formatted message (%d chars)", len(message))
    return message


async def stage_whatsapp_send(message: str, live: bool) -> bool:
    """Stage 4: Deliver the formatted message via WhatsApp or console.

    In live mode, attempts to connect to the WhatsApp bridge WebSocket and
    send the message. Falls back to console output on connection failure.

    In dry-run mode, always prints to console.

    Args:
        message: The formatted flight results message.
        live: Whether to attempt real WhatsApp delivery.

    Returns:
        True if message was sent via WhatsApp, False if printed to console.
    """
    if not live:
        print("\n[DRY RUN] WhatsApp message output:\n")
        print(message)
        logger.info("Stage 4 complete: printed to console (dry-run)")
        return False

    # Attempt live WhatsApp delivery via the Baileys bridge.
    bridge_url = os.environ.get("WHATSAPP_BRIDGE_URL", "ws://localhost:3001")
    chat_id = os.environ.get("WHATSAPP_DEMO_CHAT_ID", "")

    if not chat_id:
        print("\n[LIVE] WHATSAPP_DEMO_CHAT_ID not set, printing to console:\n")
        print(message)
        logger.warning("Stage 4: no WHATSAPP_DEMO_CHAT_ID configured")
        return False

    try:
        import websockets

        async with websockets.connect(bridge_url, open_timeout=5) as ws:
            # Authenticate if token is configured.
            token = os.environ.get("WHATSAPP_BRIDGE_TOKEN", "")
            if token:
                await ws.send(json.dumps({"type": "auth", "token": token}))

            payload = json.dumps({
                "type": "send",
                "to": chat_id,
                "text": message,
            }, ensure_ascii=False)

            await ws.send(payload)
            logger.info("Stage 4 complete: sent via WhatsApp to %s", chat_id)
            print(f"\n[LIVE] Message sent via WhatsApp to {chat_id}")
            return True

    except Exception as exc:
        logger.warning("WhatsApp bridge unavailable (%s), falling back to console", exc)
        print(f"\n[LIVE] WhatsApp bridge unavailable ({exc}), printing to console:\n")
        print(message)
        return False


# ---------------------------------------------------------------------------
# Save results to markdown
# ---------------------------------------------------------------------------


def save_results_markdown(
    message: str,
    flights: list[dict[str, Any]],
    dag_nodes: list[TaskNode],
    sent_via_whatsapp: bool,
    live: bool,
) -> Path:
    """Save demo output to demos/flight_search_results.md.

    Args:
        message: The formatted flight message.
        flights: Structured flight data.
        dag_nodes: The DAG node definitions.
        sent_via_whatsapp: Whether the message was delivered via WhatsApp.
        live: Whether the demo ran in live mode.

    Returns:
        Path to the saved markdown file.
    """
    demos_dir = _PROJECT_ROOT / "demos"
    demos_dir.mkdir(exist_ok=True)
    output_path = demos_dir / "flight_search_results.md"

    mode_label = "live" if live else "dry-run"
    delivery = "WhatsApp" if sent_via_whatsapp else "Console"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections: list[str] = [
        "# Flight Search Demo Results",
        "",
        f"Generated: {timestamp}",
        f"Mode: {mode_label}",
        f"Delivery: {delivery}",
        "",
        "## Task",
        "",
        (
            "Find cheap flights from Amsterdam to Barcelona next week "
            "and message me the results on WhatsApp."
        ),
        "",
        "## DAG Decomposition",
        "",
        "```",
        "browser_search -> data_extraction -> formatting -> whatsapp_send",
        "```",
        "",
        "| Node | Domain | Description |",
        "|------|--------|-------------|",
    ]

    for node in dag_nodes:
        sections.append(
            f"| `{node.id}` | {node.domain.value} | {node.description[:60]}... |"
        )

    sections.extend([
        "",
        "## Flight Data (structured)",
        "",
        "```json",
        json.dumps(flights, indent=2),
        "```",
        "",
        "## Formatted Message",
        "",
        "```",
        message,
        "```",
        "",
    ])

    output_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Results saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(live: bool) -> None:
    """Execute the full flight search pipeline end-to-end.

    Args:
        live: If True, use real browser + WhatsApp. Otherwise use mocks.
    """
    mode = "LIVE" if live else "DRY RUN"
    print(f"=== Flight Search Demo ({mode}) ===\n")

    # Define the DAG (for documentation / introspection).
    dag_nodes = build_flight_search_dag()
    print(f"DAG: {len(dag_nodes)} nodes")
    for node in dag_nodes:
        deps = f" (requires: {', '.join(node.requires)})" if node.requires else ""
        print(f"  [{node.domain.value:>9}] {node.id}{deps}")
    print()

    # Stage 1: Browser search
    print("[1/4] Browser search...")
    raw_text = await stage_browser_search(live)
    print(f"      Extracted {len(raw_text)} chars of raw text")

    # Stage 2: Data extraction
    print("[2/4] Data extraction...")
    flights = stage_data_extraction(raw_text, live)
    print(f"      Found {len(flights)} flights")

    # Stage 3: Format message
    print("[3/4] Formatting results...")
    message = stage_format_message(flights)

    # Stage 4: WhatsApp send (or console fallback)
    print("[4/4] Delivering message...")
    sent_via_whatsapp = await stage_whatsapp_send(message, live)

    # Save results
    output_path = save_results_markdown(
        message=message,
        flights=flights,
        dag_nodes=dag_nodes,
        sent_via_whatsapp=sent_via_whatsapp,
        live=live,
    )
    print(f"\nResults saved to {output_path}")
    print("=== Done ===")


def main() -> None:
    """Entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Flight search demo: AMS -> BCN with WhatsApp notification",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run in live mode (real browser + WhatsApp). Default is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode with mocked data (default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    live = args.live  # --live overrides --dry-run
    asyncio.run(run_pipeline(live))


if __name__ == "__main__":
    main()
