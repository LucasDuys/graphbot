"""End-to-end tests for the flight search demo pipeline.

Validates dry-run mode with mocked browser, output file creation,
WhatsApp send skip behavior, and DAG structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on the path so demo script imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.demo_flight_search import (
    MockBrowserTool,
    _mock_flight_rows,
    _mock_raw_html_text,
    build_flight_search_dag,
    save_results_markdown,
    stage_browser_search,
    stage_data_extraction,
    stage_format_message,
    stage_whatsapp_send,
)
from core_gb.types import Domain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_flights() -> list[dict[str, Any]]:
    """Return the standard mock flight data."""
    return _mock_flight_rows()


@pytest.fixture
def formatted_message(mock_flights: list[dict[str, Any]]) -> str:
    """Return a formatted flight message from mock data."""
    return stage_format_message(mock_flights)


# ---------------------------------------------------------------------------
# Dry-run mode -- valid flight data
# ---------------------------------------------------------------------------


class TestDryRunFlightData:
    """Tests that dry-run mode produces valid, structured flight data."""

    @pytest.mark.asyncio
    async def test_browser_search_returns_raw_text(self) -> None:
        """stage_browser_search in dry-run mode should return non-empty text."""
        raw_text = await stage_browser_search(live=False)

        assert isinstance(raw_text, str)
        assert len(raw_text) > 0
        assert "Amsterdam" in raw_text or "AMS" in raw_text
        assert "Barcelona" in raw_text or "BCN" in raw_text

    def test_data_extraction_returns_flights(self) -> None:
        """stage_data_extraction in dry-run should return 6 structured flights."""
        raw_text = _mock_raw_html_text()
        flights = stage_data_extraction(raw_text, live=False)

        assert isinstance(flights, list)
        assert len(flights) == 6

    def test_flight_records_have_required_fields(self, mock_flights: list[dict[str, Any]]) -> None:
        """Each flight record must contain all required keys."""
        required_keys = {"airline", "flight_no", "departure", "arrival", "duration", "stops", "price_eur"}

        for flight in mock_flights:
            missing = required_keys - set(flight.keys())
            assert not missing, f"Flight {flight.get('flight_no', '?')} missing keys: {missing}"

    def test_flight_prices_are_positive_integers(self, mock_flights: list[dict[str, Any]]) -> None:
        """All flight prices must be positive integers."""
        for flight in mock_flights:
            assert isinstance(flight["price_eur"], int)
            assert flight["price_eur"] > 0

    def test_flight_airlines_are_known(self, mock_flights: list[dict[str, Any]]) -> None:
        """Mock flights should contain realistic airline names."""
        expected_airlines = {"Transavia", "Vueling", "KLM", "easyJet", "Ryanair", "Lufthansa"}
        actual_airlines = {f["airline"] for f in mock_flights}
        assert actual_airlines == expected_airlines

    def test_format_message_sorted_by_price(self, mock_flights: list[dict[str, Any]]) -> None:
        """Formatted message should list flights cheapest first."""
        message = stage_format_message(mock_flights)
        lines = message.splitlines()

        # Find lines that contain EUR prices (data rows after the header separator).
        price_lines: list[int] = []
        for line in lines:
            if "EUR" in line and "---" not in line and "Best deal" not in line and "Prices are" not in line:
                # Extract the EUR price at the end of the line.
                parts = line.rsplit("EUR", 1)
                if len(parts) == 2:
                    try:
                        price_lines.append(int(parts[1].strip()))
                    except ValueError:
                        continue

        assert len(price_lines) == 6, f"Expected 6 price rows, got {len(price_lines)}"
        assert price_lines == sorted(price_lines), "Flights are not sorted by price ascending"

    def test_format_message_contains_header(self, formatted_message: str) -> None:
        """Formatted message should contain route and header info."""
        assert "Amsterdam" in formatted_message or "AMS" in formatted_message
        assert "Barcelona" in formatted_message or "BCN" in formatted_message
        assert "Airline" in formatted_message

    def test_format_message_shows_best_deal(self, formatted_message: str) -> None:
        """Formatted message should highlight the cheapest flight."""
        assert "Best deal" in formatted_message
        # Ryanair at EUR 39 is the cheapest in mock data.
        assert "Ryanair" in formatted_message
        assert "39" in formatted_message

    def test_format_empty_flights(self) -> None:
        """Formatting an empty flight list should return a no-results message."""
        message = stage_format_message([])
        assert "No flights found" in message


# ---------------------------------------------------------------------------
# Output file tests
# ---------------------------------------------------------------------------


class TestOutputSave:
    """Tests for saving results to demos/flight_search_results.md."""

    def test_save_results_creates_file(self, tmp_path: Path, mock_flights: list[dict[str, Any]]) -> None:
        """save_results_markdown should write a markdown file."""
        dag_nodes = build_flight_search_dag()
        message = stage_format_message(mock_flights)

        with patch("scripts.demo_flight_search._PROJECT_ROOT", tmp_path):
            saved = save_results_markdown(
                message=message,
                flights=mock_flights,
                dag_nodes=dag_nodes,
                sent_via_whatsapp=False,
                live=False,
            )

        assert saved.exists()
        assert saved.name == "flight_search_results.md"

    def test_saved_file_contains_flight_data(self, tmp_path: Path, mock_flights: list[dict[str, Any]]) -> None:
        """The saved markdown should contain structured flight JSON."""
        dag_nodes = build_flight_search_dag()
        message = stage_format_message(mock_flights)

        with patch("scripts.demo_flight_search._PROJECT_ROOT", tmp_path):
            saved = save_results_markdown(
                message=message,
                flights=mock_flights,
                dag_nodes=dag_nodes,
                sent_via_whatsapp=False,
                live=False,
            )

        content = saved.read_text(encoding="utf-8")
        assert "Flight Search Demo Results" in content
        assert "Transavia" in content
        assert "dry-run" in content
        assert "```json" in content

    def test_saved_file_contains_dag_section(self, tmp_path: Path, mock_flights: list[dict[str, Any]]) -> None:
        """The saved markdown should document the DAG decomposition."""
        dag_nodes = build_flight_search_dag()
        message = stage_format_message(mock_flights)

        with patch("scripts.demo_flight_search._PROJECT_ROOT", tmp_path):
            saved = save_results_markdown(
                message=message,
                flights=mock_flights,
                dag_nodes=dag_nodes,
                sent_via_whatsapp=False,
                live=False,
            )

        content = saved.read_text(encoding="utf-8")
        assert "DAG Decomposition" in content
        assert "flight_search_browser" in content
        assert "flight_search_send" in content


# ---------------------------------------------------------------------------
# WhatsApp send skip tests
# ---------------------------------------------------------------------------


class TestWhatsAppSendSkip:
    """Tests that WhatsApp send is correctly skipped when bridge is not running."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_whatsapp(self, formatted_message: str) -> None:
        """In dry-run mode, stage_whatsapp_send should return False (not sent)."""
        sent = await stage_whatsapp_send(formatted_message, live=False)
        assert sent is False

    @pytest.mark.asyncio
    async def test_live_mode_no_chat_id_skips(self, formatted_message: str) -> None:
        """Live mode without WHATSAPP_DEMO_CHAT_ID should skip sending."""
        with patch.dict("os.environ", {"WHATSAPP_DEMO_CHAT_ID": ""}, clear=False):
            sent = await stage_whatsapp_send(formatted_message, live=True)
        assert sent is False

    @pytest.mark.asyncio
    async def test_live_mode_bridge_unavailable_skips(self, formatted_message: str) -> None:
        """Live mode with unreachable bridge should fall back to console."""
        env = {
            "WHATSAPP_DEMO_CHAT_ID": "test-chat-id",
            "WHATSAPP_BRIDGE_URL": "ws://localhost:59999",  # Non-existent port.
        }
        with patch.dict("os.environ", env, clear=False):
            sent = await stage_whatsapp_send(formatted_message, live=True)
        assert sent is False

    @pytest.mark.asyncio
    async def test_dry_run_prints_message(self, formatted_message: str, capsys: pytest.CaptureFixture[str]) -> None:
        """Dry-run WhatsApp stage should print the message to console."""
        await stage_whatsapp_send(formatted_message, live=False)
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "AMS" in captured.out or "Amsterdam" in captured.out


# ---------------------------------------------------------------------------
# MockBrowserTool tests
# ---------------------------------------------------------------------------


class TestMockBrowserTool:
    """Tests for the mock browser used in dry-run mode."""

    @pytest.mark.asyncio
    async def test_navigate_returns_success(self) -> None:
        """MockBrowserTool.navigate should return success with correct URL."""
        browser = MockBrowserTool()
        result = await browser.navigate("https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_extract_text_returns_flight_data(self) -> None:
        """MockBrowserTool.extract_text should return text containing flight info."""
        browser = MockBrowserTool()
        await browser.navigate("https://google.com/travel/flights")
        result = await browser.extract_text("body")

        assert result["success"] is True
        text = result["text"]
        assert "AMS" in text or "Amsterdam" in text
        assert "BCN" in text or "Barcelona" in text
        assert "EUR" in text

    @pytest.mark.asyncio
    async def test_screenshot_returns_mock_bytes(self) -> None:
        """MockBrowserTool.screenshot should return mock data."""
        browser = MockBrowserTool()
        result = await browser.screenshot(path="/tmp/test.png")

        assert result["success"] is True
        assert result["path"] == "/tmp/test.png"

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self) -> None:
        """MockBrowserTool.close should complete without error."""
        browser = MockBrowserTool()
        await browser.close()  # Should not raise.


# ---------------------------------------------------------------------------
# DAG structure tests
# ---------------------------------------------------------------------------


class TestDAGStructure:
    """Tests for the flight search DAG definition."""

    def test_dag_has_four_nodes(self) -> None:
        """The flight search DAG should contain exactly 4 nodes."""
        nodes = build_flight_search_dag()
        assert len(nodes) == 4

    def test_dag_node_ids(self) -> None:
        """DAG node IDs should match the expected pipeline stages."""
        nodes = build_flight_search_dag()
        ids = [n.id for n in nodes]
        assert ids == [
            "flight_search_browser",
            "flight_search_extract",
            "flight_search_format",
            "flight_search_send",
        ]

    def test_dag_sequential_dependencies(self) -> None:
        """Each node (except the first) should depend on its predecessor."""
        nodes = build_flight_search_dag()

        assert nodes[0].requires == []
        assert nodes[1].requires == ["flight_search_browser"]
        assert nodes[2].requires == ["flight_search_extract"]
        assert nodes[3].requires == ["flight_search_format"]

    def test_dag_domains(self) -> None:
        """DAG nodes should have correct domain assignments."""
        nodes = build_flight_search_dag()
        domains = [n.domain for n in nodes]

        assert domains[0] == Domain.BROWSER
        assert domains[1] == Domain.SYNTHESIS
        assert domains[2] == Domain.SYNTHESIS
        assert domains[3] == Domain.COMMS

    def test_all_nodes_are_atomic(self) -> None:
        """All flight search DAG nodes should be atomic (leaf nodes)."""
        nodes = build_flight_search_dag()
        for node in nodes:
            assert node.is_atomic is True, f"Node {node.id} should be atomic"


# ---------------------------------------------------------------------------
# Full pipeline integration (mocked)
# ---------------------------------------------------------------------------


class TestFullPipelineMocked:
    """Integration test running the full pipeline stages with mocks."""

    @pytest.mark.asyncio
    async def test_full_dry_run_pipeline(self, tmp_path: Path) -> None:
        """Run all 4 stages end-to-end in dry-run mode and verify outputs."""
        # Stage 1: Browser search
        raw_text = await stage_browser_search(live=False)
        assert len(raw_text) > 0

        # Stage 2: Data extraction
        flights = stage_data_extraction(raw_text, live=False)
        assert len(flights) == 6

        # Stage 3: Formatting
        message = stage_format_message(flights)
        assert "Best deal" in message

        # Stage 4: WhatsApp send (skipped in dry-run)
        sent = await stage_whatsapp_send(message, live=False)
        assert sent is False

        # Save results
        dag_nodes = build_flight_search_dag()
        with patch("scripts.demo_flight_search._PROJECT_ROOT", tmp_path):
            saved = save_results_markdown(
                message=message,
                flights=flights,
                dag_nodes=dag_nodes,
                sent_via_whatsapp=sent,
                live=False,
            )
        assert saved.exists()
        content = saved.read_text(encoding="utf-8")
        assert "Console" in content  # Delivery method should be Console, not WhatsApp.
