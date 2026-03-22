"""Tests for navigation graph storage and retrieval.

Covers T154 acceptance criteria:
- Navigation sequences cached in knowledge graph for reuse
- Store as Skill nodes: url, action_sequence, extracted_data_template
- On similar navigation task, retrieve cached sequence instead of re-planning
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph.store import GraphStore
from tools_gb.nav_cache import NavigationCache, NavigationSequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> GraphStore:
    """In-memory graph store with schema initialized."""
    s = GraphStore(db_path=None)
    s.initialize()
    return s


@pytest.fixture
def nav_cache(store: GraphStore) -> NavigationCache:
    """NavigationCache backed by the test graph store."""
    return NavigationCache(store)


# ---------------------------------------------------------------------------
# Test: NavigationSequence data model
# ---------------------------------------------------------------------------


class TestNavigationSequence:
    """NavigationSequence dataclass holds navigation step data."""

    def test_create_navigation_sequence(self) -> None:
        """NavigationSequence can be created with all fields."""
        seq = NavigationSequence(
            url="https://example.com",
            action_sequence=[
                {"action": "navigate", "url": "https://example.com"},
                {"action": "click", "selector": "#login"},
                {"action": "fill", "selector": "#email", "value": "user@test.com"},
            ],
            extracted_data_template={"title": "body > h1", "content": "main"},
        )
        assert seq.url == "https://example.com"
        assert len(seq.action_sequence) == 3
        assert seq.extracted_data_template["title"] == "body > h1"

    def test_create_minimal_sequence(self) -> None:
        """NavigationSequence works with minimal data."""
        seq = NavigationSequence(
            url="https://example.com",
            action_sequence=[{"action": "navigate", "url": "https://example.com"}],
        )
        assert seq.url == "https://example.com"
        assert seq.extracted_data_template == {}

    def test_sequence_to_dict(self) -> None:
        """NavigationSequence serializes to dict for graph storage."""
        seq = NavigationSequence(
            url="https://example.com/data",
            action_sequence=[{"action": "navigate", "url": "https://example.com/data"}],
            extracted_data_template={"price": ".price-tag"},
        )
        d = seq.to_dict()
        assert d["url"] == "https://example.com/data"
        assert isinstance(d["action_sequence"], str)  # JSON-encoded
        assert isinstance(d["extracted_data_template"], str)  # JSON-encoded

    def test_sequence_from_dict(self) -> None:
        """NavigationSequence deserializes from dict (graph retrieval)."""
        raw = {
            "url": "https://example.com",
            "action_sequence": json.dumps([{"action": "navigate", "url": "https://example.com"}]),
            "extracted_data_template": json.dumps({"title": "h1"}),
        }
        seq = NavigationSequence.from_dict(raw)
        assert seq.url == "https://example.com"
        assert seq.action_sequence == [{"action": "navigate", "url": "https://example.com"}]
        assert seq.extracted_data_template == {"title": "h1"}

    def test_sequence_roundtrip(self) -> None:
        """to_dict -> from_dict preserves all data."""
        original = NavigationSequence(
            url="https://example.com/deep/path",
            action_sequence=[
                {"action": "navigate", "url": "https://example.com/deep/path"},
                {"action": "click", "selector": "a.next-page"},
                {"action": "extract_text", "selector": ".results"},
            ],
            extracted_data_template={"results": ".results", "pagination": ".page-num"},
        )
        restored = NavigationSequence.from_dict(original.to_dict())
        assert restored.url == original.url
        assert restored.action_sequence == original.action_sequence
        assert restored.extracted_data_template == original.extracted_data_template


# ---------------------------------------------------------------------------
# Test: NavigationCache stores sequences in graph
# ---------------------------------------------------------------------------


class TestNavigationCacheStore:
    """Storing navigation sequences in the knowledge graph."""

    def test_store_navigation_sequence(self, nav_cache: NavigationCache) -> None:
        """store() creates a Skill node in the graph."""
        seq = NavigationSequence(
            url="https://example.com",
            action_sequence=[{"action": "navigate", "url": "https://example.com"}],
            extracted_data_template={"title": "h1"},
        )
        skill_id = nav_cache.store(seq)

        assert skill_id is not None
        assert isinstance(skill_id, str)
        assert len(skill_id) > 0

    def test_stored_sequence_retrievable_by_id(
        self, nav_cache: NavigationCache, store: GraphStore
    ) -> None:
        """Stored sequence can be retrieved from the graph by Skill node id."""
        seq = NavigationSequence(
            url="https://example.com/page",
            action_sequence=[
                {"action": "navigate", "url": "https://example.com/page"},
                {"action": "extract_text", "selector": "body"},
            ],
            extracted_data_template={"body_text": "body"},
        )
        skill_id = nav_cache.store(seq)

        node = store.get_node("Skill", skill_id)
        assert node is not None
        assert node["name"] == "nav:https://example.com/page"

    def test_store_multiple_sequences(self, nav_cache: NavigationCache) -> None:
        """Multiple navigation sequences can be stored independently."""
        seq1 = NavigationSequence(
            url="https://site-a.com",
            action_sequence=[{"action": "navigate", "url": "https://site-a.com"}],
        )
        seq2 = NavigationSequence(
            url="https://site-b.com",
            action_sequence=[{"action": "navigate", "url": "https://site-b.com"}],
        )
        id1 = nav_cache.store(seq1)
        id2 = nav_cache.store(seq2)

        assert id1 != id2


# ---------------------------------------------------------------------------
# Test: NavigationCache retrieves cached sequences
# ---------------------------------------------------------------------------


class TestNavigationCacheRetrieve:
    """Retrieving cached navigation sequences by URL."""

    def test_retrieve_by_exact_url(self, nav_cache: NavigationCache) -> None:
        """Exact URL match retrieves the cached sequence."""
        seq = NavigationSequence(
            url="https://example.com/data",
            action_sequence=[
                {"action": "navigate", "url": "https://example.com/data"},
                {"action": "extract_text", "selector": ".data-table"},
            ],
            extracted_data_template={"table": ".data-table"},
        )
        nav_cache.store(seq)

        cached = nav_cache.find_by_url("https://example.com/data")

        assert cached is not None
        assert cached.url == "https://example.com/data"
        assert len(cached.action_sequence) == 2
        assert cached.extracted_data_template == {"table": ".data-table"}

    def test_retrieve_nonexistent_url_returns_none(self, nav_cache: NavigationCache) -> None:
        """Unknown URL returns None."""
        cached = nav_cache.find_by_url("https://never-stored.com")
        assert cached is None

    def test_retrieve_by_domain(self, nav_cache: NavigationCache) -> None:
        """find_by_domain returns all sequences for a given domain."""
        nav_cache.store(NavigationSequence(
            url="https://shop.example.com/products",
            action_sequence=[{"action": "navigate", "url": "https://shop.example.com/products"}],
        ))
        nav_cache.store(NavigationSequence(
            url="https://shop.example.com/cart",
            action_sequence=[{"action": "navigate", "url": "https://shop.example.com/cart"}],
        ))
        nav_cache.store(NavigationSequence(
            url="https://other-site.com/page",
            action_sequence=[{"action": "navigate", "url": "https://other-site.com/page"}],
        ))

        results = nav_cache.find_by_domain("shop.example.com")

        assert len(results) == 2
        urls = {r.url for r in results}
        assert "https://shop.example.com/products" in urls
        assert "https://shop.example.com/cart" in urls

    def test_retrieve_by_domain_empty(self, nav_cache: NavigationCache) -> None:
        """find_by_domain returns empty list for unknown domain."""
        results = nav_cache.find_by_domain("unknown.com")
        assert results == []


# ---------------------------------------------------------------------------
# Test: NavigationCache updates existing sequences
# ---------------------------------------------------------------------------


class TestNavigationCacheUpdate:
    """Updating cached navigation sequences on successful re-execution."""

    def test_store_overwrites_existing_url(self, nav_cache: NavigationCache) -> None:
        """Storing a sequence for an already-cached URL updates it."""
        seq_v1 = NavigationSequence(
            url="https://example.com",
            action_sequence=[{"action": "navigate", "url": "https://example.com"}],
            extracted_data_template={"title": "h1"},
        )
        id_v1 = nav_cache.store(seq_v1)

        seq_v2 = NavigationSequence(
            url="https://example.com",
            action_sequence=[
                {"action": "navigate", "url": "https://example.com"},
                {"action": "click", "selector": "#accept-cookies"},
                {"action": "extract_text", "selector": "body"},
            ],
            extracted_data_template={"title": "h1", "body": "body"},
        )
        id_v2 = nav_cache.store(seq_v2)

        # Same node updated, not a new node
        assert id_v1 == id_v2

        cached = nav_cache.find_by_url("https://example.com")
        assert cached is not None
        assert len(cached.action_sequence) == 3

    def test_delete_cached_sequence(self, nav_cache: NavigationCache) -> None:
        """delete() removes a cached sequence by URL."""
        nav_cache.store(NavigationSequence(
            url="https://example.com/temp",
            action_sequence=[{"action": "navigate", "url": "https://example.com/temp"}],
        ))

        deleted = nav_cache.delete("https://example.com/temp")
        assert deleted is True

        cached = nav_cache.find_by_url("https://example.com/temp")
        assert cached is None

    def test_delete_nonexistent_returns_false(self, nav_cache: NavigationCache) -> None:
        """delete() returns False for URL not in cache."""
        deleted = nav_cache.delete("https://never-stored.com")
        assert deleted is False
