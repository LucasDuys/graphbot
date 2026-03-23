"""Tests for pattern cache pollution fixes (T179).

Covers:
1. Shell/code/browser domain tasks do NOT create patterns
2. Unfilled slots in match result trigger fallback (return None)
3. Polluted patterns are purged on startup via PatternStore.purge_polluted()
4. Domain scoping: patterns store source domain, matcher filters by domain
5. Integration: full round-trip extraction -> matching with domain scoping
"""

from __future__ import annotations

import json

from core_gb.patterns import PatternExtractor, PatternMatcher, PatternStore
from core_gb.types import (
    Domain,
    ExecutionResult,
    FlowType,
    Pattern,
    TaskNode,
    TaskStatus,
)
from graph.store import GraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> GraphStore:
    """Create an in-memory GraphStore with schema initialized."""
    store = GraphStore(db_path=None)
    store.initialize()
    return store


def _success_result(root_id: str = "root") -> ExecutionResult:
    return ExecutionResult(
        root_id=root_id,
        output="done",
        success=True,
        total_nodes=5,
        total_tokens=320,
        total_latency_ms=850.0,
    )


def _make_shell_tree() -> list[TaskNode]:
    """Build a tree with CODE-domain leaves (shell execution)."""
    root = TaskNode(
        id="root",
        description="Check the Python version and pip version",
        children=["leaf_py", "leaf_pip", "agg"],
        domain=Domain.SYNTHESIS,
        complexity=2,
        flow_type=FlowType.PARALLEL,
    )
    leaf_py = TaskNode(
        id="leaf_py",
        description="Run python --version to get the Python version",
        parent_id="root",
        is_atomic=True,
        domain=Domain.CODE,
        complexity=1,
        provides=["python_version"],
    )
    leaf_pip = TaskNode(
        id="leaf_pip",
        description="Run pip --version to get the pip version",
        parent_id="root",
        is_atomic=True,
        domain=Domain.CODE,
        complexity=1,
        provides=["pip_version"],
    )
    agg = TaskNode(
        id="agg",
        description="Aggregate version information",
        parent_id="root",
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=1,
        consumes=["python_version", "pip_version"],
        provides=["version_summary"],
    )
    return [root, leaf_py, leaf_pip, agg]


def _make_browser_tree() -> list[TaskNode]:
    """Build a tree with BROWSER-domain leaves."""
    root = TaskNode(
        id="root",
        description="Scrape prices from Amazon and eBay",
        children=["leaf_amz", "leaf_ebay", "agg"],
        domain=Domain.SYNTHESIS,
        complexity=2,
        flow_type=FlowType.PARALLEL,
    )
    leaf_amz = TaskNode(
        id="leaf_amz",
        description="Browse Amazon for product price",
        parent_id="root",
        is_atomic=True,
        domain=Domain.BROWSER,
        complexity=1,
        provides=["amazon_price"],
    )
    leaf_ebay = TaskNode(
        id="leaf_ebay",
        description="Browse eBay for product price",
        parent_id="root",
        is_atomic=True,
        domain=Domain.BROWSER,
        complexity=1,
        provides=["ebay_price"],
    )
    agg = TaskNode(
        id="agg",
        description="Compare prices",
        parent_id="root",
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=1,
        consumes=["amazon_price", "ebay_price"],
        provides=["price_comparison"],
    )
    return [root, leaf_amz, leaf_ebay, agg]


def _make_web_tree() -> list[TaskNode]:
    """Build a tree with WEB-domain leaves (safe for caching)."""
    root = TaskNode(
        id="root",
        description="Compare the weather in Amsterdam, London, and Berlin",
        children=["leaf_ams", "leaf_lon", "leaf_ber", "agg"],
        domain=Domain.SYNTHESIS,
        complexity=2,
        flow_type=FlowType.PARALLEL,
    )
    leaf_ams = TaskNode(
        id="leaf_ams",
        description="Current weather in Amsterdam",
        parent_id="root",
        is_atomic=True,
        domain=Domain.WEB,
        complexity=1,
        provides=["weather_amsterdam"],
    )
    leaf_lon = TaskNode(
        id="leaf_lon",
        description="Current weather in London",
        parent_id="root",
        is_atomic=True,
        domain=Domain.WEB,
        complexity=1,
        provides=["weather_london"],
    )
    leaf_ber = TaskNode(
        id="leaf_ber",
        description="Current weather in Berlin",
        parent_id="root",
        is_atomic=True,
        domain=Domain.WEB,
        complexity=1,
        provides=["weather_berlin"],
    )
    agg = TaskNode(
        id="agg",
        description="Aggregate weather results",
        parent_id="root",
        is_atomic=True,
        domain=Domain.SYNTHESIS,
        complexity=1,
        consumes=["weather_amsterdam", "weather_london", "weather_berlin"],
        provides=["weather_summary"],
    )
    return [root, leaf_ams, leaf_lon, leaf_ber, agg]


# ---------------------------------------------------------------------------
# 1. Shell/code/browser domain tasks do NOT create patterns
# ---------------------------------------------------------------------------


class TestDomainBlocklistExtraction:
    """PatternExtractor must refuse to extract patterns from tool-executed domains."""

    def test_code_domain_blocked(self) -> None:
        """Tasks with CODE-domain atomic leaves must NOT produce a pattern."""
        extractor = PatternExtractor()
        nodes = _make_shell_tree()
        result = _success_result()

        pattern = extractor.extract(
            task="Check the Python version and pip version",
            nodes=nodes,
            result=result,
        )

        assert pattern is None

    def test_browser_domain_blocked(self) -> None:
        """Tasks with BROWSER-domain atomic leaves must NOT produce a pattern."""
        extractor = PatternExtractor()
        nodes = _make_browser_tree()
        result = _success_result()

        pattern = extractor.extract(
            task="Scrape prices from Amazon and eBay",
            nodes=nodes,
            result=result,
        )

        assert pattern is None

    def test_web_domain_allowed(self) -> None:
        """Tasks with WEB-domain leaves are safe and SHOULD produce patterns."""
        extractor = PatternExtractor()
        nodes = _make_web_tree()
        result = _success_result()

        pattern = extractor.extract(
            task="Compare the weather in Amsterdam, London, and Berlin",
            nodes=nodes,
            result=result,
        )

        assert pattern is not None

    def test_synthesis_only_allowed(self) -> None:
        """Tasks with SYNTHESIS-domain leaves should produce patterns."""
        extractor = PatternExtractor()
        nodes = [
            TaskNode(
                id="root",
                description="Compare pros and cons of A vs B",
                children=["pros", "cons", "agg"],
                domain=Domain.SYNTHESIS,
            ),
            TaskNode(
                id="pros",
                description="List pros of topic A",
                parent_id="root",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                provides=["pros_list"],
            ),
            TaskNode(
                id="cons",
                description="List cons of topic B",
                parent_id="root",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                provides=["cons_list"],
            ),
            TaskNode(
                id="agg",
                description="Aggregate comparison",
                parent_id="root",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                consumes=["pros_list", "cons_list"],
                provides=["comparison"],
            ),
        ]
        result = _success_result()

        pattern = extractor.extract(
            task="Compare pros and cons of A vs B",
            nodes=nodes,
            result=result,
        )

        assert pattern is not None

    def test_mixed_domain_with_code_blocked(self) -> None:
        """If ANY atomic leaf is in a blocked domain, no pattern is extracted."""
        extractor = PatternExtractor()
        nodes = [
            TaskNode(
                id="root",
                description="Get weather and run a script",
                children=["web_leaf", "code_leaf"],
                domain=Domain.SYNTHESIS,
            ),
            TaskNode(
                id="web_leaf",
                description="Fetch weather data",
                parent_id="root",
                is_atomic=True,
                domain=Domain.WEB,
                provides=["weather"],
            ),
            TaskNode(
                id="code_leaf",
                description="Run analysis script",
                parent_id="root",
                is_atomic=True,
                domain=Domain.CODE,
                provides=["analysis"],
            ),
        ]
        result = _success_result()

        pattern = extractor.extract(
            task="Get weather and run a script",
            nodes=nodes,
            result=result,
        )

        assert pattern is None


# ---------------------------------------------------------------------------
# 2. Unfilled slots trigger fallback (return None)
# ---------------------------------------------------------------------------


class TestUnfilledSlotFallback:
    """PatternMatcher.match() must return None when slots cannot be filled."""

    def test_unfilled_slot_returns_none(self) -> None:
        """A pattern match where regex extraction leaves unfilled slots
        should return None to force decomposition."""
        matcher = PatternMatcher()
        # Pattern with 3 slots but task only fills 2 via fuzzy match
        pattern = Pattern(
            id="p-unfilled",
            trigger="Compare {slot_0}, {slot_1}, and {slot_2} weather",
            description="Weather comparison",
            variable_slots=("slot_0", "slot_1", "slot_2"),
            tree_template="[]",
            success_count=5,
        )
        # Task that matches structurally via Levenshtein but cannot
        # fill all slots (fuzzy match has empty bindings)
        result = matcher.match(
            "Tell me about the weather in two cities",
            [pattern],
            threshold=0.3,  # Low threshold to allow fuzzy match
        )

        # Even if Levenshtein score is above threshold, unfilled slots
        # should cause None return
        assert result is None

    def test_all_slots_filled_returns_match(self) -> None:
        """When all slots are successfully filled, match proceeds normally."""
        matcher = PatternMatcher()
        pattern = Pattern(
            id="p-filled",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Multiplication",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
        )

        result = matcher.match("Calculate 7 times 8", [pattern])

        assert result is not None
        matched, bindings = result
        assert matched.id == "p-filled"
        assert bindings == {"slot_0": "7", "slot_1": "8"}

    def test_no_slots_pattern_matches_normally(self) -> None:
        """A pattern with no variable slots matches without slot validation."""
        matcher = PatternMatcher()
        pattern = Pattern(
            id="p-noslots",
            trigger="What is the meaning of life",
            description="Philosophy",
            variable_slots=(),
            tree_template="[]",
            success_count=5,
        )

        result = matcher.match(
            "What is the meaning of life", [pattern]
        )

        assert result is not None
        matched, bindings = result
        assert matched.id == "p-noslots"
        assert bindings == {}


# ---------------------------------------------------------------------------
# 3. Polluted patterns purged on startup
# ---------------------------------------------------------------------------


class TestPollutedPatternPurge:
    """PatternStore.purge_polluted() removes patterns with pollution markers."""

    def test_purge_removes_shell_markers(self) -> None:
        """Patterns with shell-specific slot names or markers are purged."""
        store = _make_store()
        ps = PatternStore(store)

        # Polluted pattern: tree_template references python_version
        polluted = Pattern(
            id="polluted-1",
            trigger="Check the {slot_0} version",
            description="Version check",
            variable_slots=("slot_0",),
            tree_template=json.dumps([{
                "description": "Run python --version",
                "domain": "code",
                "is_atomic": True,
                "complexity": 1,
                "provides": ["python_version"],
                "consumes": [],
            }]),
            success_count=1,
        )
        ps.save(polluted)

        purged_count = ps.purge_polluted()

        assert purged_count == 1
        assert len(ps.load_all()) == 0

    def test_purge_preserves_clean_patterns(self) -> None:
        """Clean patterns without pollution markers survive the purge."""
        store = _make_store()
        ps = PatternStore(store)

        clean = Pattern(
            id="clean-1",
            trigger="Compare weather in {slot_0} and {slot_1}",
            description="Weather comparison",
            variable_slots=("slot_0", "slot_1"),
            tree_template=json.dumps([{
                "description": "Fetch weather for city",
                "domain": "web",
                "is_atomic": True,
                "complexity": 1,
                "provides": ["weather"],
                "consumes": [],
            }]),
            success_count=5,
        )
        ps.save(clean)

        purged_count = ps.purge_polluted()

        assert purged_count == 0
        assert len(ps.load_all()) == 1

    def test_purge_mixed_set(self) -> None:
        """Only polluted patterns are removed; clean patterns remain."""
        store = _make_store()
        ps = PatternStore(store)

        polluted = Pattern(
            id="polluted-mix",
            trigger="Install {slot_0}",
            description="Package install",
            variable_slots=("slot_0",),
            tree_template=json.dumps([{
                "description": "pip install package",
                "domain": "code",
                "is_atomic": True,
                "complexity": 1,
                "provides": [],
                "consumes": [],
            }]),
            success_count=1,
        )
        clean = Pattern(
            id="clean-mix",
            trigger="Summarize {slot_0}",
            description="Summarization",
            variable_slots=("slot_0",),
            tree_template=json.dumps([{
                "description": "Generate summary of topic",
                "domain": "synthesis",
                "is_atomic": True,
                "complexity": 1,
                "provides": ["summary"],
                "consumes": [],
            }]),
            success_count=10,
        )
        ps.save(polluted)
        ps.save(clean)

        purged_count = ps.purge_polluted()

        assert purged_count == 1
        remaining = ps.load_all()
        assert len(remaining) == 1
        assert remaining[0].id == "clean-mix"

    def test_purge_detects_unfilled_slot_indicators(self) -> None:
        """Patterns whose tree_template contains unfilled slot syntax are purged."""
        store = _make_store()
        ps = PatternStore(store)

        broken = Pattern(
            id="broken-slots",
            trigger="Do {slot_0}",
            description="Broken pattern",
            variable_slots=("slot_0",),
            tree_template=json.dumps([{
                "description": "Result: [No data for python_version]",
                "domain": "synthesis",
                "is_atomic": True,
                "complexity": 1,
                "provides": [],
                "consumes": [],
            }]),
            success_count=1,
        )
        ps.save(broken)

        purged_count = ps.purge_polluted()

        assert purged_count == 1
        assert len(ps.load_all()) == 0

    def test_purge_detects_various_markers(self) -> None:
        """Various pollution markers (sudo, /usr/, .exe, etc.) are detected."""
        store = _make_store()
        ps = PatternStore(store)

        markers_to_test = [
            ("sudo-pattern", "sudo apt-get update"),
            ("path-pattern", "Read /usr/local/bin/config"),
            ("exe-pattern", "Run program.exe"),
            ("npm-pattern", "npm install react"),
            ("git-pattern", "git clone repository"),
            ("chmod-pattern", "chmod +x script"),
            ("bash-pattern", "Execute bash command"),
        ]

        for pid, description in markers_to_test:
            p = Pattern(
                id=pid,
                trigger="Do {slot_0}",
                description="Test",
                variable_slots=("slot_0",),
                tree_template=json.dumps([{
                    "description": description,
                    "domain": "code",
                    "is_atomic": True,
                    "complexity": 1,
                    "provides": [],
                    "consumes": [],
                }]),
                success_count=1,
            )
            ps.save(p)

        purged_count = ps.purge_polluted()

        assert purged_count == len(markers_to_test)
        assert len(ps.load_all()) == 0


# ---------------------------------------------------------------------------
# 4. Domain scoping: patterns store source domain, matcher filters
# ---------------------------------------------------------------------------


class TestDomainScoping:
    """Patterns carry source_domain; matcher only returns domain-compatible matches."""

    def test_extracted_pattern_has_source_domain(self) -> None:
        """Patterns extracted from web-domain tasks carry source_domain='web'."""
        extractor = PatternExtractor()
        nodes = _make_web_tree()
        result = _success_result()

        pattern = extractor.extract(
            task="Compare the weather in Amsterdam, London, and Berlin",
            nodes=nodes,
            result=result,
        )

        assert pattern is not None
        assert pattern.source_domain == "web"

    def test_synthesis_domain_extraction(self) -> None:
        """Pure synthesis tasks get source_domain='synthesis'."""
        extractor = PatternExtractor()
        nodes = [
            TaskNode(
                id="root",
                description="Compare pros and cons",
                children=["a", "b"],
                domain=Domain.SYNTHESIS,
            ),
            TaskNode(
                id="a",
                description="List pros of topic",
                parent_id="root",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                provides=["pros"],
            ),
            TaskNode(
                id="b",
                description="List cons of topic",
                parent_id="root",
                is_atomic=True,
                domain=Domain.SYNTHESIS,
                provides=["cons"],
            ),
        ]
        result = _success_result()

        pattern = extractor.extract(
            task="Compare pros and cons",
            nodes=nodes,
            result=result,
        )

        assert pattern is not None
        assert pattern.source_domain == "synthesis"

    def test_matcher_filters_by_domain(self) -> None:
        """When domain is specified, matcher only considers matching patterns."""
        matcher = PatternMatcher()

        web_pattern = Pattern(
            id="web-p",
            trigger="Compare {slot_0} and {slot_1}",
            description="Web comparison",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
            source_domain="web",
        )
        code_pattern = Pattern(
            id="code-p",
            trigger="Compare {slot_0} and {slot_1}",
            description="Code comparison",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
            source_domain="code",
        )

        # With domain="web", only web pattern should be considered
        result = matcher.match(
            "Compare apples and oranges",
            [web_pattern, code_pattern],
            domain="web",
        )

        assert result is not None
        matched, _ = result
        assert matched.id == "web-p"

    def test_matcher_allows_general_domain(self) -> None:
        """Patterns with source_domain='general' match any domain."""
        matcher = PatternMatcher()

        general_pattern = Pattern(
            id="gen-p",
            trigger="Compare {slot_0} and {slot_1}",
            description="General comparison",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
            source_domain="general",
        )

        result = matcher.match(
            "Compare apples and oranges",
            [general_pattern],
            domain="web",
        )

        assert result is not None
        matched, _ = result
        assert matched.id == "gen-p"

    def test_matcher_no_domain_returns_all(self) -> None:
        """When no domain is specified, all patterns are considered."""
        matcher = PatternMatcher()

        pattern = Pattern(
            id="any-p",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Multiplication",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
            source_domain="web",
        )

        result = matcher.match("Calculate 7 times 8", [pattern])

        assert result is not None
        matched, _ = result
        assert matched.id == "any-p"

    def test_domain_scoping_prevents_cross_domain_match(self) -> None:
        """A code-domain pattern is excluded when matching for web-domain tasks."""
        matcher = PatternMatcher()

        code_only = Pattern(
            id="code-only",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Code calculation",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=10,
            source_domain="code",
        )

        result = matcher.match(
            "Calculate 7 times 8",
            [code_only],
            domain="web",
        )

        assert result is None


# ---------------------------------------------------------------------------
# 5. PatternStore round-trip with source_domain
# ---------------------------------------------------------------------------


class TestPatternStoreDomain:
    """PatternStore correctly persists and loads source_domain."""

    def test_save_and_load_source_domain(self) -> None:
        """source_domain field is persisted and loaded correctly."""
        store = _make_store()
        ps = PatternStore(store)

        pattern = Pattern(
            id="domain-test",
            trigger="Do {slot_0}",
            description="Test",
            variable_slots=("slot_0",),
            tree_template="[]",
            success_count=1,
            source_domain="web",
        )
        ps.save(pattern)

        loaded = ps.load_all()
        assert len(loaded) == 1
        assert loaded[0].source_domain == "web"

    def test_default_source_domain_is_general(self) -> None:
        """Patterns without explicit source_domain default to 'general'."""
        store = _make_store()
        ps = PatternStore(store)

        pattern = Pattern(
            id="no-domain",
            trigger="Do {slot_0}",
            description="Test",
            variable_slots=("slot_0",),
            tree_template="[]",
            success_count=1,
        )
        ps.save(pattern)

        loaded = ps.load_all()
        assert len(loaded) == 1
        assert loaded[0].source_domain == "general"

    def test_delete_pattern(self) -> None:
        """PatternStore.delete() removes a single pattern by ID."""
        store = _make_store()
        ps = PatternStore(store)

        ps.save(Pattern(
            id="del-1",
            trigger="A",
            description="A",
            tree_template="[]",
            success_count=1,
        ))
        ps.save(Pattern(
            id="del-2",
            trigger="B",
            description="B",
            tree_template="[]",
            success_count=1,
        ))

        deleted = ps.delete("del-1")

        assert deleted is True
        remaining = ps.load_all()
        assert len(remaining) == 1
        assert remaining[0].id == "del-2"

    def test_delete_nonexistent_returns_false(self) -> None:
        """Deleting a nonexistent pattern returns False."""
        store = _make_store()
        ps = PatternStore(store)

        deleted = ps.delete("does-not-exist")

        assert deleted is False
