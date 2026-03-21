"""Tests for EntityResolver -- text mention to graph entity matching."""

from __future__ import annotations

import pytest

from graph.resolver import EntityResolver
from graph.store import GraphStore


@pytest.fixture
def store() -> GraphStore:
    """Provide an initialized in-memory GraphStore with seed entities."""
    s = GraphStore(db_path=None)
    s.initialize()
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def seeded_store(store: GraphStore) -> dict[str, str]:
    """Seed the store with test entities and return a mapping of name -> id."""
    ids: dict[str, str] = {}
    ids["Lucas Duys"] = store.create_node("User", {
        "id": "user-lucas",
        "name": "Lucas Duys",
        "role": "student",
        "institution": "TU/e",
    })
    ids["GraphBot"] = store.create_node("Project", {
        "id": "proj-graphbot",
        "name": "GraphBot",
        "path": "/dev/graphbot",
        "language": "Python",
        "status": "active",
    })
    ids["OpenRouter"] = store.create_node("Service", {
        "id": "svc-openrouter",
        "name": "OpenRouter",
        "type": "LLM API gateway",
    })
    ids["Alice Smith"] = store.create_node("Contact", {
        "id": "contact-alice",
        "name": "Alice Smith",
        "relationship": "classmate",
    })
    ids["Python"] = store.create_node("Skill", {
        "id": "skill-python",
        "name": "Python",
        "description": "programming language",
    })
    ids["readme.md"] = store.create_node("File", {
        "id": "file-readme",
        "path": "readme.md",
        "type": "markdown",
        "description": "project readme file",
    })
    return ids


@pytest.fixture
def resolver(store: GraphStore, seeded_store: dict[str, str]) -> EntityResolver:
    """Provide an EntityResolver backed by the seeded store."""
    return EntityResolver(store)


class TestExactMatch:
    """Layer 1: exact normalized match."""

    def test_exact_match(self, resolver: EntityResolver, seeded_store: dict[str, str]) -> None:
        """Exact name match returns confidence 1.0."""
        results = resolver.resolve("Lucas Duys")
        assert len(results) >= 1
        top_id, top_conf = results[0]
        assert top_id == "user-lucas"
        assert top_conf == 1.0

    def test_case_insensitive_match(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """Case-insensitive matching still returns confidence 1.0."""
        results = resolver.resolve("lucas duys")
        assert len(results) >= 1
        top_id, top_conf = results[0]
        assert top_id == "user-lucas"
        assert top_conf == 1.0

    def test_whitespace_stripped(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """Leading/trailing whitespace is stripped before matching."""
        results = resolver.resolve("  GraphBot  ")
        assert len(results) >= 1
        assert results[0][0] == "proj-graphbot"
        assert results[0][1] == 1.0

    def test_file_path_match(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """File entities are matched by their path field."""
        results = resolver.resolve("readme.md")
        assert len(results) >= 1
        assert results[0][0] == "file-readme"
        assert results[0][1] == 1.0


class TestFuzzyMatch:
    """Layer 2: Levenshtein ratio matching."""

    def test_typo_match(self, resolver: EntityResolver, seeded_store: dict[str, str]) -> None:
        """A minor typo still resolves with high confidence."""
        results = resolver.resolve("Lucas Duis")
        assert len(results) >= 1
        top_id, top_conf = results[0]
        assert top_id == "user-lucas"
        assert top_conf > 0.8

    def test_fuzzy_project_name(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """Fuzzy match on project name with minor edit."""
        results = resolver.resolve("GrapBot")
        assert len(results) >= 1
        top_id, top_conf = results[0]
        assert top_id == "proj-graphbot"
        assert top_conf > 0.8


class TestKeywordMatch:
    """Layer 3: BM25-style keyword matching."""

    def test_keyword_match_description(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """Keywords from description match the entity."""
        results = resolver.resolve("programming language")
        assert len(results) >= 1
        # Python skill has description "programming language"
        matched_ids = [r[0] for r in results]
        assert "skill-python" in matched_ids
        # Find the Python result and check confidence
        for eid, conf in results:
            if eid == "skill-python":
                assert conf > 0.3
                break

    def test_partial_keyword_match(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """Partial keyword overlap still matches above threshold."""
        results = resolver.resolve("project readme")
        matched_ids = [r[0] for r in results]
        assert "file-readme" in matched_ids


class TestNoMatch:
    """Edge case: no matching entities."""

    def test_no_match_returns_empty(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """Completely unknown mention returns empty list."""
        results = resolver.resolve("xyznonexistent")
        assert results == []


class TestTopKLimiting:
    """Verify top_k parameter limits results."""

    def test_top_k_limits_results(self, store: GraphStore) -> None:
        """At most top_k results are returned."""
        # Seed many similar entities
        for i in range(20):
            store.create_node("Contact", {
                "id": f"contact-{i}",
                "name": f"Alice Variant {i}",
                "relationship": "test",
            })
        resolver = EntityResolver(store)
        results = resolver.resolve("Alice Variant", top_k=3)
        assert len(results) <= 3


class TestMultipleMatchesRanking:
    """Verify results are ranked by confidence descending."""

    def test_ranking_order(self, resolver: EntityResolver, seeded_store: dict[str, str]) -> None:
        """Results are sorted by confidence descending."""
        # "Pytho" should fuzzy-match "Python" and results should be sorted
        results = resolver.resolve("Pytho")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i][1] >= results[i + 1][1]

    def test_exact_beats_fuzzy(
        self, resolver: EntityResolver, seeded_store: dict[str, str]
    ) -> None:
        """Exact match (1.0) ranks above fuzzy match."""
        results = resolver.resolve("Python")
        assert results[0][0] == "skill-python"
        assert results[0][1] == 1.0


class TestAccuracyBenchmark:
    """Accuracy benchmark: >= 90% (45/50) on mixed mention pairs."""

    def test_accuracy_over_90_percent(self, store: GraphStore) -> None:
        """Resolve >= 45 out of 50 mention-entity pairs correctly (top-3)."""
        # Seed a richer graph
        store.create_node("User", {"id": "u-lucas", "name": "Lucas Duys", "role": "student"})
        store.create_node("User", {"id": "u-bob", "name": "Bob Johnson", "role": "engineer"})
        store.create_node("Project", {
            "id": "p-graphbot", "name": "GraphBot", "path": "/dev/graphbot",
            "language": "Python", "status": "active",
        })
        store.create_node("Project", {
            "id": "p-pitchr", "name": "Pitchr", "path": "/dev/pitchr",
            "language": "TypeScript", "status": "active",
        })
        store.create_node("Service", {
            "id": "s-openrouter", "name": "OpenRouter", "type": "API gateway",
        })
        store.create_node("Service", {
            "id": "s-github", "name": "GitHub", "type": "VCS",
        })
        store.create_node("Service", {
            "id": "s-supabase", "name": "Supabase", "type": "BaaS",
        })
        store.create_node("Contact", {
            "id": "c-alice", "name": "Alice Smith", "relationship": "classmate",
        })
        store.create_node("Contact", {
            "id": "c-charlie", "name": "Charlie Brown", "relationship": "colleague",
        })
        store.create_node("Contact", {
            "id": "c-diana", "name": "Diana Prince", "relationship": "mentor",
        })
        store.create_node("Skill", {
            "id": "sk-python", "name": "Python", "description": "programming language",
        })
        store.create_node("Skill", {
            "id": "sk-rust", "name": "Rust", "description": "systems programming language",
        })
        store.create_node("Skill", {
            "id": "sk-ts", "name": "TypeScript", "description": "typed JavaScript superset",
        })
        store.create_node("Skill", {
            "id": "sk-docker", "name": "Docker", "description": "container platform",
        })
        store.create_node("File", {
            "id": "f-readme", "path": "README.md", "type": "markdown",
            "description": "project documentation",
        })
        store.create_node("File", {
            "id": "f-main", "path": "main.py", "type": "python",
            "description": "application entry point",
        })

        # 50 mention-expected pairs: mix of exact, case-insensitive, fuzzy, keyword
        pairs: list[tuple[str, str]] = [
            # Exact matches (15)
            ("Lucas Duys", "u-lucas"),
            ("Bob Johnson", "u-bob"),
            ("GraphBot", "p-graphbot"),
            ("Pitchr", "p-pitchr"),
            ("OpenRouter", "s-openrouter"),
            ("GitHub", "s-github"),
            ("Supabase", "s-supabase"),
            ("Alice Smith", "c-alice"),
            ("Charlie Brown", "c-charlie"),
            ("Diana Prince", "c-diana"),
            ("Python", "sk-python"),
            ("Rust", "sk-rust"),
            ("TypeScript", "sk-ts"),
            ("Docker", "sk-docker"),
            ("README.md", "f-readme"),
            # Case-insensitive matches (10)
            ("lucas duys", "u-lucas"),
            ("bob johnson", "u-bob"),
            ("graphbot", "p-graphbot"),
            ("pitchr", "p-pitchr"),
            ("openrouter", "s-openrouter"),
            ("github", "s-github"),
            ("supabase", "s-supabase"),
            ("alice smith", "c-alice"),
            ("python", "sk-python"),
            ("docker", "sk-docker"),
            # Fuzzy matches -- typos (15)
            ("Lucas Duis", "u-lucas"),
            ("Bob Johnsen", "u-bob"),
            ("GrapBot", "p-graphbot"),
            ("Pitchr", "p-pitchr"),
            ("OpenRoutr", "s-openrouter"),
            ("GitHb", "s-github"),
            ("Supabse", "s-supabase"),
            ("Alice Smth", "c-alice"),
            ("Charlie Bown", "c-charlie"),
            ("Diana Princ", "c-diana"),
            ("Pythn", "sk-python"),
            ("Rast", "sk-rust"),
            ("TypeScrpt", "sk-ts"),
            ("Dokcer", "sk-docker"),
            ("main.p", "f-main"),
            # Keyword matches (10)
            ("programming language", "sk-python"),
            ("systems programming", "sk-rust"),
            ("typed JavaScript", "sk-ts"),
            ("container platform", "sk-docker"),
            ("project documentation", "f-readme"),
            ("application entry", "f-main"),
            ("API gateway", "s-openrouter"),
            ("classmate", "c-alice"),
            ("colleague", "c-charlie"),
            ("mentor", "c-diana"),
        ]

        assert len(pairs) == 50

        resolver = EntityResolver(store)
        correct = 0
        for mention, expected_id in pairs:
            results = resolver.resolve(mention, top_k=3)
            result_ids = [r[0] for r in results]
            if expected_id in result_ids:
                correct += 1

        assert correct >= 45, f"Accuracy {correct}/50 ({correct * 2}%) is below 90% threshold"
