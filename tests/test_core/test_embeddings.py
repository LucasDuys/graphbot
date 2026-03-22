"""Tests for the semantic embedding layer and embedding-based pattern matching.

Uses a fake embedding service to avoid downloading the actual model in tests.
Tests verify:
- EmbeddingService interface and lazy loading behavior
- Embedding-based similarity scoring in PatternMatcher._score_match()
- Semantically similar but lexically different tasks match
- Similarity threshold filtering
- Context-aware cache keys combining task + domain + complexity
- Score combination: final_score = max(regex_score, levenshtein_score, embedding_score)
"""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock, patch

import numpy as np

from core_gb.embeddings import EmbeddingService
from core_gb.patterns import PatternMatcher
from core_gb.types import Pattern


# ---------------------------------------------------------------------------
# Fake embedding service for deterministic, model-free testing
# ---------------------------------------------------------------------------

class FakeEmbeddingService(EmbeddingService):
    """A fake embedding service that returns pre-configured embeddings.

    Maps text strings to pre-defined vectors. Unknown texts get a zero vector.
    """

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        super().__init__(model_name="fake-model")
        self._mapping: dict[str, list[float]] = mapping or {}
        self._encode_calls: list[list[str]] = []

    def _load_model(self) -> None:
        """No-op: fake service needs no real model."""
        pass

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return pre-configured embeddings for the given texts."""
        text_list = list(texts)
        self._encode_calls.append(text_list)
        results: list[list[float]] = []
        dim = 384  # Match all-MiniLM-L6-v2 dimensionality
        for text in text_list:
            if text in self._mapping:
                results.append(self._mapping[text])
            else:
                results.append([0.0] * dim)
        return results


def _normalized_vector(values: list[float], dim: int = 384) -> list[float]:
    """Create a normalized vector from a seed, padded to dim dimensions."""
    vec = values + [0.0] * (dim - len(values))
    arr = np.array(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def _similar_vectors(dim: int = 384) -> tuple[list[float], list[float]]:
    """Return two vectors with high cosine similarity (~0.95+)."""
    base = [1.0, 0.5, 0.3, 0.2] + [0.1] * (dim - 4)
    variant = [1.0, 0.5, 0.3, 0.25] + [0.1] * (dim - 4)
    return _normalized_vector(base, dim), _normalized_vector(variant, dim)


def _dissimilar_vectors(dim: int = 384) -> tuple[list[float], list[float]]:
    """Return two vectors with low cosine similarity (~0.0)."""
    a = [1.0, 0.0, 0.0, 0.0] + [0.0] * (dim - 4)
    b = [0.0, 1.0, 0.0, 0.0] + [0.0] * (dim - 4)
    return _normalized_vector(a, dim), _normalized_vector(b, dim)


# ---------------------------------------------------------------------------
# Tests for EmbeddingService itself
# ---------------------------------------------------------------------------

class TestEmbeddingService:

    def test_lazy_load_not_called_on_init(self) -> None:
        """Model should not be loaded at construction time."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        assert service._model is None

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Cosine similarity of identical normalized vectors should be ~1.0."""
        vec = _normalized_vector([1.0, 2.0, 3.0])
        sim = EmbeddingService.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Cosine similarity of orthogonal vectors should be ~0.0."""
        a, b = _dissimilar_vectors()
        sim = EmbeddingService.cosine_similarity(a, b)
        assert abs(sim) < 0.05

    def test_cosine_similarity_similar_vectors(self) -> None:
        """Cosine similarity of near-identical vectors should be high."""
        a, b = _similar_vectors()
        sim = EmbeddingService.cosine_similarity(a, b)
        assert sim > 0.90

    def test_cache_key_includes_domain_and_complexity(self) -> None:
        """Context-aware cache keys must combine text + domain + complexity."""
        key1 = EmbeddingService.cache_key("deploy the app", "code", 3)
        key2 = EmbeddingService.cache_key("deploy the app", "web", 3)
        key3 = EmbeddingService.cache_key("deploy the app", "code", 5)
        # Same text, different domain/complexity -> different keys
        assert key1 != key2
        assert key1 != key3
        # Same everything -> same key
        assert key1 == EmbeddingService.cache_key("deploy the app", "code", 3)

    def test_default_similarity_threshold(self) -> None:
        """Default embedding similarity threshold should be 0.85."""
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        assert service.similarity_threshold == 0.85

    def test_configurable_similarity_threshold(self) -> None:
        """Similarity threshold should be configurable at init."""
        service = EmbeddingService(
            model_name="all-MiniLM-L6-v2", similarity_threshold=0.7
        )
        assert service.similarity_threshold == 0.7


# ---------------------------------------------------------------------------
# Tests for embedding-based matching in PatternMatcher
# ---------------------------------------------------------------------------

class TestEmbeddingPatternMatching:

    def test_semantically_similar_lexically_different_match(self) -> None:
        """Tasks that mean the same thing but use different words should match
        via embedding similarity when regex and Levenshtein fail."""
        # "What is the weather forecast for tomorrow" vs trigger
        # "Get tomorrow's weather prediction" -- lexically very different
        task_text = "What is the weather forecast for tomorrow"
        trigger_text = "Get tomorrow's weather prediction"

        vec_a, vec_b = _similar_vectors()
        fake_service = FakeEmbeddingService(mapping={
            task_text.lower(): vec_a,
            trigger_text.lower(): vec_b,
        })

        matcher = PatternMatcher(embedding_service=fake_service)
        pattern = Pattern(
            id="semantic-1",
            trigger=trigger_text,
            description="Tomorrow weather",
            variable_slots=(),
            tree_template="[]",
            success_count=5,
        )

        result = matcher.match(task_text, [pattern], threshold=0.7)
        assert result is not None
        matched, _ = result
        assert matched.id == "semantic-1"

    def test_semantically_dissimilar_no_match(self) -> None:
        """Tasks that are semantically unrelated should not match."""
        task_text = "Deploy the application to production"
        trigger_text = "Make a cup of coffee"

        vec_a, vec_b = _dissimilar_vectors()
        fake_service = FakeEmbeddingService(mapping={
            task_text.lower(): vec_a,
            trigger_text.lower(): vec_b,
        })

        matcher = PatternMatcher(embedding_service=fake_service)
        pattern = Pattern(
            id="unrelated",
            trigger=trigger_text,
            description="Coffee",
            variable_slots=(),
            tree_template="[]",
            success_count=5,
        )

        result = matcher.match(task_text, [pattern], threshold=0.7)
        assert result is None

    def test_embedding_threshold_filtering(self) -> None:
        """Embedding scores below the configurable threshold are rejected."""
        task_text = "analyze the dataset"
        trigger_text = "process the data"

        # Create vectors with moderate similarity (~0.7)
        base = [1.0, 0.5, 0.3] + [0.0] * 381
        variant = [0.7, 0.6, 0.1] + [0.0] * 381
        vec_a = _normalized_vector(base)
        vec_b = _normalized_vector(variant)
        sim = EmbeddingService.cosine_similarity(vec_a, vec_b)

        fake_service = FakeEmbeddingService(mapping={
            task_text.lower(): vec_a,
            trigger_text.lower(): vec_b,
        })
        # Set embedding threshold above the actual similarity
        fake_service.similarity_threshold = sim + 0.1

        matcher = PatternMatcher(embedding_service=fake_service)
        pattern = Pattern(
            id="moderate",
            trigger=trigger_text,
            description="Data processing",
            variable_slots=(),
            tree_template="[]",
            success_count=5,
        )

        # The embedding score will be below the embedding threshold,
        # so it should be clamped to 0.0, and Levenshtein will also be low
        result = matcher.match(task_text, [pattern], threshold=0.7)
        assert result is None

    def test_score_combination_max_of_three(self) -> None:
        """Final score should be max(regex, levenshtein, embedding).

        If regex matches perfectly (1.0), embedding does not lower the score."""
        task_text = "Calculate 7 times 8"
        trigger_text = "Calculate {slot_0} times {slot_1}"

        # Even with low embedding similarity, regex match should dominate
        vec_a, vec_b = _dissimilar_vectors()
        fake_service = FakeEmbeddingService(mapping={
            task_text.lower(): vec_a,
            trigger_text.lower(): vec_b,
        })

        matcher = PatternMatcher(embedding_service=fake_service)
        pattern = Pattern(
            id="regex-wins",
            trigger=trigger_text,
            description="Multiplication",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
        )

        result = matcher.match(task_text, [pattern])
        assert result is not None
        matched, bindings = result
        assert matched.id == "regex-wins"
        assert bindings["slot_0"] == "7"
        assert bindings["slot_1"] == "8"

    def test_matcher_works_without_embedding_service(self) -> None:
        """PatternMatcher should work normally when no embedding service is set.

        This ensures backward compatibility with existing code."""
        matcher = PatternMatcher()
        pattern = Pattern(
            id="no-embed",
            trigger="Calculate {slot_0} times {slot_1}",
            description="Multiplication",
            variable_slots=("slot_0", "slot_1"),
            tree_template="[]",
            success_count=5,
        )

        result = matcher.match("Calculate 7 times 8", [pattern])
        assert result is not None
        matched, bindings = result
        assert matched.id == "no-embed"

    def test_embedding_best_of_multiple_patterns(self) -> None:
        """When multiple patterns exist, the one with highest embedding
        similarity should win (when regex and Levenshtein do not match)."""
        task_text = "send a notification to the team"

        vec_task = _normalized_vector([1.0, 0.8, 0.6, 0.4])
        vec_close = _normalized_vector([0.95, 0.85, 0.55, 0.45])
        vec_far = _normalized_vector([0.0, 0.0, 1.0, 0.0])

        fake_service = FakeEmbeddingService(mapping={
            task_text.lower(): vec_task,
            "alert the group members".lower(): vec_close,
            "compile the source code".lower(): vec_far,
        })

        matcher = PatternMatcher(embedding_service=fake_service)

        close_pattern = Pattern(
            id="close",
            trigger="alert the group members",
            description="Notification",
            variable_slots=(),
            tree_template="[]",
            success_count=5,
        )
        far_pattern = Pattern(
            id="far",
            trigger="compile the source code",
            description="Compile",
            variable_slots=(),
            tree_template="[]",
            success_count=5,
        )

        result = matcher.match(task_text, [far_pattern, close_pattern], threshold=0.5)
        assert result is not None
        matched, _ = result
        assert matched.id == "close"

    def test_embedding_score_with_slots_uses_structural_text(self) -> None:
        """When computing embedding similarity for a pattern with slots,
        the structural text (slots stripped) should be used."""
        task_text = "summarize the quarterly report"
        structural_text = "summarize the  report"  # after stripping {slot_0}

        vec_task = _normalized_vector([1.0, 0.7, 0.3])
        vec_structural = _normalized_vector([0.95, 0.75, 0.35])

        fake_service = FakeEmbeddingService(mapping={
            task_text.lower(): vec_task,
            structural_text.lower(): vec_structural,
            # Also map the collapsed version without double spaces
            "summarize the report": vec_structural,
        })

        matcher = PatternMatcher(embedding_service=fake_service)
        pattern = Pattern(
            id="slotted",
            trigger="summarize the {slot_0} report",
            description="Report summary",
            variable_slots=("slot_0",),
            tree_template="[]",
            success_count=5,
        )

        score, _ = matcher._score_match(task_text, pattern)
        # Score should incorporate embedding similarity
        # The exact value depends on whether Levenshtein or embedding is higher
        assert score > 0.5

    def test_success_rate_weighting_still_applies_with_embeddings(self) -> None:
        """Embedding scores are still weighted by success rate factor."""
        task_text = "notify the team about the deployment"
        trigger_text = "alert the group about the release"

        vec_a, vec_b = _similar_vectors()
        fake_service = FakeEmbeddingService(mapping={
            task_text.lower(): vec_a,
            trigger_text.lower(): vec_b,
        })

        matcher = PatternMatcher(embedding_service=fake_service)
        # Pattern with terrible success rate
        pattern = Pattern(
            id="bad-rate",
            trigger=trigger_text,
            description="Notification",
            variable_slots=(),
            tree_template="[]",
            success_count=0,
            failure_count=10,
        )

        # 0% success rate should be skipped entirely
        result = matcher.match(task_text, [pattern], threshold=0.5)
        assert result is None
