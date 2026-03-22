"""Semantic embedding layer for pattern matching.

Wraps sentence-transformers to compute embeddings for task descriptions and
pattern triggers. Embeddings enable matching semantically similar but lexically
different tasks against cached patterns.

The model (all-MiniLM-L6-v2) is lazy-loaded on first use to avoid startup
overhead. All embedding computations are cached to minimize redundant work.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Compute and cache semantic embeddings for task descriptions.

    Wraps sentence-transformers with lazy model loading and an in-memory cache.

    Args:
        model_name: The sentence-transformers model to load. Defaults to
            ``all-MiniLM-L6-v2`` (384-dimensional, fast, good quality).
        similarity_threshold: Minimum cosine similarity for an embedding
            match to be considered valid. Defaults to 0.85.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
    ) -> None:
        self._model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model: object | None = None
        self._cache: dict[str, list[float]] = {}

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model on first use.

        Subclasses (e.g. test fakes) can override this to skip loading.
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading sentence-transformers model: %s", self._model_name
            )
            self._model = SentenceTransformer(self._model_name)
            logger.info("Model loaded successfully.")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Embedding-based matching will be unavailable."
            )
            self._model = None
        except Exception:
            logger.exception("Failed to load embedding model %s", self._model_name)
            self._model = None

    def _ensure_model(self) -> bool:
        """Ensure the model is loaded. Returns True if available."""
        if self._model is None:
            self._load_model()
        return self._model is not None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors.

        Uses an in-memory cache keyed on the text content. Cache misses
        are batched and sent to the model together.

        Args:
            texts: Sequence of text strings to encode.

        Returns:
            List of embedding vectors (each a list of floats), one per
            input text. Returns zero vectors if the model is unavailable.
        """
        text_list = list(texts)
        results: list[list[float] | None] = [None] * len(text_list)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache first
        for i, text in enumerate(text_list):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Encode uncached texts
        if uncached_texts:
            if not self._ensure_model():
                # Model unavailable: return zero vectors
                dim = 384
                for i in uncached_indices:
                    results[i] = [0.0] * dim
            else:
                embeddings = self._model.encode(  # type: ignore[union-attr]
                    uncached_texts, convert_to_numpy=True
                )
                for idx, emb in zip(uncached_indices, embeddings):
                    vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    self._cache[text_list[idx]] = vec
                    results[idx] = vec

        # All entries should be filled by now
        return [r if r is not None else [0.0] * 384 for r in results]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts via their embeddings.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity in [-1.0, 1.0]. Returns 0.0 if embeddings
            cannot be computed.
        """
        vecs = self.encode([text_a, text_b])
        return self.cosine_similarity(vecs[0], vecs[1])

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either
            vector has zero norm.
        """
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def cache_key(text: str, domain: str, complexity: int) -> str:
        """Create a context-aware cache key combining text + domain + complexity.

        This ensures that the same task description in different contexts
        (different domains or complexity levels) gets distinct cache entries.

        Args:
            text: The task description or trigger text.
            domain: The task domain (e.g. "code", "web", "file").
            complexity: The task complexity level.

        Returns:
            A deterministic hash string.
        """
        raw = f"{text}|{domain}|{complexity}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
