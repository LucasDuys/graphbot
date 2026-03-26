"""Lightweight prompt compression using TF-IDF sentence scoring.

Compresses context sections by removing low-information sentences while
retaining high-density content. Uses pure-Python TF-IDF (no external
dependencies) to score sentence importance.

Target: 2-4x compression on context sections with <5% quality loss.
Only compresses when context exceeds 50% of token budget.
"""

from __future__ import annotations

import math
import re
from collections import Counter


# Minimum token count below which compression is skipped entirely.
_MIN_TOKENS_FOR_COMPRESSION: int = 100

# Sentence boundary pattern: split on period, exclamation, question mark
# followed by whitespace or end-of-string, but not on common abbreviations.
_SENTENCE_BOUNDARY: re.Pattern[str] = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)


class PromptCompressor:
    """Compresses prompt context sections using TF-IDF sentence scoring.

    Sentences are scored by the average TF-IDF weight of their words.
    Top-scoring sentences are kept (in original order) until the target
    token count is reached.

    Args:
        min_tokens: Minimum estimated token count before compression
            is applied. Texts shorter than this are returned unchanged.
    """

    def __init__(self, min_tokens: int = _MIN_TOKENS_FOR_COMPRESSION) -> None:
        self.min_tokens: int = min_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_compress(self, text: str, token_budget: int) -> bool:
        """Return True when *text* exceeds 50% of *token_budget*.

        Args:
            text: The context text to evaluate.
            token_budget: Total token budget for the call.

        Returns:
            True if the text should be compressed.
        """
        if token_budget <= 0:
            return False
        estimated = self._estimate_tokens(text)
        return estimated > token_budget * 0.5

    def compress(
        self,
        text: str,
        target_ratio: float = 0.5,
        token_budget: int | None = None,
    ) -> str:
        """Compress *text* by keeping the most informative sentences.

        Args:
            text: The context text to compress.
            target_ratio: Fraction of original tokens to retain (0.0-1.0).
                A value of 0.5 means keep roughly half the tokens (~2x
                compression). Defaults to 0.5.
            token_budget: Optional total token budget. When provided,
                compression is skipped if the text is under 50% of
                budget. When ``None``, only *min_tokens* is checked.

        Returns:
            Compressed text with high-information sentences preserved in
            their original order, or the original text if compression
            is not warranted.
        """
        if not text or not text.strip():
            return text

        estimated = self._estimate_tokens(text)

        # Skip compression for short texts.
        if estimated < self.min_tokens:
            return text

        # Skip if text is under 50% of budget (when budget is given).
        if token_budget is not None and not self.should_compress(text, token_budget):
            return text

        sentences = self._split_sentences(text)

        # Nothing to compress if only one sentence.
        if len(sentences) <= 1:
            return text

        tfidf = self._compute_tfidf(sentences)
        scored = self._score_sentences(sentences, tfidf)

        # Determine how many tokens to keep.
        target_tokens = max(1, int(estimated * target_ratio))

        # Select top sentences by score, preserving original order.
        # Build (original_index, sentence, score) tuples.
        indexed_scored: list[tuple[int, str, float]] = [
            (i, sentence, score)
            for i, (sentence, score) in enumerate(scored)
        ]

        # Sort by score descending to pick best sentences first.
        by_score = sorted(indexed_scored, key=lambda t: t[2], reverse=True)

        kept_indices: list[int] = []
        accumulated_tokens: int = 0

        for idx, sentence, _score in by_score:
            sentence_tokens = self._estimate_tokens(sentence)
            if accumulated_tokens + sentence_tokens > target_tokens:
                # If we have nothing yet, include at least the top sentence.
                if not kept_indices:
                    kept_indices.append(idx)
                    accumulated_tokens += sentence_tokens
                break
            kept_indices.append(idx)
            accumulated_tokens += sentence_tokens

        # Restore original order.
        kept_indices.sort()

        # Reconstruct text.
        compressed_parts = [sentences[i] for i in kept_indices]
        return " ".join(compressed_parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using word_count / 0.75 heuristic.

        Matches the estimation used by :class:`core_gb.token_budget.TokenBudget`.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count (rounded up).
        """
        if not text or not text.strip():
            return 0
        word_count = len(text.split())
        return math.ceil(word_count / 0.75)

    def _split_sentences(self, text: str) -> list[str]:
        """Split *text* into sentences at sentence boundaries.

        Uses a regex that splits after sentence-ending punctuation
        followed by whitespace and an uppercase letter. Falls back
        to treating the entire text as a single sentence when no
        boundary is found.

        Args:
            text: Text to split.

        Returns:
            List of sentence strings (stripped of leading/trailing
            whitespace, empty strings removed).
        """
        parts = _SENTENCE_BOUNDARY.split(text.strip())
        return [s.strip() for s in parts if s.strip()]

    def _compute_tfidf(self, sentences: list[str]) -> dict[str, float]:
        """Compute TF-IDF scores for all words across *sentences*.

        TF (term frequency) is computed per-document (sentence).
        IDF (inverse document frequency) is computed across all sentences.
        The returned score for each word is its maximum TF-IDF across
        all sentences, giving a corpus-level importance weight.

        Args:
            sentences: List of sentence strings.

        Returns:
            Mapping from lowercase word to its TF-IDF score.
        """
        n_docs = len(sentences)
        if n_docs == 0:
            return {}

        # Tokenize each sentence into lowercase words.
        tokenized: list[list[str]] = [
            self._tokenize(s) for s in sentences
        ]

        # Document frequency: number of sentences containing each word.
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized:
            unique = set(tokens)
            for word in unique:
                doc_freq[word] += 1

        # Compute TF-IDF: for each word, take the max TF-IDF across sentences.
        tfidf: dict[str, float] = {}

        for tokens in tokenized:
            if not tokens:
                continue
            tf_counts = Counter(tokens)
            max_count = max(tf_counts.values())
            for word, count in tf_counts.items():
                tf = count / max_count  # Normalized TF.
                idf = math.log((1 + n_docs) / (1 + doc_freq[word])) + 1
                score = tf * idf
                if word not in tfidf or score > tfidf[word]:
                    tfidf[word] = score

        return tfidf

    def _score_sentences(
        self,
        sentences: list[str],
        tfidf: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Score each sentence by average TF-IDF of its words.

        Args:
            sentences: List of sentence strings.
            tfidf: TF-IDF mapping from :meth:`_compute_tfidf`.

        Returns:
            List of ``(sentence, score)`` tuples in the same order
            as the input *sentences*.
        """
        results: list[tuple[str, float]] = []
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            if not tokens:
                results.append((sentence, 0.0))
                continue
            total = sum(tfidf.get(t, 0.0) for t in tokens)
            avg_score = total / len(tokens)
            results.append((sentence, avg_score))
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize *text* into lowercase alphabetic words.

        Strips punctuation and digits, keeping only alphabetic tokens
        of length >= 2 (to filter noise).

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase word strings.
        """
        return [
            w.lower()
            for w in re.findall(r"[a-zA-Z]+", text)
            if len(w) >= 2
        ]
