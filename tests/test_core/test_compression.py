"""Tests for TF-IDF prompt compression."""

from core_gb.compression import PromptCompressor


# -- Helpers ---------------------------------------------------------------

def _long_context() -> str:
    """Generate a multi-sentence context block with clear key facts.

    Contains a mix of high-information sentences (specific facts,
    names, numbers) and low-information filler sentences.
    """
    return (
        "GraphBot is a recursive DAG execution engine built on Kuzu. "
        "It decomposes complex tasks into trivially simple subtasks. "
        "The system processes requests through a seven-stage pipeline. "
        "This is a general statement that does not add much value. "
        "Another filler sentence that restates obvious information. "
        "Parallel execution on free LLMs achieves 7x throughput gains. "
        "The temporal knowledge graph stores entities with time-decayed relevance. "
        "Things happen and stuff occurs in the normal course of operations. "
        "Pattern matching eliminates redundant inference by reusing prior results. "
        "The decomposer uses topological sorting to schedule dependent nodes. "
        "In summary this section covered various topics about the system. "
        "Entity resolution merges duplicate nodes using embedding similarity."
    )


def _key_facts() -> list[str]:
    """Key facts that should survive compression (domain-specific terms).

    These are spread across multiple high-information sentences.
    We check that a reasonable fraction survive -- not all, since
    compression necessarily drops some content.
    """
    return [
        "temporal knowledge graph",
        "entity resolution",
        "parallel execution",
        "pattern matching",
        "decompos",  # decompose / decomposes
        "topological sorting",
        "embedding similarity",
    ]


# -- Basic compression behavior -------------------------------------------

class TestBasicCompression:
    """Compression reduces token count on long text."""

    def test_compress_reduces_tokens(self) -> None:
        compressor = PromptCompressor()
        text = _long_context()
        result = compressor.compress(text, target_ratio=0.5)
        original_tokens = compressor._estimate_tokens(text)
        compressed_tokens = compressor._estimate_tokens(result)
        assert compressed_tokens < original_tokens

    def test_compression_ratio_within_range(self) -> None:
        """Compressed output is 2-4x smaller than original."""
        compressor = PromptCompressor()
        text = _long_context()
        result = compressor.compress(text, target_ratio=0.35)
        original_tokens = compressor._estimate_tokens(text)
        compressed_tokens = compressor._estimate_tokens(result)
        ratio = original_tokens / max(1, compressed_tokens)
        assert 1.5 <= ratio <= 5.0, f"Compression ratio {ratio:.1f}x outside 1.5-5x range"

    def test_key_facts_retained(self) -> None:
        """Important factual content survives compression."""
        compressor = PromptCompressor()
        text = _long_context()
        result = compressor.compress(text, target_ratio=0.5)
        facts = _key_facts()
        retained = sum(1 for f in facts if f.lower() in result.lower())
        # At least ~40% of key facts should survive at 50% compression.
        min_retained = max(2, int(len(facts) * 0.4))
        assert retained >= min_retained, (
            f"Only {retained}/{len(facts)} key facts retained "
            f"(need {min_retained}): {result}"
        )


# -- Short text is not compressed ------------------------------------------

class TestShortTextSkipped:
    """Text under the minimum token threshold is returned unchanged."""

    def test_short_text_unchanged(self) -> None:
        compressor = PromptCompressor(min_tokens=100)
        short = "This is a brief context."
        result = compressor.compress(short, target_ratio=0.5)
        assert result == short

    def test_empty_string(self) -> None:
        compressor = PromptCompressor()
        assert compressor.compress("") == ""

    def test_whitespace_only(self) -> None:
        compressor = PromptCompressor()
        assert compressor.compress("   ") == "   "

    def test_single_sentence_unchanged(self) -> None:
        """A single sentence cannot be split further."""
        compressor = PromptCompressor(min_tokens=5)
        sentence = "The quick brown fox jumps over the lazy dog."
        result = compressor.compress(sentence, target_ratio=0.5)
        assert result == sentence


# -- should_compress threshold ---------------------------------------------

class TestShouldCompress:
    """should_compress returns True only when text > 50% of budget."""

    def test_under_half_budget(self) -> None:
        compressor = PromptCompressor()
        # 10 words -> ~14 tokens, budget 100 -> 14 < 50
        short = "one two three four five six seven eight nine ten"
        assert compressor.should_compress(short, token_budget=100) is False

    def test_over_half_budget(self) -> None:
        compressor = PromptCompressor()
        text = _long_context()
        # Long context well over 50 tokens; budget of 100 is tiny.
        assert compressor.should_compress(text, token_budget=100) is True

    def test_zero_budget(self) -> None:
        compressor = PromptCompressor()
        assert compressor.should_compress("any text", token_budget=0) is False

    def test_compress_respects_budget_threshold(self) -> None:
        """When token_budget is provided and text is under 50%, skip."""
        compressor = PromptCompressor(min_tokens=5)
        short = "one two three four five six seven eight nine ten"
        result = compressor.compress(short, target_ratio=0.5, token_budget=10000)
        assert result == short


# -- Sentence ordering preserved -------------------------------------------

class TestSentenceOrdering:
    """Compressed output preserves original sentence order."""

    def test_order_preserved(self) -> None:
        compressor = PromptCompressor(min_tokens=5)
        text = (
            "Alpha is the first letter. "
            "Beta is the second letter. "
            "Gamma is the third letter. "
            "Delta is the fourth letter. "
            "Epsilon is the fifth letter. "
            "Zeta is the sixth letter."
        )
        result = compressor.compress(text, target_ratio=0.5)
        # Extract which Greek letters remain and verify order.
        letters = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
        present = [l for l in letters if l in result]
        assert len(present) >= 2, "Should keep at least 2 sentences"
        # Verify order: each retained letter appears after the previous one.
        positions = [result.index(l) for l in present]
        assert positions == sorted(positions), "Sentence order not preserved"


# -- TF-IDF scoring --------------------------------------------------------

class TestTfidfScoring:
    """TF-IDF scoring produces reasonable importance rankings."""

    def test_unique_terms_score_higher(self) -> None:
        """Words appearing in fewer sentences get higher IDF -> higher score."""
        compressor = PromptCompressor()
        sentences = [
            "The system processes data quickly.",
            "The system handles requests efficiently.",
            "Kuzu graph database enables temporal entity resolution.",
        ]
        tfidf = compressor._compute_tfidf(sentences)
        # "kuzu" appears in 1 sentence; "system" appears in 2.
        assert tfidf.get("kuzu", 0) > tfidf.get("system", 0), (
            f"kuzu ({tfidf.get('kuzu', 0):.3f}) should outscore "
            f"system ({tfidf.get('system', 0):.3f})"
        )

    def test_sentence_with_unique_terms_scores_higher(self) -> None:
        """Sentences with rare, specific terms get higher scores."""
        compressor = PromptCompressor()
        sentences = [
            "Things happen in the system.",
            "Things occur during normal operations.",
            "Kuzu graph database stores temporal entities with decay.",
        ]
        tfidf = compressor._compute_tfidf(sentences)
        scored = compressor._score_sentences(sentences, tfidf)
        # The Kuzu sentence should have the highest score.
        scores = [s for _, s in scored]
        assert scores[2] == max(scores), (
            f"Kuzu sentence score {scores[2]:.3f} should be highest, "
            f"got scores {scores}"
        )

    def test_empty_sentences_score_zero(self) -> None:
        compressor = PromptCompressor()
        scored = compressor._score_sentences(["", "hello world"], {})
        assert scored[0][1] == 0.0


# -- Token estimation consistency -----------------------------------------

class TestTokenEstimation:
    """Token estimation matches TokenBudget heuristic."""

    def test_empty(self) -> None:
        compressor = PromptCompressor()
        assert compressor._estimate_tokens("") == 0

    def test_known_text(self) -> None:
        compressor = PromptCompressor()
        text = "The quick brown fox jumps over the lazy dog"
        # 9 words / 0.75 = 12.0
        assert compressor._estimate_tokens(text) == 12
