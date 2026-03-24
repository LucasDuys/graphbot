"""Tests for the token budget enforcer."""

from core_gb.token_budget import TokenBudget


class TestTokenEstimation:
    """Token estimation uses word_count / 0.75 heuristic."""

    def test_empty_string(self) -> None:
        budget = TokenBudget()
        assert budget.estimate_tokens("") == 0

    def test_single_word(self) -> None:
        budget = TokenBudget()
        # 1 word / 0.75 = 1.33 -> ceil to 2
        assert budget.estimate_tokens("hello") == 2

    def test_known_sentence(self) -> None:
        budget = TokenBudget()
        text = "The quick brown fox jumps over the lazy dog"
        # 9 words / 0.75 = 12.0
        assert budget.estimate_tokens(text) == 12

    def test_multiline_text(self) -> None:
        budget = TokenBudget()
        text = "line one\nline two\nline three"
        # 6 words / 0.75 = 8.0
        assert budget.estimate_tokens(text) == 8


class TestBudgetDefaults:
    """Default budget for 8B models is 4096 tokens."""

    def test_default_max_tokens(self) -> None:
        budget = TokenBudget()
        assert budget.max_tokens == 4096

    def test_custom_max_tokens(self) -> None:
        budget = TokenBudget(max_tokens=8192)
        assert budget.max_tokens == 8192

    def test_reserved_tokens(self) -> None:
        budget = TokenBudget()
        assert budget.system_prompt_reserve == 100
        assert budget.user_message_reserve == 200
        assert budget.response_reserve == 500

    def test_available_budget(self) -> None:
        budget = TokenBudget()
        # 4096 - 100 - 200 - 500 = 3296
        assert budget.available_budget == 3296

    def test_custom_reserves(self) -> None:
        budget = TokenBudget(
            max_tokens=2048,
            system_prompt_reserve=50,
            user_message_reserve=100,
            response_reserve=300,
        )
        assert budget.available_budget == 2048 - 50 - 100 - 300


class TestPriorityOrder:
    """Sections are allocated by priority: conversation > reflections > entities > patterns."""

    def test_all_sections_fit(self) -> None:
        budget = TokenBudget(max_tokens=10000)
        sections = {
            "conversation": "short conversation text",
            "reflections": "short reflections text",
            "entities": "short entities text",
            "patterns": "short patterns text",
        }
        result = budget.trim_to_budget(sections)
        assert "conversation" in result
        assert "reflections" in result
        assert "entities" in result
        assert "patterns" in result
        # All content preserved when budget is large
        assert result["conversation"] == sections["conversation"]
        assert result["reflections"] == sections["reflections"]
        assert result["entities"] == sections["entities"]
        assert result["patterns"] == sections["patterns"]

    def test_lowest_priority_trimmed_first(self) -> None:
        # Tiny budget: only room for high-priority sections
        budget = TokenBudget(
            max_tokens=850,
            system_prompt_reserve=100,
            user_message_reserve=200,
            response_reserve=500,
        )
        # available = 50 tokens
        # conversation ~ 4 words / 0.75 ~ 6 tokens
        # reflections ~ 4 words / 0.75 ~ 6 tokens
        # entities ~ 4 words / 0.75 ~ 6 tokens
        # patterns = 100 words / 0.75 ~ 134 tokens (too large)
        sections = {
            "conversation": "one two three four",
            "reflections": "five six seven eight",
            "entities": "nine ten eleven twelve",
            "patterns": " ".join(["word"] * 100),
        }
        result = budget.trim_to_budget(sections)
        # Patterns (lowest priority) should be dropped first
        assert "conversation" in result
        assert result.get("patterns", "") == ""

    def test_multiple_sections_trimmed(self) -> None:
        budget = TokenBudget(
            max_tokens=810,
            system_prompt_reserve=100,
            user_message_reserve=200,
            response_reserve=500,
        )
        # available = 10 tokens
        sections = {
            "conversation": "a b c",  # 3 words / 0.75 = 4 tokens
            "reflections": " ".join(["word"] * 50),  # way too big
            "entities": " ".join(["word"] * 50),  # way too big
            "patterns": " ".join(["word"] * 50),  # way too big
        }
        result = budget.trim_to_budget(sections)
        # Conversation (highest priority) should survive
        assert result["conversation"] == "a b c"
        # Lower priority sections should be trimmed/dropped
        assert result.get("patterns", "") == ""

    def test_empty_sections(self) -> None:
        budget = TokenBudget()
        result = budget.trim_to_budget({})
        assert result == {}

    def test_unknown_section_treated_as_lowest_priority(self) -> None:
        budget = TokenBudget(max_tokens=10000)
        sections = {
            "conversation": "hello",
            "custom_section": "some custom data",
        }
        result = budget.trim_to_budget(sections)
        assert "conversation" in result
        assert "custom_section" in result

    def test_unknown_section_dropped_before_known(self) -> None:
        budget = TokenBudget(
            max_tokens=810,
            system_prompt_reserve=100,
            user_message_reserve=200,
            response_reserve=500,
        )
        # available = 10 tokens
        sections = {
            "conversation": "a b",  # 2 words / 0.75 ~ 3 tokens
            "custom_section": " ".join(["word"] * 100),  # huge
        }
        result = budget.trim_to_budget(sections)
        assert result["conversation"] == "a b"
        assert result.get("custom_section", "") == ""


class TestTrimBehavior:
    """Trimming removes entire lowest-priority sections first."""

    def test_returns_dict_type(self) -> None:
        budget = TokenBudget()
        result = budget.trim_to_budget({"conversation": "hello"})
        assert isinstance(result, dict)

    def test_does_not_mutate_input(self) -> None:
        budget = TokenBudget()
        sections = {"conversation": "hello", "patterns": "world"}
        original = dict(sections)
        budget.trim_to_budget(sections)
        assert sections == original

    def test_zero_available_budget(self) -> None:
        budget = TokenBudget(
            max_tokens=800,
            system_prompt_reserve=100,
            user_message_reserve=200,
            response_reserve=500,
        )
        # available = 0
        sections = {"conversation": "hello world"}
        result = budget.trim_to_budget(sections)
        assert result.get("conversation", "") == ""

    def test_partial_section_values_as_empty_string(self) -> None:
        """Dropped sections should have empty string values, not be absent."""
        budget = TokenBudget(
            max_tokens=850,
            system_prompt_reserve=100,
            user_message_reserve=200,
            response_reserve=500,
        )
        # available = 50 tokens
        sections = {
            "conversation": "short",  # fits
            "patterns": " ".join(["word"] * 200),  # way too big
        }
        result = budget.trim_to_budget(sections)
        assert result["conversation"] == "short"
        assert result["patterns"] == ""
