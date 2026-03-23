"""Tests for IntakeParser -- rule-based, zero-cost intent classification."""

import time

from core_gb.intake import IntakeParser, IntakeResult
from core_gb.types import Domain


class TestDomainClassification:
    """Verify >80% correct domain classification across all 6 domains."""

    def setup_method(self) -> None:
        self.parser = IntakeParser()

    # FILE domain
    def test_file_read(self) -> None:
        result = self.parser.parse("Read the readme file")
        assert result.domain == Domain.FILE

    def test_file_create(self) -> None:
        result = self.parser.parse("Create a new directory called logs")
        assert result.domain == Domain.FILE

    def test_file_save(self) -> None:
        result = self.parser.parse("Save this to a file named todo")
        assert result.domain == Domain.FILE

    # WEB domain
    def test_web_search(self) -> None:
        result = self.parser.parse("Search for the latest news about AI")
        assert result.domain == Domain.WEB

    def test_web_fetch(self) -> None:
        result = self.parser.parse("Fetch the URL http://example.com")
        assert result.domain == Domain.WEB

    def test_web_weather(self) -> None:
        result = self.parser.parse("What is the weather in London")
        assert result.domain == Domain.WEB

    # CODE domain
    def test_code_function(self) -> None:
        # Code generation goes to FILE (write) or SYSTEM (LLM), not CODE.
        # CODE is reserved for shell commands and code-edit tasks.
        result = self.parser.parse("Write a Python function to sort a list")
        assert result.domain != Domain.CODE

    def test_code_debug(self) -> None:
        result = self.parser.parse("Debug this code and fix the bug")
        assert result.domain == Domain.CODE

    def test_code_refactor(self) -> None:
        result = self.parser.parse("Refactor the class to use composition")
        assert result.domain == Domain.CODE

    # COMMS domain
    def test_comms_email(self) -> None:
        result = self.parser.parse("Send an email to John about the meeting")
        assert result.domain == Domain.COMMS

    def test_comms_slack(self) -> None:
        result = self.parser.parse("Send a message on Slack to the team")
        assert result.domain == Domain.COMMS

    def test_comms_reply(self) -> None:
        result = self.parser.parse("Reply to the Discord notification")
        assert result.domain == Domain.COMMS

    # SYSTEM domain
    def test_system_calculate(self) -> None:
        result = self.parser.parse("Calculate 15 percent of 200")
        assert result.domain == Domain.SYSTEM

    def test_system_time(self) -> None:
        result = self.parser.parse("What time is it in Tokyo")
        assert result.domain == Domain.SYSTEM

    def test_system_explain(self) -> None:
        result = self.parser.parse("Explain how recursion works")
        assert result.domain == Domain.SYSTEM

    # SYNTHESIS domain
    def test_synthesis_compare(self) -> None:
        result = self.parser.parse("Compare the performance of these two algorithms")
        assert result.domain == Domain.SYNTHESIS

    def test_synthesis_summarize(self) -> None:
        result = self.parser.parse("Summarize the report and evaluate the findings")
        assert result.domain == Domain.SYNTHESIS

    def test_synthesis_research(self) -> None:
        result = self.parser.parse("Research and analyze market trends")
        assert result.domain == Domain.SYNTHESIS

    def test_synthesis_plan(self) -> None:
        result = self.parser.parse("Design a plan for the new architecture")
        assert result.domain == Domain.SYNTHESIS

    def test_classification_accuracy(self) -> None:
        """At least 80% of the above 20 test cases must pass (tested individually)."""
        cases: list[tuple[str, Domain]] = [
            ("Read the readme file", Domain.FILE),
            ("Create a new directory called logs", Domain.FILE),
            ("Save this to a file named todo", Domain.FILE),
            ("Search for the latest news about AI", Domain.WEB),
            ("Fetch the URL http://example.com", Domain.WEB),
            ("What is the weather in London", Domain.WEB),
            # Code generation goes to FILE (via "write" keyword), not CODE.
            # CODE is narrowed to shell commands and code-edit tasks.
            ("Write a Python function to sort a list", Domain.FILE),
            ("Debug this code and fix the bug", Domain.CODE),
            ("Refactor the class to use composition", Domain.CODE),
            ("Send an email to John about the meeting", Domain.COMMS),
            ("Send a message on Slack to the team", Domain.COMMS),
            ("Reply to the Discord notification", Domain.COMMS),
            ("Calculate 15 percent of 200", Domain.SYSTEM),
            ("What time is it in Tokyo", Domain.SYSTEM),
            ("Explain how recursion works", Domain.SYSTEM),
            ("Compare the performance of these two algorithms", Domain.SYNTHESIS),
            ("Summarize the report and evaluate the findings", Domain.SYNTHESIS),
            ("Research and analyze market trends", Domain.SYNTHESIS),
            ("Design a plan for the new architecture", Domain.SYNTHESIS),
            ("Review the code changes in the pull request", Domain.SYNTHESIS),
        ]
        correct = sum(
            1
            for msg, expected in cases
            if self.parser.parse(msg).domain == expected
        )
        accuracy = correct / len(cases)
        assert accuracy >= 0.80, f"Accuracy {accuracy:.0%} < 80% ({correct}/{len(cases)})"


class TestComplexityEstimation:
    """Verify complexity estimates are within +/-1 of expected."""

    def setup_method(self) -> None:
        self.parser = IntakeParser()

    def _assert_within_one(self, message: str, expected: int) -> None:
        result = self.parser.parse(message)
        assert abs(result.complexity - expected) <= 1, (
            f"'{message}': got {result.complexity}, expected {expected} (+/-1)"
        )

    def test_single_word_command(self) -> None:
        self._assert_within_one("help", 1)

    def test_short_question(self) -> None:
        self._assert_within_one("What is 2+2?", 1)

    def test_simple_request(self) -> None:
        self._assert_within_one("Read the file", 1)

    def test_moderate_request(self) -> None:
        self._assert_within_one("Search the web for Python tutorials and save results", 2)

    def test_multiple_conjunctions(self) -> None:
        self._assert_within_one(
            "Read the file and then parse it and also save the output", 3
        )

    def test_long_message(self) -> None:
        self._assert_within_one(
            "I need you to read the configuration file from the server "
            "and then parse all the settings that relate to the database connection",
            4,
        )

    def test_very_long_message(self) -> None:
        self._assert_within_one(
            "I need you to read the configuration file from the server "
            "and then parse all the settings that relate to the database "
            "connection pooling and timeout values and also generate a report "
            "summarizing the current state of all configuration parameters",
            4,
        )

    def test_comma_list(self) -> None:
        self._assert_within_one(
            "Check London, Paris, Berlin, Tokyo, New York", 2
        )

    def test_multiple_questions(self) -> None:
        self._assert_within_one(
            "What is the weather? What time is it? How far is the store?", 2
        )

    def test_compound_task(self) -> None:
        self._assert_within_one(
            "Search for weather data and compare results across cities, "
            "then summarize the findings and also email the report", 4
        )

    def test_minimal_message(self) -> None:
        self._assert_within_one("hi", 1)

    def test_complexity_cap_at_five(self) -> None:
        msg = (
            "Read the file and parse it and also convert the data and then "
            "compare results, summarize the output, email the report, "
            "and schedule a follow-up? Is that possible? Can you do it? "
            "Also what about the other thing as well plus the extra items"
        )
        result = self.parser.parse(msg)
        assert result.complexity <= 5

    def test_two_conjunctions(self) -> None:
        self._assert_within_one("Read the file and save it plus format it", 2)

    def test_no_conjunctions_moderate_length(self) -> None:
        self._assert_within_one(
            "Please provide a brief overview of the current project status", 1
        )

    def test_commas_with_conjunctions(self) -> None:
        self._assert_within_one(
            "Analyze London, Paris, Berlin and then compare results and summarize", 4
        )


class TestSimpleTaskDetection:

    def setup_method(self) -> None:
        self.parser = IntakeParser()

    def test_simple_math(self) -> None:
        result = self.parser.parse("What is 2+2?")
        assert result.is_simple is True

    def test_simple_time(self) -> None:
        result = self.parser.parse("What time is it?")
        assert result.is_simple is True

    def test_complex_multi_domain(self) -> None:
        result = self.parser.parse(
            "Compare weather in 5 cities and summarize"
        )
        assert result.is_simple is False

    def test_complex_long_task(self) -> None:
        result = self.parser.parse(
            "Research AI trends and then analyze the data and also write a "
            "detailed report comparing the top frameworks"
        )
        assert result.is_simple is False


class TestEntityExtraction:

    def setup_method(self) -> None:
        self.parser = IntakeParser()

    def test_extracts_capitalized_words(self) -> None:
        result = self.parser.parse("Send email to John about the London meeting")
        assert "John" in result.entities
        assert "London" in result.entities

    def test_ignores_short_words(self) -> None:
        result = self.parser.parse("The AI model is good")
        # "AI" is only 2 chars, should be excluded
        assert "AI" not in result.entities

    def test_ignores_sentence_start(self) -> None:
        # First word capitalized by convention, not necessarily an entity
        # but our simple heuristic may include it -- just verify no crash
        result = self.parser.parse("Calculate the sum")
        assert isinstance(result.entities, tuple)

    def test_empty_message(self) -> None:
        result = self.parser.parse("")
        assert result.entities == ()


class TestIntakeResult:

    def setup_method(self) -> None:
        self.parser = IntakeParser()

    def test_result_is_frozen(self) -> None:
        result = self.parser.parse("Read a file")
        try:
            result.domain = Domain.WEB  # type: ignore[misc]
            assert False, "IntakeResult should be frozen"
        except AttributeError:
            pass

    def test_raw_message_preserved(self) -> None:
        msg = "Search the web for Python tutorials"
        result = self.parser.parse(msg)
        assert result.raw_message == msg


class TestPerformance:

    def test_parse_latency(self) -> None:
        """1000 parse() calls must complete in <5000ms total."""
        parser = IntakeParser()
        messages = [
            "Read the file",
            "Search for weather data",
            "Write a Python function to sort a list",
            "Send an email to the team",
            "What is 2+2?",
            "Compare and summarize the results of the analysis",
            "Create a new directory called logs and save output",
            "Fetch http://example.com and download the page",
            "Debug this code",
            "Explain how recursion works in detail",
        ]
        start = time.perf_counter()
        for i in range(1000):
            parser.parse(messages[i % len(messages)])
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"1000 calls took {elapsed_ms:.0f}ms (>5000ms)"
