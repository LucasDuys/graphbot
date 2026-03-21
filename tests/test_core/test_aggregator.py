"""Tests for deterministic aggregation of parallel leaf outputs."""

from __future__ import annotations

import json
import time

from core_gb.aggregator import Aggregator


class TestConcatenate:
    def test_concatenate_multiple(self) -> None:
        agg = Aggregator()
        outputs = {
            "weather_ams": "Sunny, 22C",
            "weather_lon": "Rainy, 15C",
            "weather_par": "Cloudy, 18C",
        }
        result = agg.aggregate(None, outputs)
        assert "## Weather Ams" in result
        assert "Sunny, 22C" in result
        assert "## Weather Lon" in result
        assert "Rainy, 15C" in result
        assert "## Weather Par" in result
        assert "Cloudy, 18C" in result

    def test_concatenate_single(self) -> None:
        agg = Aggregator()
        outputs = {"only_key": "Just one output"}
        result = agg.aggregate(None, outputs)
        assert result == "Just one output"
        assert "##" not in result

    def test_concatenate_empty(self) -> None:
        agg = Aggregator()
        result = agg.aggregate(None, {})
        assert result == ""


class TestTemplateFill:
    def test_template_fill(self) -> None:
        agg = Aggregator()
        template = {
            "aggregation_type": "template_fill",
            "template": "Amsterdam: {weather_ams}\nLondon: {weather_lon}",
            "slot_definitions": {
                "weather_ams": "Weather in Amsterdam",
                "weather_lon": "Weather in London",
            },
        }
        outputs = {
            "weather_ams": "Sunny, 22C",
            "weather_lon": "Rainy, 15C",
        }
        result = agg.aggregate(template, outputs)
        assert result == "Amsterdam: Sunny, 22C\nLondon: Rainy, 15C"

    def test_template_fill_missing_slot(self) -> None:
        agg = Aggregator()
        template = {
            "aggregation_type": "template_fill",
            "template": "Data: {present} and {missing}",
        }
        outputs = {"present": "here"}
        result = agg.aggregate(template, outputs)
        assert "here" in result
        assert "[No data for missing]" in result


class TestMergeJson:
    def test_merge_json(self) -> None:
        agg = Aggregator()
        template = {"aggregation_type": "merge_json"}
        outputs = {
            "a": json.dumps({"city": "Amsterdam", "temp": 22}),
            "b": json.dumps({"city_lon": "London", "temp_lon": 15}),
            "c": json.dumps({"city_par": "Paris", "temp_par": 18}),
        }
        result = agg.aggregate(template, outputs)
        parsed = json.loads(result)
        assert parsed["city"] == "Amsterdam"
        assert parsed["temp"] == 22
        assert parsed["city_lon"] == "London"
        assert parsed["temp_lon"] == 15
        assert parsed["city_par"] == "Paris"
        assert parsed["temp_par"] == 18

    def test_merge_json_non_json(self) -> None:
        agg = Aggregator()
        template = {"aggregation_type": "merge_json"}
        outputs = {
            "json_key": json.dumps({"key": "value"}),
            "plain_key": "just plain text",
        }
        result = agg.aggregate(template, outputs)
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["plain_key"] == "just plain text"


class TestConfidenceRanked:
    def test_confidence_ranked(self) -> None:
        agg = Aggregator()
        template = {"aggregation_type": "confidence_ranked"}
        outputs = {
            "low": json.dumps({"confidence": 0.2, "answer": "Low confidence answer"}),
            "high": json.dumps({"confidence": 0.9, "answer": "High confidence answer"}),
            "mid": json.dumps({"confidence": 0.5, "answer": "Mid confidence answer"}),
        }
        result = agg.aggregate(template, outputs)
        lines = result.split("\n\n")
        assert lines[0] == "High confidence answer"
        assert lines[1] == "Mid confidence answer"
        assert lines[2] == "Low confidence answer"


class TestDefaults:
    def test_none_template_defaults_to_concatenate(self) -> None:
        agg = Aggregator()
        outputs = {"a": "Alpha", "b": "Beta"}
        result_none = agg.aggregate(None, outputs)
        result_concat = agg.aggregate({"aggregation_type": "concatenate"}, outputs)
        assert result_none == result_concat


class TestPerformance:
    def test_aggregation_under_1ms(self) -> None:
        agg = Aggregator()
        outputs = {f"key_{i}": f"Output number {i}" for i in range(10)}
        start = time.perf_counter()
        agg.aggregate(None, outputs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1.0, f"Aggregation took {elapsed_ms:.3f}ms, expected <1ms"
