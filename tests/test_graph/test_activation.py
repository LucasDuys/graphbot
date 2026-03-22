"""Tests for ACT-R activation model for graph nodes."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from graph.activation import ActivationModel


class TestActivationScore:
    """Activation score computation from base_level + recency_boost + frequency_boost."""

    def test_returns_float(self) -> None:
        """activation_score returns a float value."""
        model = ActivationModel()
        score = model.activation_score(access_count=5, last_accessed=datetime.now(timezone.utc))
        assert isinstance(score, float)

    def test_frequently_accessed_higher_than_stale(self) -> None:
        """A frequently accessed node scores higher than a stale node."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)

        # Frequently accessed: 100 accesses, accessed just now
        frequent_score = model.activation_score(access_count=100, last_accessed=now)

        # Stale: 1 access, accessed 30 days ago
        stale_time = now - timedelta(days=30)
        stale_score = model.activation_score(access_count=1, last_accessed=stale_time)

        assert frequent_score > stale_score

    def test_more_accesses_increases_score(self) -> None:
        """More accesses should yield a higher score, all else equal."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)

        score_low = model.activation_score(access_count=2, last_accessed=now)
        score_high = model.activation_score(access_count=50, last_accessed=now)

        assert score_high > score_low

    def test_recent_access_increases_score(self) -> None:
        """More recent access should yield a higher score, all else equal."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)

        score_recent = model.activation_score(access_count=5, last_accessed=now)
        score_old = model.activation_score(
            access_count=5, last_accessed=now - timedelta(days=7)
        )

        assert score_recent > score_old

    def test_recency_decays_over_time(self) -> None:
        """Recency boost should decrease as time since last access grows."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)

        scores = []
        for days_ago in [0, 1, 7, 30, 90]:
            t = now - timedelta(days=days_ago)
            scores.append(model.activation_score(access_count=5, last_accessed=t))

        # Each score should be >= the next (monotonically non-increasing)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at {i} days ago ({scores[i]}) should be >= "
                f"score at {i+1} days ago ({scores[i+1]})"
            )

    def test_zero_access_count_returns_baseline(self) -> None:
        """Zero access count should still produce a valid (low) score."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)
        score = model.activation_score(access_count=0, last_accessed=now)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_none_last_accessed_uses_minimum_score(self) -> None:
        """None last_accessed should produce minimum recency (as if very old)."""
        model = ActivationModel()
        score_none = model.activation_score(access_count=5, last_accessed=None)
        score_recent = model.activation_score(
            access_count=5, last_accessed=datetime.now(timezone.utc)
        )
        assert score_recent > score_none

    def test_score_is_non_negative(self) -> None:
        """Activation scores should never be negative."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)

        for count in [0, 1, 5, 100]:
            for days in [0, 1, 30, 365]:
                t = now - timedelta(days=days)
                score = model.activation_score(access_count=count, last_accessed=t)
                assert score >= 0.0, f"Negative score for count={count}, days_ago={days}"


class TestActivationComponents:
    """Test individual components of the activation formula."""

    def test_base_level_increases_with_access_count(self) -> None:
        """base_level is log of access count, so it grows with accesses."""
        model = ActivationModel()
        base_low = model.base_level(access_count=1)
        base_high = model.base_level(access_count=100)
        assert base_high > base_low

    def test_base_level_zero_count(self) -> None:
        """base_level with zero accesses should return zero or a small value."""
        model = ActivationModel()
        base = model.base_level(access_count=0)
        assert base >= 0.0

    def test_recency_boost_recent(self) -> None:
        """recency_boost is highest for very recent access."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)
        boost_now = model.recency_boost(last_accessed=now)
        boost_old = model.recency_boost(last_accessed=now - timedelta(days=30))
        assert boost_now > boost_old

    def test_recency_boost_none(self) -> None:
        """recency_boost with None last_accessed returns zero."""
        model = ActivationModel()
        boost = model.recency_boost(last_accessed=None)
        assert boost == 0.0

    def test_frequency_boost_proportional(self) -> None:
        """frequency_boost increases with access count."""
        model = ActivationModel()
        fb_low = model.frequency_boost(access_count=2)
        fb_high = model.frequency_boost(access_count=50)
        assert fb_high > fb_low

    def test_frequency_boost_zero(self) -> None:
        """frequency_boost with zero accesses returns zero."""
        model = ActivationModel()
        fb = model.frequency_boost(access_count=0)
        assert fb == 0.0

    def test_frequency_boost_sublinear(self) -> None:
        """frequency_boost grows sublinearly (logarithmic) to prevent runaway scores."""
        model = ActivationModel()
        fb_10 = model.frequency_boost(access_count=10)
        fb_100 = model.frequency_boost(access_count=100)
        fb_1000 = model.frequency_boost(access_count=1000)

        # The ratio between 100 and 10 should be larger than between 1000 and 100
        # (logarithmic growth means diminishing returns)
        ratio_1 = fb_100 / fb_10
        ratio_2 = fb_1000 / fb_100
        assert ratio_2 < ratio_1


class TestActivationModelConfiguration:
    """Test configurable parameters of the activation model."""

    def test_custom_decay_rate(self) -> None:
        """Custom decay_rate changes how fast recency drops off."""
        fast_decay = ActivationModel(decay_rate=1.0)
        slow_decay = ActivationModel(decay_rate=0.01)
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        # With fast decay, a week-old access should get much less boost
        fast_score = fast_decay.recency_boost(last_accessed=week_ago)
        slow_score = slow_decay.recency_boost(last_accessed=week_ago)
        assert slow_score > fast_score

    def test_custom_frequency_weight(self) -> None:
        """Custom frequency_weight scales the frequency boost."""
        low_weight = ActivationModel(frequency_weight=0.1)
        high_weight = ActivationModel(frequency_weight=2.0)

        fb_low = low_weight.frequency_boost(access_count=10)
        fb_high = high_weight.frequency_boost(access_count=10)
        assert fb_high > fb_low

    def test_custom_recency_weight(self) -> None:
        """Custom recency_weight scales the recency boost."""
        low_weight = ActivationModel(recency_weight=0.1)
        high_weight = ActivationModel(recency_weight=2.0)
        now = datetime.now(timezone.utc)

        rb_low = low_weight.recency_boost(last_accessed=now)
        rb_high = high_weight.recency_boost(last_accessed=now)
        assert rb_high > rb_low


class TestActivationScoreBatch:
    """Test batch scoring for multiple nodes."""

    def test_score_batch_returns_sorted(self) -> None:
        """score_batch returns nodes sorted by activation score descending."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)

        nodes = [
            {"id": "stale", "access_count": 1, "last_accessed": now - timedelta(days=30)},
            {"id": "hot", "access_count": 100, "last_accessed": now},
            {"id": "warm", "access_count": 10, "last_accessed": now - timedelta(days=1)},
        ]

        scored = model.score_batch(nodes)

        assert len(scored) == 3
        # Should be sorted descending by score
        for i in range(len(scored) - 1):
            assert scored[i][1] >= scored[i + 1][1]

        # Hot should be first
        assert scored[0][0]["id"] == "hot"

    def test_score_batch_empty(self) -> None:
        """score_batch with empty list returns empty list."""
        model = ActivationModel()
        scored = model.score_batch([])
        assert scored == []

    def test_score_batch_preserves_node_data(self) -> None:
        """score_batch does not modify the original node dicts."""
        model = ActivationModel()
        now = datetime.now(timezone.utc)
        node = {"id": "test", "access_count": 5, "last_accessed": now, "extra": "data"}
        scored = model.score_batch([node])
        assert scored[0][0] is node
        assert "extra" in scored[0][0]
