"""Tests for the wallet scoring engine."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from sharp_discovery.config import ScoringConfig
from sharp_discovery.models import WalletMarketResult
from sharp_discovery.scorer import compute_wallet_score, _z_score


def _make_results(
    wallet: str = "0xabc",
    count: int = 25,
    win_fraction: float = 0.6,
    roi_base: float = 0.10,
    hold_fraction: float = 0.8,
    avg_entry: float = 0.50,
) -> list[WalletMarketResult]:
    """Generate synthetic wallet market results."""
    results = []
    now = datetime.utcnow()
    for i in range(count):
        won = i < int(count * win_fraction)
        roi = roi_base if won else -roi_base * 0.5
        results.append(
            WalletMarketResult(
                wallet=wallet,
                market_id=f"m{i}",
                won=won,
                roi=roi,
                avg_entry=avg_entry,
                held_to_expiration=i < int(count * hold_fraction),
                total_bought=100,
                total_sold=20 if i < int(count * hold_fraction) else 100,
                resolution_date=now - timedelta(days=180 - i * 7),
            )
        )
    return results


class TestSharpePathScoring:
    def test_basic_scoring(self):
        results = _make_results()
        config = ScoringConfig(min_resolved_markets=20)
        score = compute_wallet_score("0xabc", results, config)
        assert score is not None
        assert score.address == "0xabc"
        assert score.win_rate == pytest.approx(0.75)
        assert score.resolved_market_count == 25
        assert score.sharpe_ratio != 0.0
        assert score.sharpe_ci_lower < score.sharpe_ratio < score.sharpe_ci_upper

    def test_below_min_markets_returns_none(self):
        results = _make_results(count=5)
        config = ScoringConfig(min_resolved_markets=20)
        assert compute_wallet_score("0xabc", results, config) is None

    def test_below_min_hold_ratio_returns_none(self):
        results = _make_results(count=25, hold_fraction=0.3)
        config = ScoringConfig(min_resolved_markets=20, min_hold_ratio=0.70)
        assert compute_wallet_score("0xabc", results, config) is None

    def test_significance_gate_filters_weak_sharpe(self):
        config = ScoringConfig(min_resolved_markets=10)
        results = _make_results(count=12, win_fraction=0.55, hold_fraction=1.0)
        assert compute_wallet_score("0xweak", results, config) is None

    def test_composite_is_sharpe_times_penalty(self):
        """No extreme entries → composite == sharpe."""
        config = ScoringConfig(min_resolved_markets=20)
        results = _make_results(win_fraction=0.8, hold_fraction=1.0, avg_entry=0.50)
        score = compute_wallet_score("0xtest", results, config)
        assert score is not None
        assert score.extreme_price_ratio == 0.0
        assert score.composite_score == pytest.approx(score.sharpe_ratio)

    def test_longer_track_record_tightens_ci(self):
        config = ScoringConfig(min_resolved_markets=20)
        short = compute_wallet_score("s", _make_results(count=25, win_fraction=0.7), config)
        long = compute_wallet_score("l", _make_results(count=100, win_fraction=0.7), config)
        assert short is not None and long is not None
        assert (long.sharpe_ci_upper - long.sharpe_ci_lower) < (short.sharpe_ci_upper - short.sharpe_ci_lower)


class TestConsistencyPath:
    def test_all_wins_near_zero_stdev(self):
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xcons", market_id=f"m{i}", won=True,
                roi=0.001, avg_entry=0.999,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(15)
        ]
        config = ScoringConfig(min_resolved_markets=10)
        score = compute_wallet_score("0xcons", results, config)
        assert score is not None
        assert score.sharpe_ratio == 0.0
        # All entries at 0.999 >= 0.95 → extreme_price_ratio = 1.0
        assert score.extreme_price_ratio == pytest.approx(1.0)
        # Penalized: composite = win_rate * log(15) * (1 - 0.5 * 1.0)
        expected = 1.0 * math.log(15) * 0.5
        assert score.composite_score == pytest.approx(expected)

    def test_rejects_low_win_rate(self):
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xlow", market_id=f"m{i}",
                won=i < 10, roi=0.001, avg_entry=0.50,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(15)
        ]
        config = ScoringConfig(min_resolved_markets=10)
        assert compute_wallet_score("0xlow", results, config) is None


class TestExtremePricePenalty:
    def test_no_penalty_for_normal_entries(self):
        """Wallets entering at reasonable prices get no penalty."""
        config = ScoringConfig(min_resolved_markets=20)
        results = _make_results(win_fraction=0.8, hold_fraction=1.0, avg_entry=0.50)
        score = compute_wallet_score("0xnormal", results, config)
        assert score is not None
        assert score.extreme_price_ratio == 0.0
        assert score.composite_score == pytest.approx(score.sharpe_ratio)

    def test_full_penalty_for_all_extreme(self):
        """100% extreme entries → score * (1 - penalty)."""
        config = ScoringConfig(
            min_resolved_markets=20,
            extreme_price_threshold=0.95,
            extreme_price_penalty=0.50,
        )
        results = _make_results(win_fraction=0.8, hold_fraction=1.0, avg_entry=0.96)
        score = compute_wallet_score("0xextreme", results, config)
        assert score is not None
        assert score.extreme_price_ratio == pytest.approx(1.0)
        assert score.composite_score == pytest.approx(score.sharpe_ratio * 0.50)

    def test_partial_penalty(self):
        """Half extreme entries → score * (1 - penalty * 0.5)."""
        config = ScoringConfig(
            min_resolved_markets=10,
            extreme_price_threshold=0.95,
            extreme_price_penalty=0.50,
        )
        now = datetime.utcnow()
        results = []
        for i in range(20):
            won = i < 16  # 80% win
            roi = 0.10 if won else -0.05
            # Half at normal price, half at extreme
            entry = 0.96 if i < 10 else 0.50
            results.append(WalletMarketResult(
                wallet="0xhalf", market_id=f"m{i}", won=won, roi=roi,
                avg_entry=entry, held_to_expiration=True,
                total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            ))
        score = compute_wallet_score("0xhalf", results, config)
        assert score is not None
        assert score.extreme_price_ratio == pytest.approx(0.5)
        assert score.composite_score == pytest.approx(score.sharpe_ratio * 0.75)

    def test_penalty_threshold_configurable(self):
        """Custom threshold at 0.90 catches more entries."""
        config = ScoringConfig(
            min_resolved_markets=20,
            extreme_price_threshold=0.90,
            extreme_price_penalty=0.50,
        )
        # All entries at 0.92 → above 0.90 threshold
        results = _make_results(win_fraction=0.8, hold_fraction=1.0, avg_entry=0.92)
        score = compute_wallet_score("0xcustom", results, config)
        assert score is not None
        assert score.extreme_price_ratio == pytest.approx(1.0)
        assert score.composite_score == pytest.approx(score.sharpe_ratio * 0.50)

    def test_consistency_path_also_penalized(self):
        """Extreme price penalty applies to consistency path too."""
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xcons_ext", market_id=f"m{i}", won=True,
                roi=0.001, avg_entry=0.999,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(15)
        ]
        config = ScoringConfig(
            min_resolved_markets=10,
            extreme_price_penalty=0.50,
        )
        score = compute_wallet_score("0xcons_ext", results, config)
        assert score is not None
        # Without penalty: 1.0 * log(15) ≈ 2.708
        # With penalty (100% extreme): * 0.50
        assert score.composite_score == pytest.approx(1.0 * math.log(15) * 0.50)

    def test_zero_penalty_config(self):
        """Setting penalty to 0 disables it."""
        config = ScoringConfig(
            min_resolved_markets=20,
            extreme_price_penalty=0.0,
        )
        results = _make_results(win_fraction=0.8, hold_fraction=1.0, avg_entry=0.99)
        score = compute_wallet_score("0xzero", results, config)
        assert score is not None
        assert score.composite_score == pytest.approx(score.sharpe_ratio)


class TestZScore:
    def test_95_confidence(self):
        assert _z_score(0.95) == pytest.approx(1.96, abs=0.01)

    def test_99_confidence(self):
        assert _z_score(0.99) == pytest.approx(2.576, abs=0.01)

    def test_90_confidence(self):
        assert _z_score(0.90) == pytest.approx(1.645, abs=0.01)
