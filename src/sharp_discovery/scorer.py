"""Wallet scoring: filter hard, rank by Sharpe."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from statistics import mean, stdev

from sharp_discovery.config import ScoringConfig
from sharp_discovery.models import WalletMarketResult, WalletScore


def compute_wallet_score(
    wallet: str,
    results: list[WalletMarketResult],
    config: ScoringConfig,
    min_recency_days: int = 0,
) -> WalletScore | None:
    """Score = Sharpe ratio, gated by hard filters.

    Filters:
    - min_resolved_markets (default 20)
    - min_hold_ratio (default 0.70)
    - min_total_volume (default 1000 USDC total bought)
    - min_recency_days (must have traded within N days)
    - Sharpe CI lower bound > 0 (statistically significant)

    Returns None if the wallet doesn't pass filters.
    """
    if len(results) < config.min_resolved_markets:
        return None

    # Recency filter
    trade_dates = [r.last_trade_date for r in results if r.last_trade_date]
    last_trade = max(trade_dates) if trade_dates else None
    if min_recency_days > 0:
        if not last_trade:
            return None
        cutoff = datetime.utcnow() - timedelta(days=min_recency_days)
        if last_trade < cutoff:
            return None

    # Hold ratio filter
    held_count = sum(1 for r in results if r.held_to_expiration)
    hold_ratio = held_count / len(results)
    if hold_ratio < config.min_hold_ratio:
        return None

    held_results = [r for r in results if r.held_to_expiration]
    if len(held_results) < 2:
        return None

    # Volume filter — total USDC bought across all markets
    total_volume = sum(r.total_bought * r.avg_entry for r in results)
    if total_volume < config.min_total_volume:
        return None

    # Sharpe
    wins = sum(1 for r in held_results if r.won)
    win_rate = wins / len(held_results)

    rois = [r.roi for r in held_results]
    avg_roi = mean(rois)
    sd = stdev(rois)

    if sd < 0.001:
        return None  # can't compute meaningful Sharpe

    sharpe = max(min(avg_roi / sd, 10.0), -10.0)

    # CI gate — Sharpe must be statistically significant
    n = len(rois)
    se = math.sqrt((1.0 + sharpe ** 2 / 2.0) / n)
    z = _z_score(config.ci_confidence)
    ci_lower = sharpe - z * se
    ci_upper = sharpe + z * se

    if ci_lower <= 0:
        return None

    # Extreme price ratio (for display, not scoring)
    extreme_count = sum(
        1 for r in held_results if r.avg_entry >= config.extreme_price_threshold
    )
    extreme_ratio = extreme_count / len(held_results)

    return WalletScore(
        address=wallet,
        win_rate=win_rate,
        avg_roi=avg_roi,
        sharpe_ratio=sharpe,
        sharpe_ci_lower=ci_lower,
        sharpe_ci_upper=ci_upper,
        hold_ratio=hold_ratio,
        resolved_market_count=len(results),
        composite_score=sharpe,
        extreme_price_ratio=extreme_ratio,
        last_trade_date=last_trade,
    )


def _z_score(confidence: float) -> float:
    """Approximate z-score for a two-tailed confidence level."""
    alpha = (1.0 - confidence) / 2.0
    t = math.sqrt(-2.0 * math.log(alpha))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
