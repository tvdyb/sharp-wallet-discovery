"""Wallet scoring engine with dual-path Sharpe/consistency scoring and extreme-price penalty."""

from __future__ import annotations

import math
from statistics import mean, stdev

from sharp_discovery.config import ScoringConfig
from sharp_discovery.models import WalletMarketResult, WalletScore


def compute_wallet_score(
    wallet: str,
    results: list[WalletMarketResult],
    config: ScoringConfig,
) -> WalletScore | None:
    """Compute wallet rating via dual-path scoring with extreme-price penalty.

    **Sharpe path** (ROI stdev >= min_roi_stdev):
        composite_score = Sharpe point estimate * penalty_factor,
        gated by CI lower bound > 0.

    **Consistency path** (ROI stdev < min_roi_stdev):
        composite_score = win_rate * log(n_held) * penalty_factor.

    **Extreme-price penalty**: wallets that predominantly enter at >= 95 cents
    get their composite_score discounted. If 100% of held entries are extreme,
    score is multiplied by (1 - extreme_price_penalty). The penalty scales
    linearly with the fraction of extreme entries.

    Returns None if the wallet doesn't meet minimum filters.
    """
    if len(results) < config.min_resolved_markets:
        return None

    held_count = sum(1 for r in results if r.held_to_expiration)
    hold_ratio = held_count / len(results)
    if hold_ratio < config.min_hold_ratio:
        return None

    held_results = [r for r in results if r.held_to_expiration]
    if len(held_results) < 2:
        return None

    wins = sum(1 for r in held_results if r.won)
    win_rate = wins / len(held_results)

    rois = [r.roi for r in held_results]
    avg_roi = mean(rois)

    # Extreme-price penalty: fraction of held entries at >= threshold
    extreme_count = sum(
        1 for r in held_results if r.avg_entry >= config.extreme_price_threshold
    )
    extreme_ratio = extreme_count / len(held_results)
    penalty_factor = 1.0 - config.extreme_price_penalty * extreme_ratio

    sd = stdev(rois)

    if sd >= config.min_roi_stdev:
        # ── Sharpe path ──────────────────────────────────────────────
        sharpe = max(min(avg_roi / sd, 10.0), -10.0)

        n = len(rois)
        se = math.sqrt((1.0 + sharpe ** 2 / 2.0) / n)

        z = _z_score(config.ci_confidence)
        ci_lower = sharpe - z * se
        ci_upper = sharpe + z * se

        if ci_lower <= 0:
            return None

        return WalletScore(
            address=wallet,
            win_rate=win_rate,
            avg_roi=avg_roi,
            sharpe_ratio=sharpe,
            sharpe_ci_lower=ci_lower,
            sharpe_ci_upper=ci_upper,
            hold_ratio=hold_ratio,
            resolved_market_count=len(results),
            composite_score=sharpe * penalty_factor,
            extreme_price_ratio=extreme_ratio,
        )
    else:
        # ── Consistency path ─────────────────────────────────────────
        if len(held_results) < config.min_resolved_markets:
            return None
        if win_rate < 0.85:
            return None

        consistency_score = win_rate * math.log(len(held_results))

        return WalletScore(
            address=wallet,
            win_rate=win_rate,
            avg_roi=avg_roi,
            sharpe_ratio=0.0,
            sharpe_ci_lower=0.0,
            sharpe_ci_upper=0.0,
            hold_ratio=hold_ratio,
            resolved_market_count=len(results),
            composite_score=consistency_score * penalty_factor,
            extreme_price_ratio=extreme_ratio,
        )


def _z_score(confidence: float) -> float:
    """Approximate z-score for a two-tailed confidence level.

    Uses the rational approximation from Abramowitz & Stegun.
    """
    alpha = (1.0 - confidence) / 2.0
    t = math.sqrt(-2.0 * math.log(alpha))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
