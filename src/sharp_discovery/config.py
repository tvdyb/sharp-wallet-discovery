"""Configuration for the discovery engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APIConfig:
    """Polymarket API configuration."""

    clob_base_url: str = "https://clob.polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    rate_limit_per_second: float = 20.0
    request_timeout: float = 30.0
    max_retries: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0


@dataclass(frozen=True)
class ScoringConfig:
    """Wallet scoring parameters.

    Two scoring paths:
    1. Sharpe path: wallets with sufficient ROI variance are ranked by Sharpe
       point estimate, gated by CI lower bound > 0.
    2. Consistency path: wallets with near-zero ROI variance but high win rate
       and enough markets are scored via win_rate * log(n_held).

    Both paths apply an extreme_price_penalty that discounts wallets
    whose held positions are dominated by entries at >= 95 cents.
    """

    min_resolved_markets: int = 20
    min_hold_ratio: float = 0.70
    min_total_volume: float = 1000.0  # min USDC bought across all markets
    ci_confidence: float = 0.90
    extreme_price_threshold: float = 0.95  # for display only


@dataclass(frozen=True)
class DiscoveryConfig:
    """Top-level discovery configuration."""

    api: APIConfig = APIConfig()
    scoring: ScoringConfig = ScoringConfig()
    min_volume: float = 10000.0  # skip markets with < $10k volume
    max_markets: int = 0  # 0 = unlimited
    min_recency_days: int = 0  # 0 = no recency filter
    top_wallets: int = 50
    db_path: str = "sharp_discovery.db"
    cache_path: str = "data_cache.json"
