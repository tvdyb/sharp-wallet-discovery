"""Configuration for the discovery engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APIConfig:
    """Polymarket API configuration."""

    clob_base_url: str = "https://clob.polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    rate_limit_per_second: float = 50.0
    request_timeout: float = 30.0
    max_retries: int = 10
    backoff_base: float = 1.0
    backoff_max: float = 120.0


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
    extreme_price_threshold: float = 0.95  # entries at or above this are "extreme"
    extreme_price_penalty: float = 0.50  # max discount factor when 100% of entries are extreme
    min_roi_stdev: float = 0.001  # below this → consistency path
    min_consistency_win_rate: float = 0.85  # required for consistency path
    max_both_sides_ratio: float = 0.50  # filter market makers: max % of markets with trades on multiple outcomes
    max_trades_per_market: float = 20.0  # filter market makers: max avg trades per market
    exclude_sports: bool = True  # filter out sports/esports markets


@dataclass(frozen=True)
class DiscoveryConfig:
    """Top-level discovery configuration."""

    api: APIConfig = APIConfig()
    scoring: ScoringConfig = ScoringConfig()
    min_volume: float = 10000.0  # skip markets with < $10k volume
    max_markets: int = 0  # 0 = unlimited
    min_recency_days: int = 0  # 0 = no recency filter
    top_wallets: int = 50
    leaderboard_depth: int = 500  # how many leaderboard entries to pull
    db_path: str = "sharp_discovery.db"
    cache_path: str = "data_cache.json"
    use_goldsky: bool = True
    goldsky_since: str = ""  # YYYY-MM-DD, default computed at runtime
    goldsky_cache: str = "goldsky_trades.csv"


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for smart-money signal generation."""

    min_market_volume: float = 50_000.0  # only markets above this volume
    min_wallet_score: float = 0.0  # minimum composite_score to include
    max_markets: int = 100  # cap on active markets to analyze
