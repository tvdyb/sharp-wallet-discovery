"""Data models for wallet discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MarketToken:
    """A tradeable token (YES or NO) within a market."""

    token_id: str
    outcome: str = ""
    winner: bool = False


@dataclass
class Market:
    """Resolved market metadata from Gamma API."""

    condition_id: str
    question: str
    slug: str = ""
    outcome: str = ""
    resolution_date: datetime | None = None
    tokens: list[MarketToken] = field(default_factory=list)
    volume: float = 0.0


@dataclass
class Trade:
    """A single trade from the CLOB API."""

    id: str
    market: str
    asset_id: str
    side: str  # BUY or SELL
    size: float
    price: float
    timestamp: datetime | None = None
    owner: str = ""  # wallet address


@dataclass
class WalletMarketResult:
    """A wallet's participation in a single resolved market.

    Tracks the avg_entry price so we can detect and penalize
    wallets that exclusively buy at extreme prices (95+ cents).
    """

    wallet: str
    market_id: str
    won: bool
    roi: float
    avg_entry: float  # volume-weighted average BUY price
    held_to_expiration: bool
    total_bought: float
    total_sold: float
    resolution_date: datetime | None = None
    last_trade_date: datetime | None = None


@dataclass
class WalletScore:
    """Computed score for a wallet.

    Dual-path scoring on held-to-expiration positions:

    1. **Sharpe path** (ROI stdev >= min_roi_stdev): composite_score is the
       Sharpe point estimate with extreme-price penalty applied.
       CI lower bound must be > 0 for qualification (significance gate).
    2. **Consistency path** (ROI stdev < min_roi_stdev, win_rate >= 85%):
       composite_score = win_rate * log(n_held), also penalized for
       extreme entry prices.
    """

    address: str
    win_rate: float = 0.0
    avg_roi: float = 0.0
    sharpe_ratio: float = 0.0
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    hold_ratio: float = 0.0
    resolved_market_count: int = 0
    composite_score: float = 0.0
    extreme_price_ratio: float = 0.0  # fraction of entries at >= 95 cents
    last_trade_date: datetime | None = None
    scored_at: datetime = field(default_factory=datetime.utcnow)
