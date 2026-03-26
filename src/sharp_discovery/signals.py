"""Smart-money signal aggregation from sharp wallet positions."""

from __future__ import annotations

import asyncio
import math
import re
import time

import structlog

from sharp_discovery.api import DataAPIClient, GammaClient
from sharp_discovery.config import APIConfig, SignalConfig
from sharp_discovery.models import (
    ActiveMarket,
    MarketSignal,
    Trade,
    WalletPosition,
    WalletScore,
)
from sharp_discovery.scanner import _EXCLUDED_RE

logger = structlog.get_logger()


def _log(msg: str) -> None:
    print(msg, flush=True)


class SignalEngine:
    """Generates market signals from sharp wallet positions."""

    def __init__(
        self,
        signal_config: SignalConfig,
        api_config: APIConfig,
    ) -> None:
        self._config = signal_config
        self._api_config = api_config

    async def run(self, scores: list[WalletScore]) -> list[MarketSignal]:
        """Generate signals for active politics/culture markets.

        1. Fetch large active markets from Gamma (excluding sports/crypto-minute)
        2. Re-fetch activity for qualified wallets
        3. Compute net positions per wallet per market
        4. Aggregate weighted signals
        """
        # Filter to wallets above minimum score
        wallets = [
            s for s in scores
            if s.composite_score >= self._config.min_wallet_score
        ]
        if not wallets:
            _log("No qualifying wallets for signal generation")
            return []

        score_by_addr = {s.address: s.composite_score for s in wallets}
        _log(f"Generating signals from {len(wallets)} sharp wallets")

        async with GammaClient(self._api_config) as gamma:
            _log("Fetching active markets...")
            active_markets = await gamma.get_active_markets(
                min_volume=self._config.min_market_volume,
                limit=self._config.max_markets * 3,  # fetch extra, filter later
            )

        # Filter to politics/culture — exclude sports/crypto-minute/leagues
        _SPORTS_LEAGUE_RE = re.compile(
            r"(?i)\b(NBA|NFL|MLB|NHL|MLS|WNBA|Premier League|La Liga|Serie A"
            r"|Bundesliga|Ligue 1|Champions League|Europa League"
            r"|MVP|Ballon d.Or|Cy Young|Heisman"
            r"|Stanley Cup|World Series|Super Bowl|FIFA|World Cup"
            r"|Wimbledon|Roland Garros|French Open|US Open|Masters Tournament"
            r"|UFC|Boxing|F1|Formula|Grand Prix|NASCAR"
            r"|ATP|WTA|PGA|LPGA)\b"
        )
        markets = [
            m for m in active_markets
            if not _EXCLUDED_RE.search(m.question)
            and not _SPORTS_LEAGUE_RE.search(m.question)
        ]
        markets.sort(key=lambda m: m.volume, reverse=True)
        markets = markets[: self._config.max_markets]
        _log(f"Found {len(markets)} qualifying active markets (from {len(active_markets)} total)")

        if not markets:
            return []

        # Build token_id → (condition_id, outcome) AND condition_id → market lookup
        token_lookup: dict[str, tuple[str, str]] = {}
        market_cids: set[str] = set()
        for m in markets:
            market_cids.add(m.condition_id)
            for tok in m.tokens:
                token_lookup[tok.token_id] = (m.condition_id, tok.outcome)
        _log(f"Token lookup: {len(token_lookup)} tokens across {len(market_cids)} markets")

        # Fetch wallet activity in parallel
        _log(f"Fetching activity for {len(wallets)} wallets...")
        wallet_trades: dict[str, list[Trade]] = {}
        sem = asyncio.Semaphore(20)
        done = 0
        start = time.monotonic()

        async with DataAPIClient(self._api_config) as data_api:
            async def fetch_one(addr: str) -> None:
                nonlocal done
                async with sem:
                    try:
                        trades, _, _ = await data_api.get_wallet_activity(addr)
                        wallet_trades[addr] = trades
                    except Exception as e:
                        logger.warning("signal_fetch_failed", wallet=addr, error=str(e))
                    done += 1
                    if done % 10 == 0 or done == len(wallets):
                        _log(f"  [{done}/{len(wallets)}] ({time.monotonic() - start:.0f}s)")

            await asyncio.gather(*[fetch_one(s.address) for s in wallets])

        # Compute net positions: wallet -> condition_id -> outcome -> {shares, cost}
        # Match by token_id (asset_id) when available, fall back to condition_id match
        positions: dict[str, dict[str, dict[str, dict]]] = {}
        matched_tokens = 0
        matched_cids = 0
        for addr, trades in wallet_trades.items():
            for t in trades:
                cid = None
                outcome = None

                # Try token_id match first (gives us the outcome)
                if t.asset_id in token_lookup:
                    cid, outcome = token_lookup[t.asset_id]
                    matched_tokens += 1
                # Fall back to condition_id match (trade.market == conditionId)
                elif t.market in market_cids:
                    cid = t.market
                    # Infer outcome from price: if buying at >0.5, likely YES side
                    outcome = "Yes" if t.price > 0.5 else "No"
                    matched_cids += 1
                else:
                    continue

                pos = (
                    positions
                    .setdefault(addr, {})
                    .setdefault(cid, {})
                    .setdefault(outcome, {"shares": 0.0, "cost": 0.0})
                )
                if t.side == "BUY":
                    pos["shares"] += t.size
                    pos["cost"] += t.size * t.price
                elif t.side == "SELL":
                    pos["shares"] -= t.size

        _log(f"Position matches: {matched_tokens} by token, {matched_cids} by conditionId")

        # Aggregate signals per market
        market_by_cid = {m.condition_id: m for m in markets}
        signals: list[MarketSignal] = []

        for m in markets:
            outcome_signals: dict[str, float] = {}
            wallet_positions: list[WalletPosition] = []

            for addr, wallet_pos in positions.items():
                if m.condition_id not in wallet_pos:
                    continue
                score = score_by_addr.get(addr, 0)
                for outcome, pos in wallet_pos[m.condition_id].items():
                    net = pos["shares"]
                    if abs(net) < 0.01:
                        continue
                    avg_entry = pos["cost"] / max(pos["shares"] + max(-net, 0), 0.01)
                    dollar_size = abs(net) * avg_entry

                    # Dampen large positions with log
                    weight = score * math.log1p(abs(net))
                    if net > 0:
                        outcome_signals[outcome] = outcome_signals.get(outcome, 0) + weight
                    else:
                        # Sold out = bearish on this outcome
                        outcome_signals[outcome] = outcome_signals.get(outcome, 0) - weight

                    wallet_positions.append(WalletPosition(
                        address=addr,
                        composite_score=score,
                        outcome=outcome,
                        net_shares=net,
                        avg_entry=avg_entry,
                        dollar_size=dollar_size,
                    ))

            if not wallet_positions:
                continue

            signals.append(MarketSignal(
                condition_id=m.condition_id,
                question=m.question,
                slug=m.slug,
                volume=m.volume,
                signals=outcome_signals,
                positions=sorted(
                    wallet_positions,
                    key=lambda p: p.composite_score * abs(p.net_shares),
                    reverse=True,
                ),
            ))

        # Sort by total signal strength
        signals.sort(
            key=lambda s: max(abs(v) for v in s.signals.values()) if s.signals else 0,
            reverse=True,
        )
        _log(f"Generated signals for {len(signals)} markets")
        return signals
