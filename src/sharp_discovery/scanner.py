"""Wallet scanner: fetch markets, analyze trades, score wallets."""

from __future__ import annotations

import asyncio
import time

import structlog

from sharp_discovery.api import DataAPIClient, GammaClient
from sharp_discovery.config import DiscoveryConfig
from sharp_discovery.db import Database
from sharp_discovery.models import Market, Trade, WalletMarketResult, WalletScore
from sharp_discovery.scorer import compute_wallet_score

logger = structlog.get_logger()


def _log(msg: str) -> None:
    print(msg, flush=True)


class WalletScanner:
    """Scans resolved Polymarket markets and scores wallets."""

    def __init__(self, config: DiscoveryConfig, db: Database) -> None:
        self._config = config
        self._db = db

    async def run(self) -> list[WalletScore]:
        """Run the full discovery pipeline.

        1. Fetch all resolved markets (filtered by volume)
        2. For each market, fetch trades and build per-wallet results
        3. Score each wallet with dual-path scoring + extreme price penalty
        4. Persist to database
        """
        async with (
            GammaClient(self._config.api) as gamma,
            DataAPIClient(self._config.api) as data_api,
        ):
            _log("Fetching resolved markets...")
            markets = await gamma.get_resolved_markets(
                min_volume=self._config.min_volume
            )
            _log(f"Found {len(markets)} resolved markets (volume >= ${self._config.min_volume:,.0f})")

            wallet_data: dict[str, list[WalletMarketResult]] = {}
            start_time = time.monotonic()
            ok_count = 0
            fail_count = 0
            total_trades = 0

            for i, market in enumerate(markets, 1):
                try:
                    results = await self._analyze_market(market, data_api)
                    for r in results:
                        wallet_data.setdefault(r.wallet, []).append(r)
                    ok_count += 1
                    total_trades += len(results)
                except Exception as e:
                    fail_count += 1
                    logger.warning("market_failed", market=market.condition_id, error=str(e))

                if i % 50 == 0 or i == len(markets):
                    elapsed = time.monotonic() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(markets) - i) / rate if rate > 0 else 0
                    _log(
                        f"  [{i}/{len(markets)}] ok={ok_count} fail={fail_count} "
                        f"wallets={len(wallet_data)} trades={total_trades} | "
                        f"{rate:.1f} mkts/s, ETA {eta:.0f}s"
                    )

                # Rate limit: ~2 requests per second
                await asyncio.sleep(0.5)

            _log(f"\nScoring {len(wallet_data)} wallets...")
            scores: list[WalletScore] = []
            sharpe_count = 0
            consistency_count = 0

            for wallet, results in wallet_data.items():
                score = compute_wallet_score(wallet, results, self._config.scoring)
                if score is not None:
                    scores.append(score)
                    await self._db.upsert_wallet_score(score)
                    if score.sharpe_ratio != 0.0:
                        sharpe_count += 1
                    else:
                        consistency_count += 1

            scores.sort(key=lambda s: s.composite_score, reverse=True)
            _log(
                f"Qualified: {len(scores)} wallets "
                f"({sharpe_count} Sharpe, {consistency_count} consistency) "
                f"out of {len(wallet_data)} checked"
            )
            return scores

    async def _analyze_market(
        self, market: Market, data_api: DataAPIClient
    ) -> list[WalletMarketResult]:
        """Analyze a single resolved market to extract wallet results."""
        results: list[WalletMarketResult] = []
        winner_token_ids = {t.token_id for t in market.tokens if t.winner}
        all_token_ids = {t.token_id for t in market.tokens}

        # Fetch all trades for this market in one call
        trades = await data_api.get_trades_for_market(
            condition_id=market.condition_id, limit=500
        )

        # Group trades by (wallet, token)
        wallet_token_trades: dict[str, dict[str, list[Trade]]] = {}
        for trade in trades:
            if not trade.owner or not trade.asset_id:
                continue
            wallet_token_trades.setdefault(trade.owner, {}).setdefault(
                trade.asset_id, []
            ).append(trade)

        for wallet, token_trades in wallet_token_trades.items():
            for token_id, wtrades in token_trades.items():
                buys = [t for t in wtrades if t.side == "BUY"]
                sells = [t for t in wtrades if t.side == "SELL"]
                total_bought = sum(t.size for t in buys)
                total_sold = sum(t.size for t in sells)
                net_position = total_bought - total_sold

                if total_bought == 0:
                    continue

                # Volume-weighted average entry price
                buy_cost = sum(t.price * t.size for t in buys)
                avg_entry = buy_cost / total_bought

                # Held to expiration: kept >50% of position through resolution
                exit_ratio = total_sold / total_bought
                held_to_expiration = exit_ratio < 0.5

                is_winner_token = token_id in winner_token_ids

                if net_position > 0:
                    won = is_winner_token
                    payout = net_position * 1.0 if is_winner_token else 0.0
                    cost = net_position * avg_entry if avg_entry > 0 else 0.0
                    roi = (payout - cost) / cost if cost > 0 else 0.0
                else:
                    sell_revenue = sum(t.price * t.size for t in sells)
                    cost = buy_cost
                    roi = (sell_revenue - cost) / cost if cost > 0 else 0.0
                    won = roi > 0

                results.append(WalletMarketResult(
                    wallet=wallet,
                    market_id=market.condition_id,
                    won=won,
                    roi=roi,
                    avg_entry=avg_entry,
                    held_to_expiration=held_to_expiration,
                    total_bought=total_bought,
                    total_sold=total_sold,
                    resolution_date=market.resolution_date,
                ))

        return results
