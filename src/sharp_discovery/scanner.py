"""Wallet scanner: fetch markets, analyze trades, score wallets."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime

import structlog

from sharp_discovery.api import DataAPIClient, GammaClient
from sharp_discovery.config import DiscoveryConfig
from sharp_discovery.db import Database
from sharp_discovery.models import Market, MarketToken, Trade, WalletMarketResult, WalletScore
from sharp_discovery.scorer import compute_wallet_score

logger = structlog.get_logger()

SAVE_EVERY = 500  # save cache every N markets


def _log(msg: str) -> None:
    print(msg, flush=True)


class WalletScanner:
    """Scans resolved Polymarket markets and scores wallets."""

    def __init__(self, config: DiscoveryConfig, db: Database) -> None:
        self._config = config
        self._db = db

    async def run(self) -> list[WalletScore]:
        """Run the full discovery pipeline.

        If a complete cache file exists, loads from cache (instant).
        Otherwise fetches from API with incremental saving, then scores.
        """
        cache_path = self._config.cache_path

        if cache_path and os.path.exists(cache_path):
            cache = self._read_cache(cache_path)
            if cache.get("complete"):
                _log(f"Loading complete cache from {cache_path}...")
                wallet_data = self._parse_cache(cache)
                _log(f"Loaded {len(wallet_data)} wallets from cache")
                return await self._score_wallets(wallet_data)
            else:
                _log(f"Found partial cache ({len(cache.get('markets', []))} markets), resuming fetch...")

        wallet_data = await self._fetch_all(cache_path)
        return await self._score_wallets(wallet_data)

    async def _fetch_all(self, cache_path: str) -> dict[str, list[WalletMarketResult]]:
        """Fetch all markets and trades from API, save to cache incrementally."""
        # Load partial cache if it exists
        done_cids: set[str] = set()
        cache_markets: list[dict] = []
        cache_trades: dict[str, list[dict]] = {}
        cached_market_list: list[dict] | None = None

        if cache_path and os.path.exists(cache_path):
            partial = self._read_cache(cache_path)
            cache_markets = partial.get("markets", [])
            cache_trades = partial.get("trades", {})
            cached_market_list = partial.get("market_list")
            done_cids = {m["condition_id"] for m in cache_markets}
            _log(f"  Resuming: {len(done_cids)} markets already cached")

        async with (
            GammaClient(self._config.api) as gamma,
            DataAPIClient(self._config.api) as data_api,
        ):
            # Use cached market list if available, otherwise fetch from Gamma
            if cached_market_list is not None:
                _log(f"Using cached market list ({len(cached_market_list)} markets)")
                markets = [self._dict_to_market(m) for m in cached_market_list]
            else:
                _log("Fetching resolved markets from Gamma API...")
                limit = self._config.max_markets or None  # 0 = unlimited
                markets = await gamma.get_resolved_markets(
                    min_volume=self._config.min_volume,
                    limit=limit,
                )
                limit_str = str(limit) if limit else "unlimited"
                _log(
                    f"Found {len(markets)} resolved markets "
                    f"(volume >= ${self._config.min_volume:,.0f}, "
                    f"max {limit_str})"
                )
                # Cache the market list for future resumes
                cached_market_list = [
                    {
                        "condition_id": m.condition_id,
                        "question": m.question,
                        "slug": m.slug,
                        "outcome": m.outcome,
                        "resolution_date": m.resolution_date.isoformat() if m.resolution_date else None,
                        "volume": m.volume,
                        "tokens": [{"token_id": t.token_id, "outcome": t.outcome, "winner": t.winner} for t in m.tokens],
                    }
                    for m in markets
                ]
                if cache_path:
                    self._write_cache(cache_path, cache_markets, cache_trades, complete=False, market_list=cached_market_list)
                    _log(f"  Saved market list to cache")

            # Filter out already-cached markets
            remaining = [m for m in markets if m.condition_id not in done_cids]
            _log(f"  {len(remaining)} markets to fetch ({len(done_cids)} already cached)")

            start_time = time.monotonic()
            ok_count = 0
            fail_count = 0
            total_trades = 0
            last_save = 0
            processed = 0
            sem = asyncio.Semaphore(20)
            lock = asyncio.Lock()

            async def fetch_one(market: Market) -> None:
                nonlocal ok_count, fail_count, total_trades, processed, last_save
                async with sem:
                    try:
                        raw_trades = await data_api.get_trades_for_market(
                            condition_id=market.condition_id, limit=500
                        )
                        async with lock:
                            ok_count += 1
                            total_trades += len(raw_trades)

                            cache_markets.append({
                                "condition_id": market.condition_id,
                                "question": market.question,
                                "slug": market.slug,
                                "outcome": market.outcome,
                                "resolution_date": market.resolution_date.isoformat()
                                if market.resolution_date else None,
                                "volume": market.volume,
                                "tokens": [
                                    {"token_id": t.token_id, "outcome": t.outcome, "winner": t.winner}
                                    for t in market.tokens
                                ],
                            })
                            cache_trades[market.condition_id] = [
                                {
                                    "id": t.id, "market": t.market,
                                    "asset_id": t.asset_id, "side": t.side,
                                    "size": t.size, "price": t.price,
                                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                                    "owner": t.owner,
                                }
                                for t in raw_trades
                            ]
                    except Exception as e:
                        async with lock:
                            fail_count += 1
                        logger.warning("market_failed", market=market.condition_id, error=str(e))

                    async with lock:
                        processed += 1
                        if processed % 50 == 0 or processed == len(remaining):
                            elapsed = time.monotonic() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            eta = (len(remaining) - processed) / rate if rate > 0 else 0
                            eta_m = eta / 60
                            _log(
                                f"  [{processed}/{len(remaining)}] ok={ok_count} fail={fail_count} "
                                f"cached={len(cache_markets)} trades={total_trades} | "
                                f"{rate:.1f} mkts/s, ETA {eta_m:.1f}m"
                            )

                        if cache_path and (ok_count - last_save) >= SAVE_EVERY:
                            self._write_cache(cache_path, cache_markets, cache_trades, complete=False, market_list=cached_market_list)
                            last_save = ok_count
                            _log(f"  [checkpoint] saved {len(cache_markets)} markets to cache")

            # Process in batches of 100 to avoid creating too many tasks at once
            batch_size = 100
            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start:batch_start + batch_size]
                await asyncio.gather(*[fetch_one(m) for m in batch])

            # Final save — mark complete
            if cache_path:
                self._write_cache(cache_path, cache_markets, cache_trades, complete=True, market_list=cached_market_list)
                _log(f"\nCached {len(cache_markets)} markets to {cache_path} (complete)")

        # Now parse the full cache into wallet_data
        wallet_data: dict[str, list[WalletMarketResult]] = {}
        for mkt_dict in cache_markets:
            cid = mkt_dict["condition_id"]
            market = self._dict_to_market(mkt_dict)
            trades = self._dicts_to_trades(cache_trades.get(cid, []))
            results = self._analyze_trades(market, trades)
            for r in results:
                wallet_data.setdefault(r.wallet, []).append(r)

        return wallet_data

    @staticmethod
    def _write_cache(
        cache_path: str,
        cache_markets: list[dict],
        cache_trades: dict[str, list[dict]],
        complete: bool,
        market_list: list[dict] | None = None,
    ) -> None:
        cache = {
            "markets": cache_markets,
            "trades": cache_trades,
            "fetched_at": datetime.utcnow().isoformat(),
            "complete": complete,
        }
        if market_list is not None:
            cache["market_list"] = market_list
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, cache_path)

    @staticmethod
    def _read_cache(cache_path: str) -> dict:
        with open(cache_path) as f:
            return json.load(f)

    @staticmethod
    def _dict_to_market(mkt: dict) -> Market:
        tokens = [
            MarketToken(
                token_id=t["token_id"],
                outcome=t.get("outcome", ""),
                winner=t.get("winner", False),
            )
            for t in mkt.get("tokens", [])
        ]
        res_date = None
        if mkt.get("resolution_date"):
            try:
                res_date = datetime.fromisoformat(mkt["resolution_date"])
            except (ValueError, TypeError):
                pass
        return Market(
            condition_id=mkt["condition_id"],
            question=mkt.get("question", ""),
            slug=mkt.get("slug", ""),
            outcome=mkt.get("outcome", ""),
            resolution_date=res_date,
            tokens=tokens,
            volume=mkt.get("volume", 0),
        )

    @staticmethod
    def _dicts_to_trades(raw: list[dict]) -> list[Trade]:
        trades = []
        for t in raw:
            ts = None
            if t.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(t["timestamp"])
                except (ValueError, TypeError):
                    pass
            trades.append(Trade(
                id=t.get("id", ""),
                market=t.get("market", ""),
                asset_id=t.get("asset_id", ""),
                side=t.get("side", ""),
                size=float(t.get("size", 0)),
                price=float(t.get("price", 0)),
                timestamp=ts,
                owner=t.get("owner", ""),
            ))
        return trades

    def _load_cache(self, cache_path: str) -> dict[str, list[WalletMarketResult]]:
        """Load wallet results from cached JSON."""
        cache = self._read_cache(cache_path)
        return self._parse_cache(cache)

    def _parse_cache(self, cache: dict) -> dict[str, list[WalletMarketResult]]:
        fetched_at = cache.get("fetched_at", "unknown")
        _log(f"  Cache from {fetched_at}")
        _log(f"  {len(cache['markets'])} markets, {len(cache['trades'])} with trades")

        wallet_data: dict[str, list[WalletMarketResult]] = {}
        for mkt in cache["markets"]:
            cid = mkt["condition_id"]
            market = self._dict_to_market(mkt)
            trades = self._dicts_to_trades(cache["trades"].get(cid, []))
            results = self._analyze_trades(market, trades)
            for r in results:
                wallet_data.setdefault(r.wallet, []).append(r)

        return wallet_data

    async def _score_wallets(
        self, wallet_data: dict[str, list[WalletMarketResult]]
    ) -> list[WalletScore]:
        """Score all wallets and persist to DB."""
        _log(f"Scoring {len(wallet_data)} wallets...")
        scores: list[WalletScore] = []
        sharpe_count = 0
        consistency_count = 0

        for wallet, results in wallet_data.items():
            score = compute_wallet_score(wallet, results, self._config.scoring, self._config.min_recency_days)
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

    @staticmethod
    def _analyze_trades(
        market: Market, trades: list[Trade]
    ) -> list[WalletMarketResult]:
        """Analyze trades for a single market to extract wallet results."""
        results: list[WalletMarketResult] = []
        winner_token_ids = {t.token_id for t in market.tokens if t.winner}

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

                buy_cost = sum(t.price * t.size for t in buys)
                avg_entry = buy_cost / total_bought

                # Track most recent trade
                trade_dates = [t.timestamp for t in wtrades if t.timestamp]
                last_trade = max(trade_dates) if trade_dates else None

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
                    last_trade_date=last_trade,
                ))

        return results
