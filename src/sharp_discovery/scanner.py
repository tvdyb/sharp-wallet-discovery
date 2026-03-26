"""Wallet scanner: fetch markets, analyze trades, score wallets."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime

import structlog

from sharp_discovery.api import DataAPIClient, GammaClient
from sharp_discovery.config import DiscoveryConfig
from sharp_discovery.db import Database
from sharp_discovery.goldsky import GoldskyClient
from sharp_discovery.models import Market, MarketToken, Trade, WalletMarketResult, WalletScore
from sharp_discovery.scorer import compute_wallet_score

logger = structlog.get_logger()

SAVE_EVERY = 500  # save cache every N markets

# Regex to identify sports/esports/crypto-minute markets we want to exclude
_EXCLUDED_RE = re.compile(
    r"(?i)"
    # Sports: matchups and lines
    r"(\bvs\.?\s)"                              # "X vs Y" or "X vs. Y"
    r"|(\bo/u\b)"                               # over/under lines
    r"|(^spread:)"                              # spread lines
    r"|(^total:)"                               # total lines
    r"|(^1H\s)"                                 # first-half lines
    r"|(will .+(?:win|lose|draw) on \d{4})"     # "Will X win on 2026-03-21?"
    r"|(\bFC\b)"                                # football club
    r"|(\bAFC\b)"                               # AFC
    r"|(\bCF\b)"                                # club de futbol
    r"|(\bSC\b.*(?:win|vs))"                    # sporting club
    # Esports
    r"|\b(lol|cs2|dota|valorant):"              # esports prefixes
    r"|\b(LEC|LCK|LPL|LCS)\b"                  # esports leagues
    r"|\b(BO[35])\b"                            # best-of series
    # Combat sports
    r"|\b(UFC|MMA)\s+\d"                        # UFC events
    r"|(Grand Prix)"                            # F1
    # Golf / individual sport results
    r"|(finish in the Top)"                     # "Will X finish in the Top 10"
    r"|(win the \d{4}\s+\w+\s+(?:Championship|Open|Classic|Invitational|Tournament|Masters|Cup|Trophy|Prix))"
    r"|(\bPoints O/U\b)"                        # player props "Rudy Gobert: Points O/U"
    r"|(\bO/U \d)"                              # generic over/under with number
    r"|(\bMatch O/U\b)"                         # tennis match over/under
    r"|(\bTotal Sets\b)"                        # tennis sets
    r"|(\bfastest lap\b)"                       # F1
    # Crypto/price minute-candle markets (short-term price direction bets)
    r"|(\b(?:Bitcoin|Ethereum|BTC|ETH|SOL|XRP)\s+Up or Down\b)"
    r"|(\b(?:Up or Down)\b.*\d+:\d+\s*[AP]M)"  # "Up or Down - March 24, 2:00PM"
)


def _log(msg: str) -> None:
    print(msg, flush=True)


class WalletScanner:
    """Scans resolved Polymarket markets and scores wallets."""

    def __init__(self, config: DiscoveryConfig, db: Database) -> None:
        self._config = config
        self._db = db

    async def run(self) -> list[WalletScore]:
        """Run the full discovery pipeline.

        Default: leaderboard-first approach — pull top profitable wallets
        from Polymarket leaderboard, then fetch their trade history.
        """
        wallet_data = await self._fetch_via_leaderboard()
        return await self._score_wallets(wallet_data)

    async def _fetch_via_leaderboard(self) -> dict[str, list[WalletMarketResult]]:
        """Leaderboard-first pipeline — no Gamma needed.

        1. Pull top wallets by PnL from leaderboard
        2. Fetch each wallet's activity (trades + redeems) in parallel
        3. Use REDEEM events to determine wins (no Gamma market metadata needed)
        """
        _log("=== Leaderboard-first pipeline ===")

        async with DataAPIClient(self._config.api) as data_api:
            # Step 1: Pull top wallets from leaderboard
            _log("Fetching leaderboard...")
            candidates: list[dict] = []
            for offset in range(0, self._config.leaderboard_depth, 50):
                batch = await data_api.get_leaderboard(
                    time_period="ALL", order_by="PNL",
                    limit=50, offset=offset,
                )
                candidates.extend(batch)
                if len(batch) < 50:
                    break

            # Filter to profitable wallets with volume
            wallets = []
            for c in candidates:
                pnl = float(c.get("pnl", 0))
                vol = float(c.get("vol", 0))
                if pnl > 0 and vol >= self._config.min_volume:
                    wallets.append({
                        "address": c["proxyWallet"],
                        "name": c.get("userName", ""),
                        "pnl": pnl,
                        "vol": vol,
                    })
            _log(f"Found {len(wallets)} profitable wallets (from {len(candidates)} leaderboard entries)")

            # Step 2: Fetch trade history for each wallet in parallel
            _log(f"Fetching trade history for {len(wallets)} wallets...")
            wallet_data: dict[str, list[WalletMarketResult]] = {}
            sem = asyncio.Semaphore(20)
            lock = asyncio.Lock()
            done = 0
            start_time = time.monotonic()

            async def fetch_wallet(w: dict) -> None:
                nonlocal done
                addr = w["address"]
                async with sem:
                    try:
                        trades, redeemed_cids, market_titles = await data_api.get_wallet_activity(addr)
                    except Exception as e:
                        logger.warning("wallet_fetch_failed", wallet=addr, error=str(e))
                        async with lock:
                            done += 1
                        return

                # Filter out sports/esports markets
                if self._config.scoring.exclude_sports and market_titles:
                    sports_cids = {
                        cid for cid, title in market_titles.items()
                        if _EXCLUDED_RE.search(title)
                    }
                    # Skip wallet entirely if >50% of their markets are excluded categories
                    sports_ratio = len(sports_cids) / len(market_titles)
                    if sports_ratio > 0.50:
                        async with lock:
                            done += 1
                        logger.info(
                            "sports_wallet_filtered",
                            wallet=addr,
                            sports_ratio=f"{sports_ratio:.0%}",
                            sports_markets=len(sports_cids),
                            total_markets=len(market_titles),
                        )
                        return
                    trades = [t for t in trades if t.market not in sports_cids]
                    redeemed_cids = redeemed_cids - sports_cids

                # Group trades by market
                trades_by_market: dict[str, list[Trade]] = {}
                for t in trades:
                    if t.market:
                        trades_by_market.setdefault(t.market, []).append(t)

                # Market maker filter: check both-sides ratio and trades-per-market
                if trades_by_market:
                    both_sides_count = 0
                    total_trade_count = sum(len(ts) for ts in trades_by_market.values())
                    for cid, mkt_trades in trades_by_market.items():
                        asset_ids = {t.asset_id for t in mkt_trades if t.asset_id}
                        if len(asset_ids) > 1:
                            both_sides_count += 1
                    both_sides_ratio = both_sides_count / len(trades_by_market)
                    trades_per_market = total_trade_count / len(trades_by_market)

                    scoring = self._config.scoring
                    if (both_sides_ratio > scoring.max_both_sides_ratio
                            or trades_per_market > scoring.max_trades_per_market):
                        async with lock:
                            done += 1
                        logger.info(
                            "market_maker_filtered",
                            wallet=addr,
                            both_sides_ratio=f"{both_sides_ratio:.0%}",
                            trades_per_market=f"{trades_per_market:.1f}",
                        )
                        return

                # Analyze each market using redeem-based win detection
                # - Redeemed = won
                # - Not redeemed = lost OR still open
                #   We include non-redeemed markets as losses. This slightly
                #   over-penalizes (some are genuinely still open), but prevents
                #   the "only count wins" bias that makes every wallet look 100%.
                results: list[WalletMarketResult] = []
                for cid, market_trades in trades_by_market.items():
                    r = self._analyze_wallet_market(
                        addr, cid, market_trades, won=cid in redeemed_cids
                    )
                    if r:
                        results.append(r)

                async with lock:
                    if results:
                        wallet_data[addr] = results
                    done += 1
                    if done % 20 == 0 or done == len(wallets):
                        elapsed = time.monotonic() - start_time
                        _log(
                            f"  [{done}/{len(wallets)}] {len(wallet_data)} with results, "
                            f"{sum(len(v) for v in wallet_data.values())} market-results "
                            f"({elapsed:.0f}s)"
                        )

            await asyncio.gather(*[fetch_wallet(w) for w in wallets])

        _log(f"Analyzed {len(wallet_data)} wallets with trade data")
        return wallet_data

    @staticmethod
    def _analyze_wallet_market(
        wallet: str,
        condition_id: str,
        trades: list[Trade],
        won: bool,
    ) -> WalletMarketResult | None:
        """Analyze a wallet's trades in a single market.

        Uses REDEEM-based win detection instead of market metadata.
        """
        buys = [t for t in trades if t.side == "BUY"]
        sells = [t for t in trades if t.side == "SELL"]
        total_bought = sum(t.size for t in buys)
        total_sold = sum(t.size for t in sells)

        if total_bought == 0:
            return None

        buy_cost = sum(t.price * t.size for t in buys)
        avg_entry = buy_cost / total_bought

        trade_dates = [t.timestamp for t in trades if t.timestamp]
        last_trade = max(trade_dates) if trade_dates else None

        exit_ratio = total_sold / total_bought
        held_to_expiration = exit_ratio < 0.5
        net_position = total_bought - total_sold

        if net_position > 0:
            payout = net_position * 1.0 if won else 0.0
            cost = net_position * avg_entry if avg_entry > 0 else 0.0
            roi = (payout - cost) / cost if cost > 0 else 0.0
        else:
            sell_revenue = sum(t.price * t.size for t in sells)
            cost = buy_cost
            roi = (sell_revenue - cost) / cost if cost > 0 else 0.0
            won = roi > 0

        return WalletMarketResult(
            wallet=wallet,
            market_id=condition_id,
            won=won,
            roi=roi,
            avg_entry=avg_entry,
            held_to_expiration=held_to_expiration,
            total_bought=total_bought,
            total_sold=total_sold,
            resolution_date=None,
            last_trade_date=last_trade,
        )

    async def _fetch_via_goldsky(self) -> dict[str, list[WalletMarketResult]]:
        """Fetch markets from Gamma, trades from Goldsky subgraph."""
        _log("=== Goldsky pipeline ===")

        # Step 1: Get resolved markets from Gamma for metadata + token mapping
        async with GammaClient(self._config.api) as gamma:
            limit = self._config.max_markets or None
            _log("Fetching resolved markets from Gamma API...")
            markets = await gamma.get_resolved_markets(
                min_volume=self._config.min_volume,
                limit=limit,
            )
            _log(f"Found {len(markets)} resolved markets")

        # Step 2: Build token_id → condition_id lookup
        token_to_market: dict[str, str] = {}
        market_by_cid: dict[str, Market] = {}
        for m in markets:
            market_by_cid[m.condition_id] = m
            for t in m.tokens:
                token_to_market[t.token_id] = m.condition_id
        _log(f"Built token→market map: {len(token_to_market)} tokens across {len(market_by_cid)} markets")

        # Step 3: Fetch trades from Goldsky
        async with GoldskyClient(
            since=self._config.goldsky_since,
            cache_path=self._config.goldsky_cache,
            token_to_market=token_to_market,
        ) as goldsky:
            all_trades = await goldsky.fetch_trades()
            _log(f"Total Goldsky trades: {len(all_trades)}")

        # Step 4: Group trades by market and analyze
        trades_by_market: dict[str, list[Trade]] = {}
        unmapped = 0
        for trade in all_trades:
            # Resolve market from token if not set
            if not trade.market:
                cid = token_to_market.get(trade.asset_id, "")
                if cid:
                    trade.market = cid
                else:
                    unmapped += 1
                    continue
            trades_by_market.setdefault(trade.market, []).append(trade)

        if unmapped:
            _log(f"  {unmapped} trades with unmapped tokens (skipped)")
        _log(f"  Trades mapped to {len(trades_by_market)} markets")

        wallet_data: dict[str, list[WalletMarketResult]] = {}
        for cid, trades in trades_by_market.items():
            market = market_by_cid.get(cid)
            if not market:
                continue
            results = self._analyze_trades(market, trades)
            for r in results:
                wallet_data.setdefault(r.wallet, []).append(r)

        _log(f"Analyzed {len(wallet_data)} unique wallets")
        return wallet_data

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
            sem = asyncio.Semaphore(50)
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

            # Process in batches of 500 — semaphore controls actual concurrency
            batch_size = 500
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
