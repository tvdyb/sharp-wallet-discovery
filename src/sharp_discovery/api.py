"""Polymarket API clients for fetching markets and trades."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

import aiohttp
import structlog

from sharp_discovery.config import APIConfig
from sharp_discovery.models import Market, MarketToken, Trade

logger = structlog.get_logger()


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate if rate > 0 else 0
        self._last = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        wait = self._interval - (now - self._last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = time.monotonic()


class GammaClient:
    """Client for the Gamma (market metadata) API."""

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._limiter = RateLimiter(config.rate_limit_per_second)

    async def __aenter__(self) -> GammaClient:
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()

    async def _get(self, url: str, params: dict | None = None) -> Any:
        assert self._session is not None
        for attempt in range(self._config.max_retries):
            await self._limiter.acquire()
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        wait = min(2 ** attempt * self._config.backoff_base, 30)
                        logger.warning("rate_limited", wait=wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = min(2 ** attempt * self._config.backoff_base, self._config.backoff_max)
                logger.warning("api_retry", attempt=attempt, error=str(e), wait=wait)
                await asyncio.sleep(wait)
        raise RuntimeError(f"Failed after {self._config.max_retries} retries: {url}")

    async def get_resolved_markets(
        self, min_volume: float = 0.0, limit: int | None = None
    ) -> list[Market]:
        """Fetch all resolved markets, optionally filtered by volume."""
        markets: list[Market] = []
        offset = 0
        page_size = 100

        while True:
            url = f"{self._config.gamma_base_url}/markets"
            params = {
                "limit": page_size,
                "offset": offset,
                "status": "resolved",
            }
            data = await self._get(url, params)
            if not data:
                break

            for m in data:
                vol = float(m.get("volume", 0) or 0)
                if vol < min_volume:
                    continue

                condition_id = m.get("conditionId", m.get("condition_id", ""))
                if not condition_id:
                    continue

                # Build tokens from clobTokenIds + outcomes + outcomePrices
                tokens = self._parse_tokens(m)
                if not tokens:
                    continue

                # Need a clear winner (outcomePrices > 0.9)
                if not any(t.winner for t in tokens):
                    continue

                res_date = None
                for date_key in ("endDate", "closedTime", "resolutionDate",
                                 "resolution_date", "end_date"):
                    raw = m.get(date_key)
                    if raw:
                        try:
                            ds = str(raw).replace("Z", "+00:00")
                            if "+" in ds:
                                ds = ds.split("+")[0]
                            res_date = datetime.fromisoformat(ds)
                            break
                        except (ValueError, TypeError):
                            continue

                markets.append(Market(
                    condition_id=condition_id,
                    question=m.get("question", ""),
                    slug=m.get("slug", ""),
                    outcome=m.get("outcome", ""),
                    resolution_date=res_date,
                    tokens=tokens,
                    volume=vol,
                ))

            offset += page_size
            if len(data) < page_size:
                break
            if limit and len(markets) >= limit:
                markets = markets[:limit]
                break

        logger.info("markets_fetched", count=len(markets))
        return markets

    @staticmethod
    def _parse_tokens(m: dict) -> list[MarketToken]:
        """Parse tokens from Gamma API response.

        The API returns clobTokenIds (list of token ID strings),
        outcomes (["Yes", "No"]), and outcomePrices (prices as strings).
        Winner is the outcome with price > 0.9 after resolution.
        Also handles the alternative 'tokens' array format.
        """
        import json as _json

        # Try the nested 'tokens' format first (some endpoints)
        if "tokens" in m and isinstance(m["tokens"], list) and m["tokens"]:
            first = m["tokens"][0]
            if isinstance(first, dict) and ("token_id" in first or "tokenId" in first):
                return [
                    MarketToken(
                        token_id=t.get("token_id", t.get("tokenId", "")),
                        outcome=t.get("outcome", ""),
                        winner=bool(t.get("winner", False)),
                    )
                    for t in m["tokens"]
                ]

        # Parse clobTokenIds
        clob_ids = m.get("clobTokenIds", [])
        if isinstance(clob_ids, str):
            try:
                clob_ids = _json.loads(clob_ids) if clob_ids else []
            except _json.JSONDecodeError:
                clob_ids = []
        if not clob_ids:
            return []

        # Parse outcomes
        outcomes = m.get("outcomes", [])
        if isinstance(outcomes, str):
            try:
                outcomes = _json.loads(outcomes) if outcomes else []
            except _json.JSONDecodeError:
                outcomes = []

        # Parse outcomePrices to determine winner
        prices_raw = m.get("outcomePrices", [])
        if isinstance(prices_raw, str):
            try:
                prices_raw = _json.loads(prices_raw) if prices_raw else []
            except _json.JSONDecodeError:
                prices_raw = []

        tokens = []
        for i, tid in enumerate(clob_ids):
            outcome = outcomes[i] if i < len(outcomes) else ""
            try:
                price = float(prices_raw[i]) if i < len(prices_raw) else 0.0
            except (ValueError, TypeError):
                price = 0.0
            tokens.append(MarketToken(
                token_id=str(tid),
                outcome=outcome,
                winner=price > 0.9,
            ))
        return tokens


class DataAPIClient:
    """Client for the Polymarket Data API (trade history)."""

    DATA_API_URL = "https://data-api.polymarket.com"

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._limiter = RateLimiter(config.rate_limit_per_second)

    async def __aenter__(self) -> DataAPIClient:
        timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()

    async def _get(self, url: str, params: dict | None = None) -> Any:
        assert self._session is not None
        for attempt in range(self._config.max_retries):
            await self._limiter.acquire()
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        wait = min(2 ** attempt * self._config.backoff_base, 30)
                        logger.warning("rate_limited", wait=wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = min(2 ** attempt * self._config.backoff_base, self._config.backoff_max)
                logger.warning("api_retry", attempt=attempt, error=str(e), wait=wait)
                await asyncio.sleep(wait)
        raise RuntimeError(f"Failed after {self._config.max_retries} retries: {url}")

    async def get_trades_for_market(
        self, condition_id: str, limit: int = 500
    ) -> list[Trade]:
        """Fetch trades for a market via the Data API.

        Uses `market=` param which filters by conditionId.
        Returns trades with wallet address, asset (token ID), side, size, price.
        """
        all_trades: list[Trade] = []
        offset = 0

        while len(all_trades) < limit:
            batch_size = min(500, limit - len(all_trades))
            url = f"{self.DATA_API_URL}/trades"
            params = {
                "market": condition_id,
                "limit": batch_size,
                "offset": offset,
            }
            data = await self._get(url, params)
            if not data or not isinstance(data, list):
                break

            for t in data:
                ts = None
                raw_ts = t.get("timestamp")
                if raw_ts:
                    try:
                        ts = datetime.utcfromtimestamp(int(raw_ts))
                    except (ValueError, TypeError, OSError):
                        try:
                            ts = datetime.fromisoformat(
                                str(raw_ts).replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                all_trades.append(Trade(
                    id=t.get("transactionHash", ""),
                    market=t.get("conditionId", condition_id),
                    asset_id=t.get("asset", ""),
                    side=t.get("side", ""),
                    size=float(t.get("size", 0)),
                    price=float(t.get("price", 0)),
                    timestamp=ts,
                    owner=t.get("proxyWallet", ""),
                ))

            if len(data) < batch_size:
                break
            offset += batch_size

        return all_trades
