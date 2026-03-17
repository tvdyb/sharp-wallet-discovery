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
                vol = float(m.get("volume", 0))
                if vol < min_volume:
                    continue

                tokens = [
                    MarketToken(
                        token_id=t["token_id"],
                        outcome=t.get("outcome", ""),
                        winner=t.get("winner", False),
                    )
                    for t in m.get("tokens", [])
                ]
                res_date = None
                if m.get("resolution_date"):
                    try:
                        res_date = datetime.fromisoformat(
                            str(m["resolution_date"]).replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass

                markets.append(Market(
                    condition_id=m["condition_id"],
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


class ClobClient:
    """Client for the CLOB (trade data) API."""

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._limiter = RateLimiter(config.rate_limit_per_second)

    async def __aenter__(self) -> ClobClient:
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

    async def get_trades(self, asset_id: str, limit: int = 500) -> list[Trade]:
        """Fetch trades for a specific token."""
        url = f"{self._config.clob_base_url}/trades"
        params = {"limit": limit, "asset_id": asset_id}
        data = await self._get(url, params)
        if not data:
            return []

        trades = []
        for t in data:
            ts = None
            if t.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(str(t["timestamp"]).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            trades.append(Trade(
                id=t.get("id", ""),
                market=t.get("market", ""),
                asset_id=t.get("asset_id", asset_id),
                side=t.get("side", ""),
                size=float(t.get("size", 0)),
                price=float(t.get("price", 0)),
                timestamp=ts,
                owner=t.get("owner", ""),
            ))
        return trades
