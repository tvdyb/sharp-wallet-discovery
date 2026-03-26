"""Goldsky subgraph scraper for on-chain Polymarket trade data."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Any

import aiohttp
import structlog

from sharp_discovery.models import Trade

logger = structlog.get_logger()

GOLDSKY_ENDPOINT = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/subgraphs/"
    "orderbook-subgraph/0.0.1/gn"
)

# Platform wallets to filter out (exchange / contract addresses)
PLATFORM_WALLETS = frozenset({
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
})

# USDC asset identifier in the subgraph
USDC_ASSET_ID = "0"

# GraphQL page size
PAGE_SIZE = 1000

# CSV columns
CSV_COLUMNS = [
    "id", "market", "asset_id", "side", "size", "price",
    "timestamp", "owner",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


class GoldskyClient:
    """Fetches trade data from the Goldsky Polymarket subgraph."""

    def __init__(
        self,
        since: str = "",
        cache_path: str = "goldsky_trades.csv",
        token_to_market: dict[str, str] | None = None,
    ) -> None:
        self._since = since  # YYYY-MM-DD
        self._cache_path = cache_path
        self._token_to_market = token_to_market or {}
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> GoldskyClient:
        timeout = aiohttp.ClientTimeout(total=60.0)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()

    async def fetch_trades(self) -> list[Trade]:
        """Fetch all trades from Goldsky, with CSV caching and resume.

        Returns parsed Trade objects for non-USDC, non-platform trades.
        """
        # Resume from CSV if it exists
        existing_trades, last_ts = self._load_csv_cache()
        if existing_trades:
            _log(f"  Goldsky: resuming from {len(existing_trades)} cached trades (last ts={last_ts})")

        # Determine start timestamp
        if last_ts > 0:
            start_ts = last_ts
        elif self._since:
            start_ts = int(datetime.strptime(self._since, "%Y-%m-%d").timestamp())
        else:
            # Default: 6 months ago
            start_ts = int(time.time()) - 180 * 86400

        new_trades = await self._fetch_from_subgraph(start_ts)

        # Deduplicate by trade ID
        seen_ids = {t.id for t in existing_trades}
        for t in new_trades:
            if t.id not in seen_ids:
                existing_trades.append(t)
                seen_ids.add(t.id)

        # Save full set to CSV
        if self._cache_path:
            self._save_csv(existing_trades)
            _log(f"  Goldsky: saved {len(existing_trades)} trades to {self._cache_path}")

        return existing_trades

    async def _fetch_from_subgraph(self, start_ts: int) -> list[Trade]:
        """Page through orderFilledEvents from the Goldsky subgraph.

        Uses timestamp_gte + id_gt for pagination. The timestamp filter
        is always kept so we don't pull events from before our window.
        """
        assert self._session is not None
        trades: list[Trade] = []
        last_id = ""
        total_events = 0
        fetch_start = time.monotonic()
        save_interval = 50000  # save CSV checkpoint every N events

        while True:
            query = self._build_query(start_ts, last_id)

            try:
                async with self._session.post(
                    GOLDSKY_ENDPOINT,
                    json={"query": query},
                ) as resp:
                    if resp.status == 429:
                        _log("  Goldsky: rate limited, waiting 5s...")
                        import asyncio
                        await asyncio.sleep(5)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
            except (aiohttp.ClientError, Exception) as e:
                logger.warning("goldsky_fetch_error", error=str(e), last_id=last_id)
                break

            events = (data.get("data") or {}).get("orderFilledEvents") or []
            if not events:
                break

            total_events += len(events)
            batch_trades = self._parse_events(events)
            trades.extend(batch_trades)

            last_id = events[-1].get("id", "")

            if total_events % 10000 < PAGE_SIZE:
                elapsed = time.monotonic() - fetch_start
                last_ts = events[-1].get("timestamp", "")
                try:
                    cursor_date = datetime.utcfromtimestamp(int(last_ts)).date()
                except (ValueError, TypeError, OSError):
                    cursor_date = "?"
                _log(
                    f"  Goldsky: {total_events} events → {len(trades)} trades "
                    f"({elapsed:.0f}s, cursor={cursor_date})"
                )

            # Incremental CSV save
            if self._cache_path and total_events % save_interval < PAGE_SIZE:
                self._save_csv(trades)
                _log(f"  [checkpoint] saved {len(trades)} trades to CSV")

            if len(events) < PAGE_SIZE:
                break

        _log(f"  Goldsky: fetched {total_events} events → {len(trades)} valid trades")
        return trades

    @staticmethod
    def _build_query(timestamp_gte: int, last_id: str = "") -> str:
        """Build GraphQL query with timestamp filter and optional id cursor."""
        if last_id:
            where = f'{{ timestamp_gte: "{timestamp_gte}", id_gt: "{last_id}" }}'
        else:
            where = f'{{ timestamp_gte: "{timestamp_gte}" }}'
        return f"""{{
  orderFilledEvents(
    first: {PAGE_SIZE},
    orderBy: id,
    orderDirection: asc,
    where: {where}
  ) {{
    id
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
  }}
}}"""

    def _parse_events(self, events: list[dict]) -> list[Trade]:
        """Parse orderFilledEvents into Trade records.

        Each event produces up to 2 trades (maker + taker).
        USDC sides (asset ID "0") are skipped.
        Platform wallets are filtered out.
        """
        trades: list[Trade] = []

        for ev in events:
            event_id = ev.get("id", "")
            ts_raw = ev.get("timestamp")
            ts = None
            if ts_raw:
                try:
                    ts = datetime.utcfromtimestamp(int(ts_raw))
                except (ValueError, TypeError, OSError):
                    pass

            maker = (ev.get("maker") or "").lower()
            taker = (ev.get("taker") or "").lower()
            maker_asset = ev.get("makerAssetId", "")
            taker_asset = ev.get("takerAssetId", "")
            maker_amount = self._parse_amount(ev.get("makerAmountFilled", "0"))
            taker_amount = self._parse_amount(ev.get("takerAmountFilled", "0"))

            # Maker side: if maker asset is NOT USDC, maker sold tokens (SELL)
            # If maker asset IS USDC, maker bought tokens (BUY)
            if maker_asset != USDC_ASSET_ID and maker not in PLATFORM_WALLETS:
                # Maker gave tokens, received USDC → SELL
                token_id = maker_asset
                price = taker_amount / maker_amount if maker_amount > 0 else 0.0
                market_id = self._token_to_market.get(token_id, "")
                trades.append(Trade(
                    id=f"{event_id}-maker",
                    market=market_id,
                    asset_id=token_id,
                    side="SELL",
                    size=maker_amount,
                    price=price,
                    timestamp=ts,
                    owner=maker,
                ))

            if taker_asset != USDC_ASSET_ID and taker not in PLATFORM_WALLETS:
                # Taker gave USDC, received tokens → BUY
                token_id = taker_asset
                price = maker_amount / taker_amount if taker_amount > 0 else 0.0
                market_id = self._token_to_market.get(token_id, "")
                trades.append(Trade(
                    id=f"{event_id}-taker",
                    market=market_id,
                    asset_id=token_id,
                    side="BUY",
                    size=taker_amount,
                    price=price,
                    timestamp=ts,
                    owner=taker,
                ))

        return trades

    @staticmethod
    def _parse_amount(raw: str) -> float:
        """Parse on-chain amount string, dividing by 10^6 for USDC decimals."""
        try:
            return int(raw) / 1_000_000
        except (ValueError, TypeError):
            return 0.0

    def _load_csv_cache(self) -> tuple[list[Trade], int]:
        """Load previously cached trades from CSV. Returns (trades, max_timestamp)."""
        if not self._cache_path or not os.path.exists(self._cache_path):
            return [], 0

        trades: list[Trade] = []
        max_ts = 0

        try:
            with open(self._cache_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = None
                    if row.get("timestamp"):
                        try:
                            ts = datetime.fromisoformat(row["timestamp"])
                            row_ts = int(ts.timestamp())
                            if row_ts > max_ts:
                                max_ts = row_ts
                        except (ValueError, TypeError):
                            pass
                    trades.append(Trade(
                        id=row.get("id", ""),
                        market=row.get("market", ""),
                        asset_id=row.get("asset_id", ""),
                        side=row.get("side", ""),
                        size=float(row.get("size", 0)),
                        price=float(row.get("price", 0)),
                        timestamp=ts,
                        owner=row.get("owner", ""),
                    ))
        except Exception as e:
            logger.warning("goldsky_csv_load_error", error=str(e))
            return [], 0

        return trades, max_ts

    def _save_csv(self, trades: list[Trade]) -> None:
        """Save trades to CSV with atomic write."""
        tmp = self._cache_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for t in trades:
                writer.writerow({
                    "id": t.id,
                    "market": t.market,
                    "asset_id": t.asset_id,
                    "side": t.side,
                    "size": t.size,
                    "price": t.price,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else "",
                    "owner": t.owner,
                })
        os.replace(tmp, self._cache_path)
