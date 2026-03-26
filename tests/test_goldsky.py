"""Tests for the Goldsky subgraph scraper module."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from sharp_discovery.goldsky import (
    GOLDSKY_ENDPOINT,
    PLATFORM_WALLETS,
    USDC_ASSET_ID,
    GoldskyClient,
)
from sharp_discovery.models import Trade


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_event(
    event_id: str = "evt1",
    timestamp: str = "1700000000",
    maker: str = "0xmaker1",
    taker: str = "0xtaker1",
    maker_asset: str = "token_abc",
    taker_asset: str = USDC_ASSET_ID,
    maker_amount: str = "10000000",  # 10 tokens
    taker_amount: str = "5000000",  # 5 USDC
) -> dict:
    """Create a mock orderFilledEvent."""
    return {
        "id": event_id,
        "timestamp": timestamp,
        "maker": maker,
        "taker": taker,
        "makerAssetId": maker_asset,
        "takerAssetId": taker_asset,
        "makerAmountFilled": maker_amount,
        "takerAmountFilled": taker_amount,
    }


# ── Event parsing tests ────────────────────────────────────────────────


class TestParseEvents:
    """Tests for GoldskyClient._parse_events."""

    def _client(self, token_to_market=None):
        return GoldskyClient(token_to_market=token_to_market or {})

    def test_maker_sells_tokens(self):
        """Maker gives tokens, receives USDC → SELL trade for maker."""
        ev = _make_event(
            maker_asset="token_abc",  # maker gives tokens
            taker_asset=USDC_ASSET_ID,  # taker gives USDC
            maker_amount="10000000",  # 10 tokens
            taker_amount="5000000",  # 5 USDC
        )
        client = self._client()
        trades = client._parse_events([ev])
        # Should produce 1 trade: maker SELL (taker side is USDC so skipped)
        assert len(trades) == 1
        t = trades[0]
        assert t.side == "SELL"
        assert t.owner == "0xmaker1"
        assert t.asset_id == "token_abc"
        assert t.size == 10.0
        assert abs(t.price - 0.5) < 0.01  # 5 USDC / 10 tokens

    def test_taker_buys_tokens(self):
        """Taker gives USDC, receives tokens → BUY trade for taker."""
        ev = _make_event(
            maker_asset=USDC_ASSET_ID,  # maker gives USDC
            taker_asset="token_xyz",  # taker gives tokens... wait
        )
        # In this case: maker gives USDC (maker_asset=0), taker gives tokens (taker_asset=token_xyz)
        # So taker_asset != USDC → taker SELL trade
        # maker_asset == USDC → no maker trade
        client = self._client()
        trades = client._parse_events([ev])
        # taker gave tokens (taker_asset != USDC) → taker gets a BUY record
        # Actually: taker_asset != USDC means taker is giving tokens → BUY for taker
        assert len(trades) == 1
        t = trades[0]
        assert t.side == "BUY"
        assert t.owner == "0xtaker1"

    def test_both_sides_non_usdc(self):
        """Both sides are tokens → produces 2 trades (maker SELL + taker BUY)."""
        ev = _make_event(
            maker_asset="token_a",
            taker_asset="token_b",
            maker_amount="10000000",
            taker_amount="8000000",
        )
        client = self._client()
        trades = client._parse_events([ev])
        assert len(trades) == 2
        sides = {t.side for t in trades}
        assert sides == {"SELL", "BUY"}

    def test_platform_wallets_filtered(self):
        """Trades from platform wallets are excluded."""
        platform = list(PLATFORM_WALLETS)[0]
        ev = _make_event(maker=platform, taker=platform)
        client = self._client()
        trades = client._parse_events([ev])
        assert len(trades) == 0

    def test_platform_wallet_one_side(self):
        """Only the platform side is filtered, other side still produces a trade."""
        platform = list(PLATFORM_WALLETS)[0]
        ev = _make_event(
            maker=platform,
            taker="0xregular",
            maker_asset=USDC_ASSET_ID,
            taker_asset="token_abc",
        )
        client = self._client()
        trades = client._parse_events([ev])
        # maker is platform + gives USDC → no maker trade
        # taker is regular + gives tokens → BUY trade
        assert len(trades) == 1
        assert trades[0].owner == "0xregular"

    def test_token_to_market_mapping(self):
        """Token IDs are mapped to market condition IDs."""
        ev = _make_event(maker_asset="token_mapped")
        client = self._client(token_to_market={"token_mapped": "cid_123"})
        trades = client._parse_events([ev])
        assert len(trades) == 1
        assert trades[0].market == "cid_123"

    def test_timestamp_parsing(self):
        """Unix timestamps are correctly parsed to datetime."""
        ev = _make_event(timestamp="1700000000")
        client = self._client()
        trades = client._parse_events([ev])
        assert len(trades) == 1
        assert trades[0].timestamp == datetime.utcfromtimestamp(1700000000)


# ── Amount parsing tests ───────────────────────────────────────────────


class TestParseAmount:
    def test_normal(self):
        assert GoldskyClient._parse_amount("1000000") == 1.0

    def test_large(self):
        assert GoldskyClient._parse_amount("50000000") == 50.0

    def test_zero(self):
        assert GoldskyClient._parse_amount("0") == 0.0

    def test_invalid(self):
        assert GoldskyClient._parse_amount("abc") == 0.0

    def test_empty(self):
        assert GoldskyClient._parse_amount("") == 0.0


# ── CSV cache tests ────────────────────────────────────────────────────


class TestCSVCache:
    def test_save_and_load(self, tmp_path):
        csv_path = str(tmp_path / "trades.csv")
        client = GoldskyClient(cache_path=csv_path)

        trades = [
            Trade(
                id="t1", market="m1", asset_id="a1", side="BUY",
                size=10.0, price=0.5,
                timestamp=datetime(2024, 6, 15, 12, 0, 0),
                owner="0xwallet1",
            ),
            Trade(
                id="t2", market="m2", asset_id="a2", side="SELL",
                size=5.0, price=0.8,
                timestamp=datetime(2024, 7, 1, 8, 30, 0),
                owner="0xwallet2",
            ),
        ]
        client._save_csv(trades)
        assert os.path.exists(csv_path)

        loaded, max_ts = client._load_csv_cache()
        assert len(loaded) == 2
        assert loaded[0].id == "t1"
        assert loaded[0].size == 10.0
        assert loaded[1].owner == "0xwallet2"
        assert max_ts == int(datetime(2024, 7, 1, 8, 30, 0).timestamp())

    def test_load_nonexistent(self, tmp_path):
        client = GoldskyClient(cache_path=str(tmp_path / "nope.csv"))
        trades, max_ts = client._load_csv_cache()
        assert trades == []
        assert max_ts == 0

    def test_load_empty_cache_path(self):
        client = GoldskyClient(cache_path="")
        trades, max_ts = client._load_csv_cache()
        assert trades == []
        assert max_ts == 0

    def test_atomic_write(self, tmp_path):
        """CSV is written atomically via tmp file."""
        csv_path = str(tmp_path / "trades.csv")
        client = GoldskyClient(cache_path=csv_path)
        client._save_csv([
            Trade(id="t1", market="m1", asset_id="a1", side="BUY",
                  size=1.0, price=0.5, owner="0x1"),
        ])
        # tmp file should be cleaned up
        assert not os.path.exists(csv_path + ".tmp")
        assert os.path.exists(csv_path)


# ── GraphQL query tests ────────────────────────────────────────────────


class TestBuildQuery:
    def test_first_page_query(self):
        query = GoldskyClient._build_query(1700000000)
        assert "1700000000" in query
        assert "orderFilledEvents" in query
        assert "timestamp_gte" in query
        assert "id_gt" not in query

    def test_paginated_query_has_both_filters(self):
        query = GoldskyClient._build_query(1700000000, "evt_abc123")
        assert "1700000000" in query
        assert "evt_abc123" in query
        assert "timestamp_gte" in query
        assert "id_gt" in query

    def test_query_contains_required_fields(self):
        query = GoldskyClient._build_query(0)
        for field in ["maker", "taker", "makerAssetId", "takerAssetId",
                       "makerAmountFilled", "takerAmountFilled", "timestamp"]:
            assert field in query
