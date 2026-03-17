"""Tests for the database layer."""

from __future__ import annotations

from datetime import datetime

import pytest

from sharp_discovery.db import Database
from sharp_discovery.models import WalletScore


class TestDatabaseWalletScores:
    @pytest.mark.asyncio
    async def test_upsert_and_retrieve(self, db):
        score = WalletScore(
            address="0xabc",
            win_rate=0.65,
            avg_roi=0.12,
            sharpe_ratio=0.80,
            sharpe_ci_lower=0.50,
            sharpe_ci_upper=1.10,
            hold_ratio=0.90,
            resolved_market_count=25,
            composite_score=0.72,
            extreme_price_ratio=0.15,
            scored_at=datetime(2024, 6, 1),
        )
        await db.upsert_wallet_score(score)
        top = await db.get_top_wallets(limit=1)
        assert len(top) == 1
        assert top[0].address == "0xabc"
        assert top[0].composite_score == pytest.approx(0.72)
        assert top[0].extreme_price_ratio == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_top_wallets_ordering(self, db):
        for i, (score_val, ext) in enumerate([(0.5, 0.0), (0.9, 0.3), (0.7, 0.1)]):
            await db.upsert_wallet_score(
                WalletScore(
                    address=f"0x{i}",
                    win_rate=0.5,
                    avg_roi=0.1,
                    sharpe_ratio=0.5,
                    sharpe_ci_lower=score_val,
                    sharpe_ci_upper=1.0,
                    hold_ratio=0.5,
                    resolved_market_count=20,
                    composite_score=score_val,
                    extreme_price_ratio=ext,
                    scored_at=datetime(2024, 6, 1),
                )
            )
        top = await db.get_top_wallets(limit=2)
        assert len(top) == 2
        assert top[0].composite_score == pytest.approx(0.9)
        assert top[1].composite_score == pytest.approx(0.7)
