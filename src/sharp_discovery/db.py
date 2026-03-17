"""SQLite persistence for wallet scores."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import aiosqlite
import structlog

from sharp_discovery.models import WalletScore

logger = structlog.get_logger()

SCHEMA = """
CREATE TABLE IF NOT EXISTS wallet_scores (
    address TEXT NOT NULL,
    win_rate REAL NOT NULL,
    avg_roi REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    sharpe_ci_lower REAL NOT NULL,
    sharpe_ci_upper REAL NOT NULL,
    hold_ratio REAL NOT NULL,
    resolved_market_count INTEGER NOT NULL,
    composite_score REAL NOT NULL,
    extreme_price_ratio REAL NOT NULL DEFAULT 0.0,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (address, scored_at)
);

CREATE INDEX IF NOT EXISTS idx_ws_composite ON wallet_scores(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_ws_address ON wallet_scores(address);
"""


class Database:
    """Async SQLite database for wallet scores."""

    def __init__(self, db_path: str = "sharp_discovery.db") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        return self._conn

    async def upsert_wallet_score(self, score: WalletScore) -> None:
        conn = self._ensure_conn()
        await conn.execute(
            """INSERT OR REPLACE INTO wallet_scores
               (address, win_rate, avg_roi, sharpe_ratio, sharpe_ci_lower,
                sharpe_ci_upper, hold_ratio, resolved_market_count,
                composite_score, extreme_price_ratio, scored_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                score.address,
                score.win_rate,
                score.avg_roi,
                score.sharpe_ratio,
                score.sharpe_ci_lower,
                score.sharpe_ci_upper,
                score.hold_ratio,
                score.resolved_market_count,
                score.composite_score,
                score.extreme_price_ratio,
                score.scored_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_top_wallets(self, limit: int = 50) -> list[WalletScore]:
        conn = self._ensure_conn()
        cursor = await conn.execute(
            """SELECT DISTINCT address, win_rate, avg_roi, sharpe_ratio,
                      sharpe_ci_lower, sharpe_ci_upper, hold_ratio,
                      resolved_market_count, composite_score,
                      extreme_price_ratio, scored_at
               FROM wallet_scores
               WHERE scored_at = (
                   SELECT MAX(scored_at) FROM wallet_scores ws2
                   WHERE ws2.address = wallet_scores.address
               )
               ORDER BY composite_score DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            WalletScore(
                address=r["address"],
                win_rate=r["win_rate"],
                avg_roi=r["avg_roi"],
                sharpe_ratio=r["sharpe_ratio"],
                sharpe_ci_lower=r["sharpe_ci_lower"],
                sharpe_ci_upper=r["sharpe_ci_upper"],
                hold_ratio=r["hold_ratio"],
                resolved_market_count=r["resolved_market_count"],
                composite_score=r["composite_score"],
                extreme_price_ratio=r["extreme_price_ratio"],
                scored_at=datetime.fromisoformat(r["scored_at"]),
            )
            for r in rows
        ]
