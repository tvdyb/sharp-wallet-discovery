"""Shared fixtures for tests."""

from __future__ import annotations

import pytest
import pytest_asyncio

from sharp_discovery.db import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with Database(db_path) as database:
        yield database
