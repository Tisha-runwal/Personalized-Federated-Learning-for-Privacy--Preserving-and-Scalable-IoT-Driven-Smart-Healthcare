"""SQLite persistence layer using aiosqlite for PFL-HCare runs and metrics."""
from __future__ import annotations

import json
import time
from typing import Any

import aiosqlite

DB_PATH = "pfl_hcare_runs.db"


async def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they do not already exist."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                method      TEXT    NOT NULL,
                dataset     TEXT    NOT NULL,
                config      TEXT    NOT NULL,
                created_at  REAL    NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS round_metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL REFERENCES runs(id),
                round_num   INTEGER NOT NULL,
                metrics_json TEXT   NOT NULL
            )
            """
        )
        await db.commit()


async def create_run(
    method: str,
    dataset: str,
    config: dict[str, Any],
    db_path: str = DB_PATH,
) -> int:
    """Insert a new run record and return its auto-generated id."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO runs (method, dataset, config, created_at) VALUES (?, ?, ?, ?)",
            (method, dataset, json.dumps(config), time.time()),
        )
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


async def save_round(
    run_id: int,
    round_num: int,
    metrics: dict[str, Any],
    db_path: str = DB_PATH,
) -> None:
    """Persist metrics for a single training round."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO round_metrics (run_id, round_num, metrics_json) VALUES (?, ?, ?)",
            (run_id, round_num, json.dumps(metrics)),
        )
        await db.commit()


async def get_run_metrics(
    run_id: int,
    db_path: str = DB_PATH,
) -> list[dict[str, Any]]:
    """Return a list of {round, metrics} dicts for the given run_id."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT round_num, metrics_json FROM round_metrics WHERE run_id = ? ORDER BY round_num",
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [
        {"round": row["round_num"], "metrics": json.loads(row["metrics_json"])}
        for row in rows
    ]


async def list_runs(db_path: str = DB_PATH) -> list[dict[str, Any]]:
    """Return summary information for all stored runs."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, method, dataset, config, created_at FROM runs ORDER BY id DESC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [
        {
            "id": row["id"],
            "method": row["method"],
            "dataset": row["dataset"],
            "config": json.loads(row["config"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
