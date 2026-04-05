"""Metrics retrieval routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from server.db import get_run_metrics, list_runs

router = APIRouter()


@router.get("/runs")
async def get_runs() -> list[dict[str, Any]]:
    """Return a summary list of all stored training runs."""
    return await list_runs()


@router.get("/{run_id}")
async def get_run(run_id: int) -> dict[str, Any]:
    """Return all round metrics for the given run_id."""
    metrics = await get_run_metrics(run_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No metrics found for run_id={run_id}.")
    return {"run_id": run_id, "rounds": metrics}
