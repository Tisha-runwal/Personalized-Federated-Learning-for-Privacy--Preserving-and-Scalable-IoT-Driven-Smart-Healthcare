"""Training control routes: start, stop, status."""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import yaml
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from server.db import create_run, save_round
from server.ws.live import broadcast_metric

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Global training state
# ---------------------------------------------------------------------------

_training_state: dict[str, Any] = {
    "status": "idle",      # idle | running | completed | error
    "method": None,
    "dataset": None,
    "round": 0,
    "total_rounds": 0,
    "run_id": None,
    "error": None,
}
_stop_event = threading.Event()
_state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    method: str = Field("pfl_hcare", description="FL method name")
    dataset: str = Field("har", description="Dataset: har or mimic")
    num_clients: int = Field(10, ge=2)
    num_rounds: int = Field(50, ge=1)
    noise_multiplier: float = Field(0.5, ge=0.0)
    k_bits: int = Field(8, ge=1, le=32)
    partition_alpha: float = Field(0.5, gt=0.0)
    learning_rate: float = Field(0.01, gt=0.0)


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------

def _training_worker(request: TrainingConfig, config_path: str = "configs/default.yaml") -> None:
    """Run the FL simulation in a background thread."""
    import asyncio as _asyncio

    # Load base config
    try:
        with open(config_path) as f:
            config: dict = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    # Override with request params
    config.setdefault("training", {})
    config["training"]["num_clients"] = request.num_clients
    config["training"]["num_rounds"] = request.num_rounds
    config["training"]["learning_rate"] = request.learning_rate

    config.setdefault("dataset", {})
    config["dataset"]["name"] = request.dataset
    config["dataset"]["partition_alpha"] = request.partition_alpha

    config.setdefault("privacy", {})
    config["privacy"]["noise_multiplier"] = request.noise_multiplier

    config.setdefault("quantization", {})
    config["quantization"]["k_bits"] = request.k_bits

    # We need an event-loop to run async DB / WebSocket calls from the thread.
    loop = _asyncio.new_event_loop()

    async def _create_run_async():
        return await create_run(request.method, request.dataset, config)

    run_id: int = loop.run_until_complete(_create_run_async())

    with _state_lock:
        _training_state["run_id"] = run_id
        _training_state["total_rounds"] = request.num_rounds

    # Build MetricsCollector with callback
    from pfl_hcare.metrics.collector import MetricsCollector

    collector = MetricsCollector()

    def _on_round(entry: dict) -> None:
        if _stop_event.is_set():
            return
        rnd = entry.get("round", 0)
        metrics = entry.get("metrics", {})

        with _state_lock:
            _training_state["round"] = rnd

        # Persist + broadcast (fire-and-forget within thread loop)
        async def _persist_and_broadcast():
            await save_round(run_id, rnd, {**metrics, "method": entry.get("method")})
            await broadcast_metric(
                {
                    "type": "round_update",
                    "run_id": run_id,
                    "round": rnd,
                    "total_rounds": request.num_rounds,
                    "method": request.method,
                    "metrics": metrics,
                }
            )

        loop.run_until_complete(_persist_and_broadcast())

    collector.on_round(_on_round)

    # Run the simulation
    try:
        from pfl_hcare.fl.server import run_simulation

        final = run_simulation(config=config, method=request.method, metrics_collector=collector)
        with _state_lock:
            _training_state["status"] = "completed"

        loop.run_until_complete(
            broadcast_metric({"type": "completed", "run_id": run_id, "final": final})
        )
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        with _state_lock:
            _training_state["status"] = "error"
            _training_state["error"] = str(exc)
        loop.run_until_complete(
            broadcast_metric({"type": "error", "run_id": run_id, "error": str(exc)})
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start", status_code=202)
async def start_training(request: TrainingConfig) -> dict:
    """Start a new FL training run in the background."""
    with _state_lock:
        if _training_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Training already running.")
        _stop_event.clear()
        _training_state.update(
            {
                "status": "running",
                "method": request.method,
                "dataset": request.dataset,
                "round": 0,
                "total_rounds": request.num_rounds,
                "run_id": None,
                "error": None,
            }
        )

    thread = threading.Thread(
        target=_training_worker,
        args=(request,),
        daemon=True,
        name="fl-training",
    )
    thread.start()
    return {"detail": "Training started.", "method": request.method, "dataset": request.dataset}


@router.post("/stop", status_code=200)
async def stop_training() -> dict:
    """Signal the running training job to stop after the current round."""
    with _state_lock:
        if _training_state["status"] != "running":
            raise HTTPException(status_code=409, detail="No training is running.")
        _stop_event.set()
        _training_state["status"] = "idle"
    return {"detail": "Stop signal sent."}


@router.get("/status")
async def get_status() -> dict:
    """Return the current training status snapshot."""
    with _state_lock:
        return {
            "status": _training_state["status"],
            "method": _training_state["method"],
            "dataset": _training_state["dataset"],
            "round": _training_state["round"],
            "total_rounds": _training_state["total_rounds"],
            "run_id": _training_state["run_id"],
            "error": _training_state["error"],
        }
