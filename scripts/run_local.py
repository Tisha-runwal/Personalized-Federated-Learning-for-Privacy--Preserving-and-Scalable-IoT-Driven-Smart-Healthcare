"""CLI runner for local FL simulation.

Usage:
    python scripts/run_local.py --config configs/default.yaml --method pfl_hcare --rounds 50 --clients 5
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import os

# Ensure project root is on the path when run as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml

from pfl_hcare.metrics.collector import MetricsCollector
from pfl_hcare.fl.server import run_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_local")


def _load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _print_callback(entry: dict) -> None:
    rnd = entry.get("round", "?")
    method = entry.get("method", "?")
    metrics = entry.get("metrics", {})
    parts = [f"round={rnd}", f"method={method}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif not isinstance(v, dict):
            parts.append(f"{k}={v}")
    print("  [metrics] " + "  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local FL simulation")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--method",
        choices=["fedavg", "fedprox", "per_fedavg", "pfedme", "pfl_hcare"],
        default="pfl_hcare",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Override num_rounds from config")
    parser.add_argument("--clients", type=int, default=None, help="Override num_clients from config")
    parser.add_argument("--dataset", default=None, help="Override dataset name (har / mimic)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    config = _load_config(args.config)

    # Apply CLI overrides
    training = config.setdefault("training", {})
    if args.rounds is not None:
        training["num_rounds"] = args.rounds
    if args.clients is not None:
        training["num_clients"] = args.clients
    if args.lr is not None:
        training["learning_rate"] = args.lr
    if args.seed is not None:
        training["seed"] = args.seed
    if args.dataset is not None:
        config.setdefault("dataset", {})["name"] = args.dataset

    logger.info("Starting simulation: method=%s  rounds=%d  clients=%d",
                args.method,
                training.get("num_rounds", 50),
                training.get("num_clients", 10))

    mc = MetricsCollector()
    mc.on_round(_print_callback)

    results = run_simulation(config=config, method=args.method, metrics_collector=mc)

    print("\n=== Simulation complete ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
