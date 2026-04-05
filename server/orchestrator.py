"""Orchestrator: run all 5 FL methods sequentially and collect results."""
from __future__ import annotations

import logging
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)

_METHODS = ["fedavg", "fedprox", "per_fedavg", "pfedme", "pfl_hcare"]


def run_comparison(
    config_path: str = "configs/default.yaml",
    on_round: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run all 5 FL methods sequentially and return a dict of per-method results.

    Args:
        config_path: Path to a YAML config file.  Values in the file are used
                     as defaults; they can be overridden by the caller before
                     passing *config_path*.
        on_round:    Optional callback invoked after every training round with
                     the round-update dict produced by MetricsCollector.

    Returns:
        A dict mapping method name → final metrics dict.
    """
    from pfl_hcare.fl.server import run_simulation
    from pfl_hcare.metrics.collector import MetricsCollector

    try:
        with open(config_path) as f:
            base_config: dict = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config file %r not found; using empty defaults.", config_path)
        base_config = {}

    results: dict[str, Any] = {}

    for method in _METHODS:
        logger.info("=== Starting method: %s ===", method)

        collector = MetricsCollector()
        if on_round is not None:
            collector.on_round(on_round)

        try:
            final = run_simulation(
                config=base_config,
                method=method,
                metrics_collector=collector,
            )
        except Exception as exc:
            logger.exception("Method %r failed: %s", method, exc)
            final = {"error": str(exc)}

        results[method] = final
        logger.info("=== Finished method: %s — %s ===", method, final)

    return results
