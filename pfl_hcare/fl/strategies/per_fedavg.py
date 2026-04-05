"""Per-FedAvg strategy (MAML-based personalization)."""
from __future__ import annotations

from pfl_hcare.metrics.collector import MetricsCollector
from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy


class PerFedAvgStrategy(FedAvgStrategy):
    """Per-FedAvg: FedAvg with MAML-based personalization (handled client-side)."""

    method_name: str = "per_fedavg"

    def __init__(self, metrics_collector: MetricsCollector, **kwargs):
        super().__init__(metrics_collector=metrics_collector, **kwargs)
