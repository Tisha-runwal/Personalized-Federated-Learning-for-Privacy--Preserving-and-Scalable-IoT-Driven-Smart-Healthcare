"""FedProx strategy."""
from __future__ import annotations

from pfl_hcare.metrics.collector import MetricsCollector
from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy


class FedProxStrategy(FedAvgStrategy):
    """FedAvg strategy variant for FedProx (proximal term handled client-side)."""

    method_name: str = "fedprox"

    def __init__(self, metrics_collector: MetricsCollector, mu: float = 0.01, **kwargs):
        self.mu = mu
        super().__init__(metrics_collector=metrics_collector, **kwargs)
