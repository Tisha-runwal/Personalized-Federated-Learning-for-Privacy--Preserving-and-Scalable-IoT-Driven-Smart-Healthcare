"""pFedMe strategy (Moreau envelope personalization)."""
from __future__ import annotations

from pfl_hcare.metrics.collector import MetricsCollector
from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy


class PFedMeStrategy(FedAvgStrategy):
    """pFedMe: server aggregates global model; personalization via Moreau envelope client-side."""

    method_name: str = "pfedme"

    def __init__(self, metrics_collector: MetricsCollector, lambd: float = 15.0, **kwargs):
        self.lambd = lambd
        super().__init__(metrics_collector=metrics_collector, **kwargs)
