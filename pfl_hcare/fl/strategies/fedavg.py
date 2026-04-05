"""FedAvg strategy with MetricsCollector integration."""
from __future__ import annotations

from flwr.server.strategy import FedAvg
from flwr.common import Parameters
from pfl_hcare.metrics.collector import MetricsCollector


class FedAvgStrategy(FedAvg):
    """Wraps flwr FedAvg and records round metrics via MetricsCollector."""

    method_name: str = "fedavg"

    def __init__(self, metrics_collector: MetricsCollector, **kwargs):
        self.metrics_collector = metrics_collector
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        # Collect per-round metrics
        grad_norms = [
            res.metrics.get("grad_norm", 0.0)
            for _, res in results
            if res.metrics
        ]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        self.metrics_collector.record_round(
            round_num=server_round,
            method=self.method_name,
            num_clients=len(results),
            avg_grad_norm=avg_grad_norm,
            **metrics,
        )
        return aggregated_params, metrics
