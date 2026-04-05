"""PFL-HCare strategy: full pipeline with DP, quantization, secure aggregation,
and adaptive client selection (Eq.9)."""
from __future__ import annotations

import random
import numpy as np
import torch
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from pfl_hcare.metrics.collector import MetricsCollector
from pfl_hcare.privacy.differential_privacy import DPMechanism
from pfl_hcare.privacy.quantization import GradientQuantizer
from pfl_hcare.privacy.secure_aggregation import SimulatedSecureAggregator


class PFLHCareStrategy(FedAvg):
    """PFL-HCare full strategy with DP, quantization, secure aggregation,
    and gradient-norm-based adaptive client selection (Eq.9)."""

    method_name: str = "pfl_hcare"

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        noise_multiplier: float = 0.5,
        max_grad_norm: float = 1.0,
        k_bits: int = 8,
        adaptive_selection: bool = True,
        latency_range_ms: tuple[int, int] = (50, 200),
        seed: int = 42,
        **kwargs,
    ):
        self.metrics_collector = metrics_collector
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.k_bits = k_bits
        self.adaptive_selection = adaptive_selection
        self._client_grad_norms: dict[str, float] = {}
        self._rng = random.Random(seed)

        self.dp_mechanism = DPMechanism(
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        self.quantizer = GradientQuantizer(k_bits=k_bits)
        self.secure_agg = SimulatedSecureAggregator(
            latency_range_ms=latency_range_ms,
            seed=seed,
        )
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Adaptive client selection (Eq.9): p_i = ||grad_i|| / sum ||grad_j||
    # ------------------------------------------------------------------

    def _selection_probabilities(self, client_ids: list[str]) -> dict[str, float]:
        norms = {cid: self._client_grad_norms.get(cid, 1.0) for cid in client_ids}
        total = sum(norms.values()) or 1.0
        return {cid: v / total for cid, v in norms.items()}

    # ------------------------------------------------------------------
    # Override configure_fit to inject selection mask into config
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        config_list = super().configure_fit(server_round, parameters, client_manager)
        if not self.adaptive_selection or not self._client_grad_norms:
            return config_list
        # Build selection mask based on Eq.9 probabilities
        client_ids = [str(proxy.cid) for proxy, _ in config_list]
        probs = self._selection_probabilities(client_ids)
        selected = {
            cid for cid, p in probs.items()
            if self._rng.random() < p
        }
        # Ensure at least one client selected
        if not selected and client_ids:
            selected = {self._rng.choice(client_ids)}
        # Filter config_list to selected clients
        filtered = [
            (proxy, fit_ins)
            for proxy, fit_ins in config_list
            if str(proxy.cid) in selected
        ]
        return filtered if filtered else config_list

    # ------------------------------------------------------------------
    # Aggregate fit with quantization + secure aggregation
    # ------------------------------------------------------------------

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Update gradient norm tracking
        for _, fit_res in results:
            if fit_res.metrics:
                cid = str(fit_res.metrics.get("client_id", "unknown"))
                self._client_grad_norms[cid] = float(
                    fit_res.metrics.get("grad_norm", 1.0)
                )

        # Dequantize and decode parameters from clients
        weights_results = []
        for proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((params, fit_res.num_examples))

        # Weighted average
        total_samples = sum(n for _, n in weights_results)
        weights = [n / total_samples for _, n in weights_results]

        # Quantize for bandwidth metrics (simulate what clients send)
        all_tensors = [
            [torch.tensor(arr) for arr in params]
            for params, _ in weights_results
        ]
        _, meta = self.quantizer.quantize(all_tensors[0])
        bw_report = self.quantizer.get_bandwidth_report()

        # Secure aggregation
        aggregated_tensors = self.secure_agg.aggregate(all_tensors, weights)
        agg_report = self.secure_agg.get_report()

        # DP accounting
        dp_report = self.dp_mechanism.get_privacy_report()

        # Convert back to numpy
        aggregated_ndarrays = [t.detach().cpu().numpy() for t in aggregated_tensors]
        aggregated_params = ndarrays_to_parameters(aggregated_ndarrays)

        # Compute selection mask
        client_ids = [str(fit_res.metrics.get("client_id", i)) for i, (_, fit_res) in enumerate(results)]
        selection_probs = self._selection_probabilities(client_ids)
        selection_mask = {cid: float(p) for cid, p in selection_probs.items()}

        grad_norms = [float(fit_res.metrics.get("grad_norm", 0.0)) for _, fit_res in results if fit_res.metrics]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        metrics = {
            "num_clients": len(results),
            "avg_grad_norm": avg_grad_norm,
            "epsilon_spent": dp_report["epsilon_spent"],
            "bytes_original": bw_report["original_bytes"],
            "bytes_quantized": bw_report["quantized_bytes"],
            "compression_ratio": bw_report["compression_ratio"],
            "encryption_latency_ms": agg_report["latency_ms"],
        }

        self.metrics_collector.record_round(
            round_num=server_round,
            method=self.method_name,
            **metrics,
            selection_mask=selection_mask,
        )

        return aggregated_params, metrics
