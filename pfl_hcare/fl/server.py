"""Flower server and local simulation runner for PFL-HCare."""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset

from pfl_hcare.metrics.collector import MetricsCollector
from pfl_hcare.models.health_classifier import HealthClassifier
from pfl_hcare.models.har_classifier import HARClassifier
from pfl_hcare.maml.maml import MAMLWrapper
from pfl_hcare.privacy.differential_privacy import DPMechanism
from pfl_hcare.privacy.quantization import GradientQuantizer
from pfl_hcare.privacy.secure_aggregation import SimulatedSecureAggregator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_model(dataset_name: str) -> nn.Module:
    """Return the appropriate model for a given dataset."""
    if dataset_name in ("har", "ucihar"):
        return HARClassifier(accept_flat=True)
    return HealthClassifier()


def load_dataset(dataset_name: str):
    """Load and return (train_dataset, test_dataset)."""
    if dataset_name in ("har", "ucihar"):
        from data.har_loader import HARDataset
        train_ds = HARDataset(split="train", download=False)
        test_ds = HARDataset(split="test", download=False)
    else:
        from data.mimic_loader import MedicalDataset
        train_ds = MedicalDataset(split="train")
        test_ds = MedicalDataset(split="test")
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Local simulation (no Ray required)
# ---------------------------------------------------------------------------

def run_simulation(
    config: dict,
    method: str,
    metrics_collector: MetricsCollector,
) -> dict:
    """Run FL simulation locally with all paper components wired in.

    Pipeline per round:
      1. Adaptive client selection (Eq.9) — for pfl_hcare
      2. Local training (MAML Eq.3-4 for per_fedavg/pfl_hcare, standard for others)
      3. DP noise injection (Eq.5) — for pfl_hcare (done client-side)
      4. Gradient quantization (Eq.8) — for pfl_hcare
      5. Simulated secure aggregation (Eq.6-7) — for pfl_hcare
      6. Weighted FedAvg aggregation (Eq.1)
      7. Per-round evaluation + metrics recording
    """
    from data.partition import DirichletPartitioner
    from pfl_hcare.fl.client import PFLClient

    training_cfg = config.get("training", {})
    num_clients: int = training_cfg.get("num_clients", 10)
    num_rounds: int = training_cfg.get("num_rounds", 50)
    local_epochs: int = training_cfg.get("local_epochs", 5)
    batch_size: int = training_cfg.get("batch_size", 32)
    lr: float = training_cfg.get("learning_rate", 0.01)
    seed: int = training_cfg.get("seed", 42)
    dataset_name: str = config.get("dataset", {}).get("name", "har")
    partition_alpha: float = config.get("dataset", {}).get("partition_alpha", 0.5)
    privacy_cfg = config.get("privacy", {})
    maml_cfg = config.get("maml", {})
    quant_cfg = config.get("quantization", {})
    sec_agg_cfg = config.get("secure_aggregation", {})
    sel_cfg = config.get("client_selection", {})

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Setup components ---
    logger.info("Loading dataset: %s", dataset_name)
    train_ds, test_ds = load_dataset(dataset_name)

    logger.info("Partitioning %d samples across %d clients (alpha=%.2f)",
                len(train_ds), num_clients, partition_alpha)
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=partition_alpha, seed=seed)
    partitions = partitioner.partition(train_ds)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quantizer (Eq.8) — only for pfl_hcare
    quantizer = None
    if method == "pfl_hcare" and quant_cfg.get("enabled", True):
        quantizer = GradientQuantizer(k_bits=quant_cfg.get("k_bits", 8))

    # Simulated secure aggregation (Eq.6-7) — only for pfl_hcare
    secure_agg = None
    if method == "pfl_hcare" and sec_agg_cfg.get("simulated", True):
        lat_range = tuple(sec_agg_cfg.get("latency_range_ms", [0, 0]))
        secure_agg = SimulatedSecureAggregator(latency_range_ms=lat_range, seed=seed)

    # Server-side DP accountant for epsilon tracking
    dp_accountant = None
    if method == "pfl_hcare":
        dp_accountant = DPMechanism(
            noise_multiplier=privacy_cfg.get("noise_multiplier", 0.5),
            max_grad_norm=privacy_cfg.get("max_grad_norm", 1.0),
            delta=privacy_cfg.get("delta", 1e-5),
        )

    # Adaptive selection config (Eq.9)
    adaptive_selection = method == "pfl_hcare" and sel_cfg.get("adaptive", True)
    min_participation_interval = sel_cfg.get("min_participation_interval", 10)

    # --- Build clients ---
    clients: list[PFLClient] = []
    for cid in range(num_clients):
        model = create_model(dataset_name)

        maml_wrapper = None
        dp_mech = None

        if method in ("per_fedavg", "pfl_hcare"):
            maml_wrapper = MAMLWrapper(
                model=model,
                inner_lr=maml_cfg.get("inner_lr", 0.01),
                inner_steps=maml_cfg.get("inner_steps", 5),
                second_order=maml_cfg.get("second_order", False),
            )

        if method == "pfl_hcare":
            dp_mech = DPMechanism(
                noise_multiplier=privacy_cfg.get("noise_multiplier", 0.5),
                max_grad_norm=privacy_cfg.get("max_grad_norm", 1.0),
                delta=privacy_cfg.get("delta", 1e-5),
            )

        indices = partitions[cid]
        client_train = Subset(train_ds, indices)

        client = PFLClient(
            client_id=cid,
            model=model,
            train_dataset=client_train,
            test_dataset=test_ds,
            strategy=method,
            local_epochs=local_epochs,
            learning_rate=lr,
            batch_size=batch_size,
            device=device,
            mu=privacy_cfg.get("mu", 0.01),
            lambd=config.get("pfedme", {}).get("lambd", 15.0),
            maml_wrapper=maml_wrapper,
            dp_mechanism=dp_mech,
        )
        clients.append(client)

    # --- Global parameters: start from client 0's initial params ---
    global_params = clients[0].get_parameters(config={})

    # --- Tracking state ---
    client_gradient_norms: dict[int, float] = {}
    client_last_participation: dict[int, int] = {cid: 0 for cid in range(num_clients)}
    final_metrics: dict = {}

    # ===================================================================
    # FL TRAINING LOOP
    # ===================================================================
    for rnd in range(1, num_rounds + 1):
        logger.info("Round %d / %d", rnd, num_rounds)

        # ---------------------------------------------------------------
        # Step 1: Client Selection (Eq.9 for pfl_hcare, all clients otherwise)
        # ---------------------------------------------------------------
        if adaptive_selection and rnd > 1 and client_gradient_norms:
            selected_ids = _adaptive_select(
                num_clients=num_clients,
                n_select=max(2, num_clients // 2),
                gradient_norms=client_gradient_norms,
                last_participation=client_last_participation,
                current_round=rnd,
                min_interval=min_participation_interval,
                seed=seed + rnd,
            )
        else:
            selected_ids = list(range(num_clients))

        for cid in selected_ids:
            client_last_participation[cid] = rnd

        # ---------------------------------------------------------------
        # Step 2: Local Fit (MAML Eq.3-4, DP Eq.5 applied client-side)
        # ---------------------------------------------------------------
        fit_results = []
        for cid in selected_ids:
            client = clients[cid]
            updated, n_samples, fit_metrics = client.fit(
                parameters=[arr.copy() for arr in global_params],
                config={"round": rnd},
            )
            fit_results.append((cid, updated, n_samples, fit_metrics))

        # Track gradient norms for adaptive selection
        for cid, _, _, fmetrics in fit_results:
            gnorm = fmetrics.get("grad_norm", 0.0)
            if not (np.isnan(gnorm) or np.isinf(gnorm)):
                client_gradient_norms[cid] = gnorm

        # ---------------------------------------------------------------
        # Step 3: Filter NaN clients
        # ---------------------------------------------------------------
        valid_results = []
        for cid, params, n_samples, fmetrics in fit_results:
            has_nan = any(np.isnan(p).any() for p in params)
            if not has_nan:
                valid_results.append((cid, params, n_samples, fmetrics))

        if not valid_results:
            logger.warning("  All clients produced NaN — skipping aggregation")
            # Still record metrics with previous accuracy
            _record_round_metrics(
                metrics_collector, rnd, method, num_clients, selected_ids,
                fit_results, final_metrics.get("final_accuracy", 0.0),
                final_metrics.get("final_loss", 0.0), [],
                quantizer, secure_agg, dp_accountant,
            )
            continue

        all_client_params = [r[1] for r in valid_results]
        all_weights = [float(r[2]) for r in valid_results]

        # ---------------------------------------------------------------
        # Step 4: Gradient Quantization (Eq.8) — compress before "transmission"
        # ---------------------------------------------------------------
        bandwidth_report = {"original_bytes": 0, "quantized_bytes": 0,
                           "compression_ratio": 1.0, "savings_percent": 0.0}

        if quantizer is not None:
            quantized_client_params = []
            for client_p in all_client_params:
                tensors = [torch.tensor(p) for p in client_p]
                q, meta = quantizer.quantize(tensors)
                deq = quantizer.dequantize(q, meta)
                quantized_client_params.append([d.numpy() for d in deq])
            all_client_params = quantized_client_params
            bandwidth_report = quantizer.get_bandwidth_report()

        # ---------------------------------------------------------------
        # Step 5: Simulated Secure Aggregation (Eq.6-7)
        # ---------------------------------------------------------------
        encryption_report = {"status": "disabled", "latency_ms": 0}

        if secure_agg is not None:
            tensor_params = [
                [torch.tensor(p) for p in cp] for cp in all_client_params
            ]
            total_w = sum(all_weights)
            norm_w = [w / total_w for w in all_weights]
            aggregated_tensors = secure_agg.aggregate(tensor_params, norm_w)
            global_params = [t.numpy() for t in aggregated_tensors]
            encryption_report = secure_agg.get_report()
        else:
            # ---------------------------------------------------------------
            # Step 5b: Plain FedAvg aggregation (Eq.1) for non-pfl_hcare methods
            # ---------------------------------------------------------------
            total_w = sum(all_weights)
            norm_w = [w / total_w for w in all_weights]
            n_tensors = len(all_client_params[0])
            new_global: list[np.ndarray] = []
            for pidx in range(n_tensors):
                weighted = sum(
                    nw * all_client_params[ci][pidx]
                    for ci, nw in enumerate(norm_w)
                )
                new_global.append(weighted)
            global_params = new_global

        # ---------------------------------------------------------------
        # Step 6: Track server-side DP accounting
        # ---------------------------------------------------------------
        if dp_accountant is not None:
            # Track that one more round of DP was applied (client-side noise)
            dp_accountant._steps += 1
            dp_accountant._sample_rates.append(
                min(batch_size / max(len(train_ds), 1), 1.0)
            )

        # ---------------------------------------------------------------
        # Step 7: Evaluate every round
        # ---------------------------------------------------------------
        eval_results = []
        for client in clients:
            loss, n_eval, eval_metrics = client.evaluate(
                parameters=[arr.copy() for arr in global_params],
                config={"round": rnd},
            )
            eval_results.append((loss, n_eval, eval_metrics))

        total_eval = sum(r[1] for r in eval_results)
        avg_loss = sum(r[0] * r[1] for r in eval_results) / total_eval if total_eval > 0 else 0.0
        avg_acc = sum(r[2].get("accuracy", 0.0) * r[1] for r in eval_results) / total_eval if total_eval > 0 else 0.0
        per_client_acc = [r[2].get("accuracy", 0.0) for r in eval_results]

        # ---------------------------------------------------------------
        # Step 8: Record comprehensive metrics
        # ---------------------------------------------------------------
        grad_norms = [r[3].get("grad_norm", 0.0) for r in fit_results]
        avg_grad_norm = np.nanmean(grad_norms) if grad_norms else 0.0

        epsilon_spent = dp_accountant.get_epsilon() if dp_accountant else 0.0

        metrics_collector.record_round(
            round_num=rnd,
            method=method,
            global_accuracy=avg_acc,
            global_loss=avg_loss,
            num_clients=len(selected_ids),
            total_clients=num_clients,
            clients_selected=selected_ids,
            avg_grad_norm=float(avg_grad_norm),
            per_client_accuracy=per_client_acc,
            epsilon_spent=epsilon_spent,
            bytes_original=bandwidth_report["original_bytes"],
            bytes_quantized=bandwidth_report["quantized_bytes"],
            compression_ratio=bandwidth_report["compression_ratio"],
            savings_percent=bandwidth_report["savings_percent"],
            encryption_status=encryption_report["status"],
            encryption_latency_ms=encryption_report["latency_ms"],
        )

        logger.info("  Round %d — accuracy: %.4f  loss: %.4f  eps: %.2f  clients: %d/%d",
                     rnd, avg_acc, avg_loss, epsilon_spent, len(selected_ids), num_clients)

        final_metrics = {
            "final_loss": avg_loss,
            "final_accuracy": avg_acc,
            "rounds_completed": rnd,
            "method": method,
            "epsilon_spent": epsilon_spent,
        }

    return final_metrics


def _adaptive_select(
    num_clients: int,
    n_select: int,
    gradient_norms: dict[int, float],
    last_participation: dict[int, int],
    current_round: int,
    min_interval: int,
    seed: int,
) -> list[int]:
    """Adaptive client selection based on gradient norms (Eq.9).

    p_i = ||grad_i|| / sum(||grad_j||)
    Clients with larger gradient change participate more often.
    Clients that haven't participated recently are force-included.
    """
    rng = np.random.RandomState(seed)

    # Force-include clients that haven't participated recently
    forced = []
    candidates = []
    for cid in range(num_clients):
        rounds_since = current_round - last_participation.get(cid, 0)
        if rounds_since >= min_interval:
            forced.append(cid)
        else:
            candidates.append(cid)

    n_to_sample = max(0, n_select - len(forced))

    if n_to_sample > 0 and candidates:
        # Eq.9: p_i = ||grad_i|| / sum(||grad_j||)
        norms = np.array([gradient_norms.get(cid, 1.0) for cid in candidates])
        norms = np.clip(norms, 1e-8, None)  # avoid zero
        probs = norms / norms.sum()
        n_sample = min(n_to_sample, len(candidates))
        sampled_idx = rng.choice(len(candidates), size=n_sample, replace=False, p=probs)
        selected = forced + [candidates[i] for i in sampled_idx]
    else:
        selected = forced[:n_select] if len(forced) > n_select else forced

    # Ensure at least 2 clients
    if len(selected) < 2:
        remaining = [c for c in range(num_clients) if c not in selected]
        while len(selected) < min(2, num_clients) and remaining:
            selected.append(remaining.pop(0))

    return selected


def _record_round_metrics(
    mc, rnd, method, num_clients, selected_ids, fit_results,
    prev_acc, prev_loss, per_client_acc, quantizer, secure_agg, dp_accountant,
):
    """Record metrics when aggregation is skipped (NaN fallback)."""
    grad_norms = [r[3].get("grad_norm", 0.0) for r in fit_results]
    mc.record_round(
        round_num=rnd,
        method=method,
        global_accuracy=prev_acc,
        global_loss=prev_loss,
        num_clients=len(selected_ids),
        total_clients=num_clients,
        clients_selected=selected_ids,
        avg_grad_norm=float(np.nanmean(grad_norms)) if grad_norms else 0.0,
        per_client_accuracy=per_client_acc,
        epsilon_spent=dp_accountant.get_epsilon() if dp_accountant else 0.0,
        bytes_original=0, bytes_quantized=0,
        compression_ratio=1.0, savings_percent=0.0,
        encryption_status="skipped", encryption_latency_ms=0,
    )
