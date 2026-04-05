"""Weighted federated aggregation utilities (Eq.1)."""
import torch


def weighted_average(
    client_params: list[list[torch.Tensor]],
    weights: list[float],
) -> list[torch.Tensor]:
    """Compute weighted average of client parameters (Eq.1).

    Args:
        client_params: List of parameter lists, one per client.
        weights: Per-client weights (e.g., number of local samples).

    Returns:
        Averaged parameter list.
    """
    total_weight = sum(weights)
    normalized = [w / total_weight for w in weights]
    n_params = len(client_params[0])
    averaged = []
    for param_idx in range(n_params):
        weighted_sum = sum(
            nw * client_params[ci][param_idx]
            for ci, nw in enumerate(normalized)
        )
        averaged.append(weighted_sum)
    return averaged
