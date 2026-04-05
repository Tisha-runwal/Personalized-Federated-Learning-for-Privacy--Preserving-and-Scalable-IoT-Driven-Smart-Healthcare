"""Dataset information and partition-preview routes."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# ---------------------------------------------------------------------------
# Static dataset catalogue
# ---------------------------------------------------------------------------

_DATASETS = {
    "har": {
        "name": "UCI HAR",
        "description": "Human Activity Recognition using smartphone sensor data (6 activities, 561 features).",
        "num_classes": 6,
        "feature_dim": 561,
        "active_tier": "synthetic",
        "source": "UCI ML Repository",
    },
    "mimic": {
        "name": "MIMIC-III (synthetic fallback)",
        "description": "Medical records dataset. Real data requires credentialed PhysioNet access; "
                       "a synthetic surrogate is used when the raw files are absent.",
        "num_classes": 2,
        "feature_dim": 76,
        "active_tier": "synthetic",
        "source": "PhysioNet / synthetic fallback",
    },
}


@router.get("/info")
async def dataset_info() -> dict[str, Any]:
    """Return available datasets and their active data tier."""
    return {"datasets": _DATASETS}


# ---------------------------------------------------------------------------
# Partition preview
# ---------------------------------------------------------------------------

class PartitionPreviewRequest(BaseModel):
    num_clients: int = Field(10, ge=2, le=100)
    alpha: float = Field(0.5, gt=0.0, description="Dirichlet concentration parameter")
    seed: int = Field(42)


def _dirichlet_partition(
    num_clients: int,
    num_classes: int,
    total_samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Simulate class distributions for each client using a Dirichlet draw."""
    proportions = rng.dirichlet(alpha=[alpha] * num_classes, size=num_clients)
    summaries = []
    for cid in range(num_clients):
        client_total = int(total_samples * (1.0 / num_clients))
        class_counts = (proportions[cid] * client_total).astype(int)
        # Make sure we hit the target total exactly
        diff = client_total - int(class_counts.sum())
        class_counts[0] += diff
        summaries.append(
            {
                "client_id": cid,
                "total_samples": int(class_counts.sum()),
                "class_distribution": class_counts.tolist(),
            }
        )
    return summaries


def _heterogeneity_score(client_summaries: list[dict[str, Any]], num_classes: int) -> float:
    """Compute an earth-mover-distance-inspired heterogeneity score in [0, 1].

    A score near 0 means IID; near 1 means maximally non-IID.
    """
    # Average class probabilities across clients (should be ~uniform)
    distributions = []
    for cs in client_summaries:
        total = cs["total_samples"]
        if total == 0:
            distributions.append([1.0 / num_classes] * num_classes)
        else:
            distributions.append([c / total for c in cs["class_distribution"]])

    distributions_np = np.array(distributions)
    uniform = np.ones(num_classes) / num_classes

    # Mean Jensen-Shannon divergence vs uniform
    def _js_div(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        # Avoid log(0)
        p_safe = np.where(p > 0, p, 1e-12)
        q_safe = np.where(q > 0, q, 1e-12)
        m_safe = np.where(m > 0, m, 1e-12)
        kl_pm = np.sum(p_safe * np.log(p_safe / m_safe))
        kl_qm = np.sum(q_safe * np.log(q_safe / m_safe))
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    js_values = [_js_div(d, uniform) for d in distributions_np]
    # Max possible JS divergence is log(2) ≈ 0.693
    max_js = math.log(2)
    score = float(np.mean(js_values)) / max_js
    return round(min(max(score, 0.0), 1.0), 4)


@router.post("/partition-preview")
async def partition_preview(request: PartitionPreviewRequest) -> dict[str, Any]:
    """Return a synthetic partition summary and heterogeneity score."""
    rng = np.random.default_rng(request.seed)
    num_classes = 6  # HAR default
    total_samples = 7352  # Approximate HAR training size

    summaries = _dirichlet_partition(
        num_clients=request.num_clients,
        num_classes=num_classes,
        total_samples=total_samples,
        alpha=request.alpha,
        rng=rng,
    )
    score = _heterogeneity_score(summaries, num_classes)

    return {
        "num_clients": request.num_clients,
        "alpha": request.alpha,
        "seed": request.seed,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "heterogeneity_score": score,
        "clients": summaries,
    }
