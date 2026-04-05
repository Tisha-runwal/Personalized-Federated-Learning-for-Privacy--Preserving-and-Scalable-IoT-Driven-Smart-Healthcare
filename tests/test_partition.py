import numpy as np
import torch
from torch.utils.data import TensorDataset
from data.partition import DirichletPartitioner

def _make_dataset(n=1000, n_classes=6, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, 10)
    y = torch.randint(0, n_classes, (n,))
    return TensorDataset(X, y)

def test_partitioner_returns_correct_num_partitions():
    ds = _make_dataset()
    partitioner = DirichletPartitioner(num_clients=5, alpha=0.5, seed=42)
    partitions = partitioner.partition(ds)
    assert len(partitions) == 5

def test_partitioner_covers_all_samples():
    ds = _make_dataset(n=500)
    partitioner = DirichletPartitioner(num_clients=10, alpha=0.5, seed=42)
    partitions = partitioner.partition(ds)
    all_indices = []
    for indices in partitions:
        all_indices.extend(indices)
    assert sorted(all_indices) == list(range(500))

def test_low_alpha_creates_non_iid():
    ds = _make_dataset(n=1000, n_classes=6)
    partitioner = DirichletPartitioner(num_clients=5, alpha=0.1, seed=42)
    partitions = partitioner.partition(ds)
    labels = [ds[i][1].item() for i in range(len(ds))]
    for indices in partitions:
        client_labels = [labels[i] for i in indices]
        if len(client_labels) == 0:
            continue
        counts = np.bincount(client_labels, minlength=6)
        max_ratio = counts.max() / counts.sum()
        if max_ratio > 0.6:
            return
    assert False, "Expected at least one non-IID client with alpha=0.1"

def test_high_alpha_creates_near_iid():
    ds = _make_dataset(n=3000, n_classes=6)
    partitioner = DirichletPartitioner(num_clients=5, alpha=100.0, seed=42)
    partitions = partitioner.partition(ds)
    labels = [ds[i][1].item() for i in range(len(ds))]
    for indices in partitions:
        client_labels = [labels[i] for i in indices]
        counts = np.bincount(client_labels, minlength=6)
        ratios = counts / counts.sum()
        assert np.all(ratios > 0.1), f"Expected near-IID but got {ratios}"

def test_partition_deterministic():
    ds = _make_dataset()
    p1 = DirichletPartitioner(num_clients=5, alpha=0.5, seed=42).partition(ds)
    p2 = DirichletPartitioner(num_clients=5, alpha=0.5, seed=42).partition(ds)
    for a, b in zip(p1, p2):
        assert a == b

def test_heterogeneity_score():
    ds = _make_dataset(n=1000, n_classes=6)
    partitioner = DirichletPartitioner(num_clients=5, alpha=0.1, seed=42)
    partitions = partitioner.partition(ds)
    labels = [ds[i][1].item() for i in range(len(ds))]
    score = partitioner.heterogeneity_score(partitions, labels, n_classes=6)
    assert 0.0 <= score <= 1.0
