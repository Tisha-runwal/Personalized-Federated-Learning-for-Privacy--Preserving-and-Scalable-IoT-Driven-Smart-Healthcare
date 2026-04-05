"""Dirichlet-based non-IID data partitioning for federated learning."""
import numpy as np
from scipy.spatial.distance import jensenshannon
from torch.utils.data import Dataset

class DirichletPartitioner:
    def __init__(self, num_clients: int, alpha: float = 0.5, seed: int = 42):
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed

    def partition(self, dataset: Dataset) -> list[list[int]]:
        rng = np.random.RandomState(self.seed)
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
        n_classes = len(np.unique(labels))
        class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]
        client_indices: list[list[int]] = [[] for _ in range(self.num_clients)]
        for c in range(n_classes):
            indices_c = class_indices[c]
            rng.shuffle(indices_c)
            proportions = rng.dirichlet([self.alpha] * self.num_clients)
            counts = (proportions * len(indices_c)).astype(int)
            remainder = len(indices_c) - counts.sum()
            for i in range(remainder):
                counts[i % self.num_clients] += 1
            start = 0
            for client_id in range(self.num_clients):
                end = start + counts[client_id]
                client_indices[client_id].extend(indices_c[start:end])
                start = end
        for indices in client_indices:
            rng.shuffle(indices)
        return client_indices

    def heterogeneity_score(self, partitions, labels, n_classes):
        distributions = []
        for indices in partitions:
            if len(indices) == 0:
                continue
            client_labels = [labels[i] for i in indices]
            counts = np.bincount(client_labels, minlength=n_classes).astype(float)
            counts /= counts.sum()
            distributions.append(counts)
        if len(distributions) < 2:
            return 0.0
        jsd_values = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                jsd_values.append(jensenshannon(distributions[i], distributions[j]))
        return float(np.mean(jsd_values))

    def get_distribution_summary(self, partitions, labels, n_classes):
        summaries = []
        for client_id, indices in enumerate(partitions):
            client_labels = [labels[i] for i in indices]
            counts = np.bincount(client_labels, minlength=n_classes)
            total = counts.sum()
            ratios = (counts / total).tolist() if total > 0 else [0.0] * n_classes
            summaries.append({"client_id": client_id, "n_samples": len(indices), "class_distribution": ratios})
        return summaries
