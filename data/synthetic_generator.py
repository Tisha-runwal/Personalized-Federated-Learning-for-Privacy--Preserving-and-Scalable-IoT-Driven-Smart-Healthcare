"""Tier 4 fallback: generates synthetic medical vital signs data."""

import numpy as np


class SyntheticMedicalGenerator:
    FEATURE_NAMES = [
        "heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature",
        "respiratory_rate", "age", "bmi", "glucose", "cholesterol",
        "creatinine", "hemoglobin", "wbc_count",
    ]

    CLUSTER_PARAMS = {
        "healthy": {
            "heart_rate": (72, 8), "systolic_bp": (120, 10), "diastolic_bp": (80, 7),
            "spo2": (98, 1), "temperature": (36.6, 0.3), "respiratory_rate": (16, 2),
            "age": (45, 15), "bmi": (24, 3), "glucose": (90, 10),
            "cholesterol": (190, 20), "creatinine": (0.9, 0.2),
            "hemoglobin": (14, 1.5), "wbc_count": (7000, 1500),
        },
        "at_risk": {
            "heart_rate": (88, 12), "systolic_bp": (145, 15), "diastolic_bp": (92, 10),
            "spo2": (95, 2), "temperature": (37.2, 0.5), "respiratory_rate": (20, 3),
            "age": (60, 12), "bmi": (29, 4), "glucose": (130, 25),
            "cholesterol": (240, 30), "creatinine": (1.4, 0.4),
            "hemoglobin": (12, 1.5), "wbc_count": (9000, 2000),
        },
        "critical": {
            "heart_rate": (110, 20), "systolic_bp": (170, 25), "diastolic_bp": (105, 15),
            "spo2": (90, 4), "temperature": (38.5, 1.0), "respiratory_rate": (28, 5),
            "age": (70, 10), "bmi": (32, 5), "glucose": (200, 50),
            "cholesterol": (280, 40), "creatinine": (2.5, 0.8),
            "hemoglobin": (10, 2), "wbc_count": (14000, 4000),
        },
    }

    CLUSTER_PROPORTIONS = {"healthy": 0.6, "at_risk": 0.3, "critical": 0.1}

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.seed)
        X_parts, y_parts = [], []

        for cluster_name, proportion in self.CLUSTER_PROPORTIONS.items():
            n = int(self.n_samples * proportion)
            params = self.CLUSTER_PARAMS[cluster_name]
            X_cluster = np.column_stack([
                rng.normal(params[feat][0], params[feat][1], n)
                for feat in self.FEATURE_NAMES
            ])
            label = 0 if cluster_name == "healthy" else 1
            y_cluster = np.full(n, label)
            X_parts.append(X_cluster)
            y_parts.append(y_cluster)

        X = np.vstack(X_parts).astype(np.float32)
        y = np.concatenate(y_parts).astype(np.int64)

        indices = rng.permutation(len(X))
        return X[indices], y[indices]
