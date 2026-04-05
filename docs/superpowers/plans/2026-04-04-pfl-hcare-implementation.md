# PFL-HCare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the PFL-HCare personalized federated learning framework with a real-time FastAPI + React dashboard for demonstration/showcase.

**Architecture:** Layered modular — core ML library (`pfl_hcare/`), dataset handlers (`data/`), FastAPI backend (`server/`), React dashboard (`client/`), Docker setup (`docker/`). Training runs via Flower FL framework, metrics stream over WebSocket to dashboard.

**Tech Stack:** PyTorch, Flower (flwr), Opacus, FastAPI, React 18 + TypeScript + Vite, Recharts, D3.js, Tailwind CSS, Framer Motion, SQLite

---

## Phase 1: Project Scaffolding & Configuration

### Task 1: Initialize project structure and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `configs/comparison.yaml`
- Create: `pfl_hcare/__init__.py`
- Create: `pfl_hcare/models/__init__.py`
- Create: `pfl_hcare/fl/__init__.py`
- Create: `pfl_hcare/fl/strategies/__init__.py`
- Create: `pfl_hcare/privacy/__init__.py`
- Create: `pfl_hcare/maml/__init__.py`
- Create: `pfl_hcare/metrics/__init__.py`
- Create: `data/__init__.py`
- Create: `server/__init__.py`
- Create: `server/routes/__init__.py`
- Create: `server/ws/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.0
flwr>=1.5
opacus>=1.4
fastapi>=0.100
uvicorn[standard]>=0.23
websockets>=11.0
pyyaml>=6.0
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
aiosqlite>=0.19
httpx>=0.24
wfdb>=4.1
reportlab>=4.0
scipy>=1.10
pytest>=7.0
pytest-asyncio>=0.21
```

- [ ] **Step 2: Create all `__init__.py` files for the package structure**

Create empty `__init__.py` in each directory listed above. For `pfl_hcare/__init__.py`:

```python
"""PFL-HCare: Personalized Federated Learning for IoT-Driven Smart Healthcare."""

__version__ = "0.1.0"
```

All other `__init__.py` files are empty.

- [ ] **Step 3: Create `configs/default.yaml`**

```yaml
training:
  learning_rate: 0.01
  batch_size: 32
  num_clients: 10
  num_rounds: 200
  local_epochs: 5
  seed: 42

maml:
  inner_lr: 0.01
  inner_steps: 5
  second_order: true

privacy:
  noise_multiplier: 0.5
  max_grad_norm: 1.0
  delta: 1.0e-5
  target_epsilon: 10.0

quantization:
  k_bits: 8
  enabled: true

client_selection:
  adaptive: true
  min_participation_interval: 10

secure_aggregation:
  simulated: true
  latency_range_ms: [50, 200]

dataset:
  name: "har"
  partition_alpha: 0.5
  test_fraction: 0.3
```

- [ ] **Step 4: Create `configs/comparison.yaml`**

```yaml
comparison_run:
  methods: [fedavg, fedprox, per_fedavg, pfedme, pfl_hcare]
  datasets: [har, mimic]
  rounds: 200
  clients: 10
  seeds: [42, 123, 456]
  mode: sequential
  save_results: true
```

- [ ] **Step 5: Install Python dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 6: Commit**

```bash
git add requirements.txt configs/ pfl_hcare/ data/__init__.py server/ tests/__init__.py
git commit -m "feat: scaffold project structure with configs and dependencies"
```

---

## Phase 2: Dataset Layer

### Task 2: Synthetic medical data generator

**Files:**
- Create: `data/synthetic_generator.py`
- Create: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthetic.py
import numpy as np
from data.synthetic_generator import SyntheticMedicalGenerator


def test_generator_produces_correct_shape():
    gen = SyntheticMedicalGenerator(n_samples=500, seed=42)
    X, y = gen.generate()
    assert X.shape == (500, 13)
    assert y.shape == (500,)


def test_generator_produces_binary_labels():
    gen = SyntheticMedicalGenerator(n_samples=100, seed=42)
    X, y = gen.generate()
    assert set(np.unique(y)).issubset({0, 1})


def test_generator_is_deterministic():
    gen1 = SyntheticMedicalGenerator(n_samples=100, seed=42)
    gen2 = SyntheticMedicalGenerator(n_samples=100, seed=42)
    X1, y1 = gen1.generate()
    X2, y2 = gen2.generate()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generator_cluster_proportions():
    gen = SyntheticMedicalGenerator(n_samples=1000, seed=42)
    X, y = gen.generate()
    # Healthy cluster should be majority
    healthy_ratio = np.sum(y == 0) / len(y)
    assert 0.5 < healthy_ratio < 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning && python -m pytest tests/test_synthetic.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# data/synthetic_generator.py
"""Tier 4 fallback: generates synthetic medical vital signs data."""

import numpy as np


class SyntheticMedicalGenerator:
    """Generates synthetic patient data with 3 clusters: healthy, at-risk, critical.

    Features: heart_rate, systolic_bp, diastolic_bp, spo2, temperature,
    respiratory_rate, age, bmi, glucose, cholesterol, creatinine, hemoglobin, wbc_count
    """

    FEATURE_NAMES = [
        "heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature",
        "respiratory_rate", "age", "bmi", "glucose", "cholesterol",
        "creatinine", "hemoglobin", "wbc_count",
    ]

    # (mean, std) per feature for each cluster
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

        # Shuffle deterministically
        indices = rng.permutation(len(X))
        return X[indices], y[indices]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_synthetic.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/synthetic_generator.py tests/test_synthetic.py
git commit -m "feat: add synthetic medical data generator (Tier 4 fallback)"
```

### Task 3: UCI HAR dataset loader

**Files:**
- Create: `data/har_loader.py`
- Create: `scripts/download_data.py`
- Create: `tests/test_har_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_har_loader.py
import numpy as np
import torch
from data.har_loader import HARDataset


def test_har_dataset_returns_tensors():
    ds = HARDataset(root="./datasets", download=True, split="train")
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_har_dataset_shape():
    ds = HARDataset(root="./datasets", download=True, split="train")
    x, y = ds[0]
    assert x.dim() == 1  # 561 features flattened
    assert x.shape[0] == 561
    assert y.dim() == 0  # scalar label


def test_har_dataset_classes():
    ds = HARDataset(root="./datasets", download=True, split="train")
    labels = set()
    for i in range(min(100, len(ds))):
        _, y = ds[i]
        labels.add(y.item())
    assert labels.issubset({0, 1, 2, 3, 4, 5})


def test_har_dataset_length():
    ds = HARDataset(root="./datasets", download=True, split="train")
    assert len(ds) > 5000  # Training set should have ~7352 samples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_har_loader.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write download script**

```python
# scripts/download_data.py
"""Download datasets for PFL-HCare experiments."""

import os
import zipfile
import urllib.request
import sys


HAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download_har(root: str = "./datasets") -> str:
    har_dir = os.path.join(root, "UCI HAR Dataset")
    if os.path.exists(har_dir):
        print(f"UCI HAR already exists at {har_dir}")
        return har_dir

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "har.zip")
    print("Downloading UCI HAR Dataset...")
    urllib.request.urlretrieve(HAR_URL, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    os.remove(zip_path)
    print(f"UCI HAR Dataset ready at {har_dir}")
    return har_dir


def download_heart_disease(root: str = "./datasets") -> str:
    heart_path = os.path.join(root, "heart_disease", "processed.cleveland.data")
    if os.path.exists(heart_path):
        print(f"Heart Disease already exists at {heart_path}")
        return heart_path

    os.makedirs(os.path.join(root, "heart_disease"), exist_ok=True)
    print("Downloading Cleveland Heart Disease Dataset...")
    urllib.request.urlretrieve(HEART_URL, heart_path)
    print(f"Heart Disease Dataset ready at {heart_path}")
    return heart_path


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "all"
    root = sys.argv[2] if len(sys.argv) > 2 else "./datasets"

    if dataset in ("har", "all"):
        download_har(root)
    if dataset in ("heart", "all"):
        download_heart_disease(root)
    print("Done.")
```

- [ ] **Step 4: Write HAR loader implementation**

```python
# data/har_loader.py
"""UCI HAR Dataset loader for federated learning experiments."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.download_data import download_har


class HARDataset(Dataset):
    """UCI Human Activity Recognition dataset.

    6 classes: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
    561 features from accelerometer + gyroscope signals.
    """

    CLASSES = [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
        "SITTING", "STANDING", "LAYING",
    ]

    def __init__(self, root: str = "./datasets", download: bool = True, split: str = "train"):
        assert split in ("train", "test")
        har_dir = os.path.join(root, "UCI HAR Dataset")

        if download and not os.path.exists(har_dir):
            download_har(root)

        split_dir = os.path.join(har_dir, split if split == "train" else "test")
        X_path = os.path.join(split_dir, f"X_{split}.txt")
        y_path = os.path.join(split_dir, f"y_{split}.txt")

        self.X = torch.tensor(
            np.loadtxt(X_path, dtype=np.float32)
        )
        self.y = torch.tensor(
            np.loadtxt(y_path, dtype=np.int64).flatten() - 1  # 1-indexed to 0-indexed
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_har_loader.py -v`
Expected: All 4 tests PASS (downloads dataset on first run)

- [ ] **Step 6: Commit**

```bash
git add data/har_loader.py scripts/download_data.py tests/test_har_loader.py
git commit -m "feat: add UCI HAR dataset loader with auto-download"
```

### Task 4: Medical dataset loader with fallback chain

**Files:**
- Create: `data/mimic_loader.py`
- Create: `tests/test_mimic_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mimic_loader.py
import torch
from data.mimic_loader import MedicalDataset


def test_medical_dataset_fallback_to_synthetic():
    """Without MIMIC-III credentials, should fall back to synthetic."""
    ds = MedicalDataset(root="./datasets", split="train")
    assert ds.active_tier in ("mimic3_full", "mimic3_demo", "heart_disease", "synthetic")
    assert len(ds) > 0


def test_medical_dataset_returns_tensors():
    ds = MedicalDataset(root="./datasets", split="train")
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_medical_dataset_feature_count():
    ds = MedicalDataset(root="./datasets", split="train")
    x, _ = ds[0]
    assert x.shape[0] == 13  # 13 features for all tiers


def test_medical_dataset_binary_labels():
    ds = MedicalDataset(root="./datasets", split="train")
    labels = set()
    for i in range(min(50, len(ds))):
        _, y = ds[i]
        labels.add(y.item())
    assert labels.issubset({0, 1})


def test_medical_dataset_deterministic():
    ds1 = MedicalDataset(root="./datasets", split="train", seed=42)
    ds2 = MedicalDataset(root="./datasets", split="train", seed=42)
    x1, y1 = ds1[0]
    x2, y2 = ds2[0]
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mimic_loader.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# data/mimic_loader.py
"""Medical dataset loader with 4-tier fallback chain."""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.synthetic_generator import SyntheticMedicalGenerator

logger = logging.getLogger(__name__)


class MedicalDataset(Dataset):
    """Medical dataset with cascading fallback:
    Tier 1: MIMIC-III Full → Tier 2: MIMIC-III Demo → Tier 3: Heart Disease UCI → Tier 4: Synthetic
    """

    def __init__(
        self,
        root: str = "./datasets",
        split: str = "train",
        test_fraction: float = 0.3,
        seed: int = 42,
        n_synthetic: int = 2000,
    ):
        assert split in ("train", "test")
        self.root = root
        self.seed = seed
        self.active_tier = "unknown"

        X, y = self._load_with_fallback(n_synthetic)

        # Normalize features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=seed, stratify=y,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if split == "train":
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.int64)

    def _load_with_fallback(self, n_synthetic: int) -> tuple[np.ndarray, np.ndarray]:
        # Tier 1: MIMIC-III Full
        mimic_path = os.path.join(self.root, "mimic3", "processed.csv")
        if os.path.exists(mimic_path):
            logger.info("Loading MIMIC-III Full (Tier 1)")
            self.active_tier = "mimic3_full"
            return self._load_csv(mimic_path)

        # Tier 2: MIMIC-III Demo
        demo_path = os.path.join(self.root, "mimic3_demo", "processed.csv")
        if os.path.exists(demo_path):
            logger.info("Loading MIMIC-III Demo (Tier 2)")
            self.active_tier = "mimic3_demo"
            return self._load_csv(demo_path)

        # Tier 3: Heart Disease UCI
        heart_path = os.path.join(self.root, "heart_disease", "processed.cleveland.data")
        if os.path.exists(heart_path):
            logger.info("Loading Heart Disease UCI (Tier 3)")
            self.active_tier = "heart_disease"
            return self._load_heart_disease(heart_path)

        # Tier 4: Synthetic
        logger.info("Loading Synthetic Medical Data (Tier 4)")
        self.active_tier = "synthetic"
        gen = SyntheticMedicalGenerator(n_samples=n_synthetic, seed=self.seed)
        return gen.generate()

    def _load_csv(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        y = df.iloc[:, -1].values.astype(np.int64)
        X = df.iloc[:, :-1].values.astype(np.float32)
        # Pad or truncate to 13 features
        if X.shape[1] < 13:
            X = np.pad(X, ((0, 0), (0, 13 - X.shape[1])))
        elif X.shape[1] > 13:
            X = X[:, :13]
        return X, y

    def _load_heart_disease(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path, header=None, na_values="?")
        df = df.dropna()
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = (df.iloc[:, -1].values > 0).astype(np.int64)  # Binary: disease present or not
        return X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mimic_loader.py -v`
Expected: All 5 tests PASS (will use synthetic fallback)

- [ ] **Step 5: Commit**

```bash
git add data/mimic_loader.py tests/test_mimic_loader.py
git commit -m "feat: add medical dataset loader with 4-tier fallback chain"
```

### Task 5: Non-IID Dirichlet data partitioning

**Files:**
- Create: `data/partition.py`
- Create: `tests/test_partition.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_partition.py
import numpy as np
import torch
from torch.utils.data import TensorDataset
from data.partition import DirichletPartitioner


def _make_dataset(n=1000, n_classes=6):
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
    # With alpha=0.1, at least one client should have >60% of one class
    labels = [ds[i][1].item() for i in range(len(ds))]
    for indices in partitions:
        client_labels = [labels[i] for i in indices]
        if len(client_labels) == 0:
            continue
        counts = np.bincount(client_labels, minlength=6)
        max_ratio = counts.max() / counts.sum()
        if max_ratio > 0.6:
            return  # Found a non-IID client
    # At least one should be non-IID with alpha=0.1
    assert False, "Expected at least one non-IID client with alpha=0.1"


def test_high_alpha_creates_near_iid():
    ds = _make_dataset(n=3000, n_classes=6)
    partitioner = DirichletPartitioner(num_clients=5, alpha=100.0, seed=42)
    partitions = partitioner.partition(ds)
    labels = [ds[i][1].item() for i in range(len(ds))]
    # All clients should have roughly uniform distribution
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_partition.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# data/partition.py
"""Dirichlet-based non-IID data partitioning for federated learning."""

import numpy as np
from scipy.spatial.distance import jensenshannon
from torch.utils.data import Dataset


class DirichletPartitioner:
    """Partition a dataset into non-IID shards using Dirichlet distribution.

    Args:
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter. Lower = more non-IID.
        seed: Random seed for reproducibility.
    """

    def __init__(self, num_clients: int, alpha: float = 0.5, seed: int = 42):
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed

    def partition(self, dataset: Dataset) -> list[list[int]]:
        """Partition dataset indices into per-client lists."""
        rng = np.random.RandomState(self.seed)
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
        n_classes = len(np.unique(labels))

        # Group indices by class
        class_indices = [np.where(labels == c)[0].tolist() for c in range(n_classes)]

        # Dirichlet allocation
        client_indices: list[list[int]] = [[] for _ in range(self.num_clients)]

        for c in range(n_classes):
            indices_c = class_indices[c]
            rng.shuffle(indices_c)

            # Draw proportions from Dirichlet
            proportions = rng.dirichlet([self.alpha] * self.num_clients)
            # Convert proportions to counts
            counts = (proportions * len(indices_c)).astype(int)
            # Distribute remainder
            remainder = len(indices_c) - counts.sum()
            for i in range(remainder):
                counts[i % self.num_clients] += 1

            start = 0
            for client_id in range(self.num_clients):
                end = start + counts[client_id]
                client_indices[client_id].extend(indices_c[start:end])
                start = end

        # Shuffle each client's indices
        for indices in client_indices:
            rng.shuffle(indices)

        return client_indices

    def heterogeneity_score(
        self, partitions: list[list[int]], labels: list[int], n_classes: int
    ) -> float:
        """Compute mean pairwise Jensen-Shannon divergence across clients."""
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

    def get_distribution_summary(
        self, partitions: list[list[int]], labels: list[int], n_classes: int
    ) -> list[dict]:
        """Return per-client class distribution for visualization."""
        summaries = []
        for client_id, indices in enumerate(partitions):
            client_labels = [labels[i] for i in indices]
            counts = np.bincount(client_labels, minlength=n_classes)
            total = counts.sum()
            ratios = (counts / total).tolist() if total > 0 else [0.0] * n_classes
            summaries.append({
                "client_id": client_id,
                "n_samples": len(indices),
                "class_distribution": ratios,
            })
        return summaries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_partition.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/partition.py tests/test_partition.py
git commit -m "feat: add Dirichlet non-IID data partitioning"
```

---

## Phase 3: Neural Network Models

### Task 6: Health classifier model

**Files:**
- Create: `pfl_hcare/models/health_classifier.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import torch
from pfl_hcare.models.health_classifier import HealthClassifier


def test_health_classifier_forward_shape():
    model = HealthClassifier(input_dim=13, num_classes=2)
    x = torch.randn(8, 13)
    out = model(x)
    assert out.shape == (8, 2)


def test_health_classifier_param_count():
    model = HealthClassifier(input_dim=13, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000  # Should be ~15K


def test_health_classifier_gradient_flow():
    model = HealthClassifier(input_dim=13, num_classes=2)
    x = torch.randn(4, 13)
    y = torch.randint(0, 2, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# pfl_hcare/models/health_classifier.py
"""MLP classifier for medical/health prediction tasks."""

import torch
import torch.nn as nn


class HealthClassifier(nn.Module):
    """3-layer MLP for health prediction. ~15K parameters.

    Architecture: Input → Dense(128)+BN+ReLU+Dropout → Dense(64)+BN+ReLU+Dropout → Dense(32)+ReLU → Output
    """

    def __init__(self, input_dim: int = 13, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pfl_hcare/models/health_classifier.py tests/test_models.py
git commit -m "feat: add HealthClassifier MLP model"
```

### Task 7: HAR classifier model

**Files:**
- Create: `pfl_hcare/models/har_classifier.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
from pfl_hcare.models.har_classifier import HARClassifier


def test_har_classifier_forward_shape():
    model = HARClassifier(num_classes=6)
    x = torch.randn(8, 9, 128)  # 9 channels, 128 timesteps
    out = model(x)
    assert out.shape == (8, 6)


def test_har_classifier_param_count():
    model = HARClassifier(num_classes=6)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000  # Should be ~52K


def test_har_classifier_flat_input():
    """Test that 561-dim flat input also works (auto-reshaped)."""
    model = HARClassifier(num_classes=6, accept_flat=True)
    x = torch.randn(8, 561)
    out = model(x)
    assert out.shape == (8, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py::test_har_classifier_forward_shape -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# pfl_hcare/models/har_classifier.py
"""1D-CNN classifier for Human Activity Recognition."""

import torch
import torch.nn as nn


class HARClassifier(nn.Module):
    """1D-CNN for HAR sensor data. ~52K parameters.

    Architecture: Conv1D(64,k=5) → Conv1D(128,k=3) → Conv1D(64,k=3) → GAP → Dense(64) → Output
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 9, accept_flat: bool = False):
        super().__init__()
        self.accept_flat = accept_flat
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.accept_flat and x.dim() == 2:
            # Reshape 561-dim flat input to (batch, 9, 62) — truncate last 3 features
            # 561 / 9 = 62.33, so use 9 * 62 = 558 features
            x = x[:, :558].reshape(-1, self.in_channels, 62)

        # x shape: (batch, channels, timesteps)
        x = self.features(x)
        x = x.mean(dim=2)  # Global average pooling
        return self.classifier(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pfl_hcare/models/har_classifier.py tests/test_models.py
git commit -m "feat: add HARClassifier 1D-CNN model"
```

---

## Phase 4: Core ML Components

### Task 8: MAML meta-learning implementation

**Files:**
- Create: `pfl_hcare/maml/maml.py`
- Create: `tests/test_maml.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_maml.py
import torch
import torch.nn as nn
from pfl_hcare.maml.maml import MAMLWrapper
from pfl_hcare.models.health_classifier import HealthClassifier


def test_maml_inner_loop_updates_params():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=3, second_order=False)

    X = torch.randn(16, 13)
    y = torch.randint(0, 2, (16,))

    original_params = [p.clone() for p in model.parameters()]
    adapted_params = maml.inner_loop(X, y)

    # Adapted params should differ from original
    for orig, adapted in zip(original_params, adapted_params):
        assert not torch.equal(orig, adapted)


def test_maml_inner_loop_returns_list_of_tensors():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=3, second_order=False)

    X = torch.randn(8, 13)
    y = torch.randint(0, 2, (8,))
    adapted_params = maml.inner_loop(X, y)

    model_params = list(model.parameters())
    assert len(adapted_params) == len(model_params)
    for ap, mp in zip(adapted_params, model_params):
        assert ap.shape == mp.shape


def test_maml_outer_loss_computes_gradient():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=1, second_order=False)

    support_X = torch.randn(8, 13)
    support_y = torch.randint(0, 2, (8,))
    query_X = torch.randn(8, 13)
    query_y = torch.randint(0, 2, (8,))

    loss = maml.outer_loss(support_X, support_y, query_X, query_y)
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_maml_second_order_mode():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=1, second_order=True)

    support_X = torch.randn(8, 13)
    support_y = torch.randint(0, 2, (8,))
    query_X = torch.randn(8, 13)
    query_y = torch.randint(0, 2, (8,))

    loss = maml.outer_loss(support_X, support_y, query_X, query_y)
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_maml.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# pfl_hcare/maml/maml.py
"""Model-Agnostic Meta-Learning (MAML) for personalized federated learning.

Implements Eq.3 (outer loop) and Eq.4 (inner loop) from the PFL-HCare paper.
Supports both full second-order MAML and first-order approximation (FOMAML).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MAMLWrapper:
    """Wraps a PyTorch model with MAML inner/outer loop logic.

    Args:
        model: The neural network to meta-learn.
        inner_lr: Learning rate for inner loop adaptation (alpha in Eq.4).
        inner_steps: Number of gradient steps in inner loop.
        second_order: If True, use full MAML (compute second-order gradients).
                      If False, use FOMAML (first-order approximation).
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        second_order: bool = False,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.second_order = second_order

    def inner_loop(
        self,
        support_X: torch.Tensor,
        support_y: torch.Tensor,
        params: Optional[list[torch.Tensor]] = None,
    ) -> list[torch.Tensor]:
        """Run inner loop adaptation on support set (Eq.4).

        Returns adapted parameters (does not modify model.parameters() in-place).
        """
        if params is None:
            params = [p.clone() for p in self.model.parameters()]

        for _ in range(self.inner_steps):
            # Forward pass with current params
            logits = self._forward_with_params(support_X, params)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients w.r.t. params
            grads = torch.autograd.grad(
                loss, params, create_graph=self.second_order,
            )

            # Update params: w_i = w - alpha * grad (Eq.4)
            params = [p - self.inner_lr * g for p, g in zip(params, grads)]

        return params

    def outer_loss(
        self,
        support_X: torch.Tensor,
        support_y: torch.Tensor,
        query_X: torch.Tensor,
        query_y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute meta-loss on query set after inner loop adaptation (Eq.3).

        This is the loss used for the outer loop optimization.
        """
        # Inner loop: adapt on support set
        adapted_params = self.inner_loop(
            support_X, support_y,
            params=[p.clone().requires_grad_(True) for p in self.model.parameters()],
        )

        # Evaluate on query set with adapted params
        logits = self._forward_with_params(query_X, adapted_params)
        return F.cross_entropy(logits, query_y)

    def _forward_with_params(
        self, x: torch.Tensor, params: list[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using given parameters instead of model's own."""
        # Use functional forward by replacing parameters temporarily
        original_params = list(self.model.parameters())
        param_mapping = dict(zip(original_params, params))

        # Replace parameters
        for module in self.model.modules():
            for name, p in module.named_parameters(recurse=False):
                if p in param_mapping:
                    setattr(module, name, param_mapping[p])

        try:
            out = self.model(x)
        finally:
            # Restore original parameters
            for module in self.model.modules():
                for name, p in module.named_parameters(recurse=False):
                    for orig, repl in param_mapping.items():
                        if p is repl:
                            setattr(module, name, orig)

        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_maml.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pfl_hcare/maml/maml.py tests/test_maml.py
git commit -m "feat: add MAML meta-learning with FOMAML toggle"
```

### Task 9: Differential privacy module

**Files:**
- Create: `pfl_hcare/privacy/differential_privacy.py`
- Create: `tests/test_dp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dp.py
import torch
from pfl_hcare.privacy.differential_privacy import DPMechanism


def test_dp_adds_noise():
    dp = DPMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
    params = [torch.ones(10, 10)]
    noisy_params = dp.add_noise(params)
    assert not torch.equal(params[0], noisy_params[0])


def test_dp_clips_gradients():
    dp = DPMechanism(noise_multiplier=0.0, max_grad_norm=1.0, delta=1e-5)
    # Large gradient
    params = [torch.ones(100) * 100.0]
    clipped = dp.clip_gradients(params)
    norm = torch.norm(clipped[0])
    assert norm <= 1.0 + 1e-6


def test_dp_noise_scale_increases_with_multiplier():
    params = [torch.zeros(1000)]
    dp_low = DPMechanism(noise_multiplier=0.1, max_grad_norm=1.0, delta=1e-5)
    dp_high = DPMechanism(noise_multiplier=2.0, max_grad_norm=1.0, delta=1e-5)

    torch.manual_seed(42)
    noisy_low = dp_low.add_noise([p.clone() for p in params])
    torch.manual_seed(42)
    noisy_high = dp_high.add_noise([p.clone() for p in params])

    std_low = noisy_low[0].std().item()
    std_high = noisy_high[0].std().item()
    assert std_high > std_low


def test_dp_epsilon_tracking():
    dp = DPMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
    assert dp.get_epsilon() == 0.0

    params = [torch.randn(10)]
    for _ in range(10):
        dp.add_noise(params, sample_rate=0.1)

    assert dp.get_epsilon() > 0.0


def test_dp_zero_noise():
    dp = DPMechanism(noise_multiplier=0.0, max_grad_norm=1.0, delta=1e-5)
    params = [torch.ones(10)]
    clipped = dp.clip_gradients(params)
    noisy = dp.add_noise(clipped, track=False)
    # With zero noise, clipped unit vector should be returned
    assert torch.allclose(noisy[0], clipped[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dp.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# pfl_hcare/privacy/differential_privacy.py
"""Differential privacy mechanism for federated learning (Eq.5).

Uses Gaussian mechanism with RDP accounting for (epsilon, delta)-DP guarantees.
"""

import math
import torch


class DPMechanism:
    """Applies differential privacy to model updates.

    Args:
        noise_multiplier: Ratio of std of noise to sensitivity (sigma in Eq.5).
        max_grad_norm: Maximum L2 norm for gradient clipping.
        delta: Target delta for (epsilon, delta)-DP.
    """

    def __init__(
        self,
        noise_multiplier: float = 0.5,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self._steps = 0
        self._sample_rates: list[float] = []

    def clip_gradients(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        """Clip parameter updates to max_grad_norm L2 norm."""
        # Flatten all params, compute global norm
        flat = torch.cat([p.flatten() for p in params])
        total_norm = torch.norm(flat)
        clip_factor = min(1.0, self.max_grad_norm / (total_norm.item() + 1e-8))
        return [p * clip_factor for p in params]

    def add_noise(
        self,
        params: list[torch.Tensor],
        sample_rate: float = 1.0,
        track: bool = True,
    ) -> list[torch.Tensor]:
        """Add Gaussian noise to parameters (Eq.5): w_i' = w_i + N(0, sigma^2)."""
        if track:
            self._steps += 1
            self._sample_rates.append(sample_rate)

        if self.noise_multiplier == 0.0:
            return params

        sigma = self.noise_multiplier * self.max_grad_norm
        noisy = []
        for p in params:
            noise = torch.normal(mean=0.0, std=sigma, size=p.shape, device=p.device)
            noisy.append(p + noise)
        return noisy

    def get_epsilon(self) -> float:
        """Compute current epsilon using simplified RDP accounting."""
        if self._steps == 0 or self.noise_multiplier == 0.0:
            return 0.0

        # Simplified RDP-based epsilon calculation
        # For Gaussian mechanism: epsilon ≈ sqrt(2 * steps * log(1/delta)) / sigma
        avg_rate = sum(self._sample_rates) / len(self._sample_rates) if self._sample_rates else 1.0
        rdp_epsilon = math.sqrt(2.0 * self._steps * avg_rate * math.log(1.0 / self.delta))
        rdp_epsilon /= self.noise_multiplier
        return rdp_epsilon

    def get_privacy_report(self) -> dict:
        """Return current privacy state for dashboard."""
        return {
            "epsilon_spent": self.get_epsilon(),
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "steps": self._steps,
        }

    def reset(self) -> None:
        """Reset privacy accountant for a new run."""
        self._steps = 0
        self._sample_rates = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dp.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pfl_hcare/privacy/differential_privacy.py tests/test_dp.py
git commit -m "feat: add differential privacy mechanism with RDP accounting"
```

### Task 10: Gradient quantization

**Files:**
- Create: `pfl_hcare/privacy/quantization.py`
- Create: `tests/test_quantization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quantization.py
import torch
from pfl_hcare.privacy.quantization import GradientQuantizer


def test_quantize_dequantize_recovers_approximate_values():
    q = GradientQuantizer(k_bits=8)
    params = [torch.randn(100)]
    quantized, meta = q.quantize(params)
    dequantized = q.dequantize(quantized, meta)
    # 8-bit quantization should be close
    assert torch.allclose(params[0], dequantized[0], atol=0.05)


def test_quantize_reduces_size():
    q = GradientQuantizer(k_bits=8)
    params = [torch.randn(1000)]
    original_bytes = sum(p.numel() * 4 for p in params)  # float32 = 4 bytes
    quantized, meta = q.quantize(params)
    quantized_bytes = sum(p.numel() * (q.k_bits / 8) for p in quantized)
    assert quantized_bytes < original_bytes


def test_quantize_2bit():
    q = GradientQuantizer(k_bits=2)
    params = [torch.randn(100)]
    quantized, meta = q.quantize(params)
    # 2-bit: values should be 0, 1, 2, or 3
    for qp in quantized:
        assert qp.min() >= 0
        assert qp.max() <= 3


def test_quantize_16bit():
    q = GradientQuantizer(k_bits=16)
    params = [torch.randn(100)]
    quantized, meta = q.quantize(params)
    dequantized = q.dequantize(quantized, meta)
    # 16-bit should be very close
    assert torch.allclose(params[0], dequantized[0], atol=0.001)


def test_bandwidth_report():
    q = GradientQuantizer(k_bits=8)
    params = [torch.randn(1000)]
    q.quantize(params)
    report = q.get_bandwidth_report()
    assert report["original_bytes"] > 0
    assert report["quantized_bytes"] > 0
    assert report["compression_ratio"] > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_quantization.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# pfl_hcare/privacy/quantization.py
"""k-bit gradient quantization for communication-efficient FL (Eq.8).

Q(w_i) = round((w_i - w_min) / (w_max - w_min) * (2^k - 1))
"""

import torch


class GradientQuantizer:
    """Quantize model updates to k-bit representation for bandwidth reduction.

    Args:
        k_bits: Number of bits for quantization (2, 4, 8, or 16).
    """

    def __init__(self, k_bits: int = 8):
        assert k_bits in (2, 4, 8, 16), f"k_bits must be 2, 4, 8, or 16, got {k_bits}"
        self.k_bits = k_bits
        self._last_original_bytes = 0
        self._last_quantized_bytes = 0

    def quantize(
        self, params: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """Quantize parameters to k-bit. Returns quantized tensors + metadata for dequantization."""
        max_val = 2 ** self.k_bits - 1
        quantized = []
        meta = []

        original_bytes = 0
        quantized_bytes = 0

        for p in params:
            original_bytes += p.numel() * 4  # float32

            w_min = p.min().item()
            w_max = p.max().item()
            scale = w_max - w_min if w_max != w_min else 1.0

            # Eq.8: Q(w) = round((w - w_min) / (w_max - w_min) * (2^k - 1))
            q = torch.round((p - w_min) / scale * max_val).clamp(0, max_val)

            if self.k_bits <= 8:
                q = q.to(torch.uint8)
            else:
                q = q.to(torch.int16)

            quantized.append(q)
            meta.append({"w_min": w_min, "w_max": w_max, "shape": p.shape})
            quantized_bytes += p.numel() * (self.k_bits / 8)

        self._last_original_bytes = original_bytes
        self._last_quantized_bytes = quantized_bytes
        return quantized, meta

    def dequantize(
        self, quantized: list[torch.Tensor], meta: list[dict]
    ) -> list[torch.Tensor]:
        """Reconstruct float parameters from quantized representation."""
        max_val = 2 ** self.k_bits - 1
        params = []

        for q, m in zip(quantized, meta):
            w_min = m["w_min"]
            w_max = m["w_max"]
            scale = w_max - w_min if w_max != w_min else 1.0

            p = q.float() / max_val * scale + w_min
            params.append(p.reshape(m["shape"]))

        return params

    def get_bandwidth_report(self) -> dict:
        """Return bandwidth metrics for dashboard."""
        ratio = (
            self._last_original_bytes / self._last_quantized_bytes
            if self._last_quantized_bytes > 0
            else 0.0
        )
        return {
            "original_bytes": self._last_original_bytes,
            "quantized_bytes": self._last_quantized_bytes,
            "compression_ratio": ratio,
            "savings_percent": (1.0 - 1.0 / ratio) * 100 if ratio > 0 else 0.0,
            "k_bits": self.k_bits,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_quantization.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pfl_hcare/privacy/quantization.py tests/test_quantization.py
git commit -m "feat: add k-bit gradient quantization (Eq.8)"
```

### Task 11: Simulated secure aggregation

**Files:**
- Create: `pfl_hcare/privacy/secure_aggregation.py`
- Create: `tests/test_secure_agg.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_secure_agg.py
import torch
from pfl_hcare.privacy.secure_aggregation import SimulatedSecureAggregator


def test_encrypt_decrypt_preserves_values():
    sa = SimulatedSecureAggregator(latency_range_ms=(0, 0))
    params = [torch.randn(10, 10)]
    encrypted = sa.encrypt(params)
    decrypted = sa.decrypt(encrypted)
    for orig, dec in zip(params, decrypted):
        assert torch.equal(orig, dec)


def test_encrypt_produces_metadata():
    sa = SimulatedSecureAggregator(latency_range_ms=(50, 100))
    params = [torch.randn(10)]
    encrypted = sa.encrypt(params)
    report = sa.get_report()
    assert report["status"] == "encrypted"
    assert report["latency_ms"] >= 0


def test_aggregate_multiple_clients():
    sa = SimulatedSecureAggregator(latency_range_ms=(0, 0))
    client_params = [
        [torch.ones(5) * 1.0],
        [torch.ones(5) * 3.0],
    ]
    weights = [0.5, 0.5]
    result = sa.aggregate(client_params, weights)
    expected = torch.ones(5) * 2.0
    assert torch.allclose(result[0], expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_secure_agg.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# pfl_hcare/privacy/secure_aggregation.py
"""Simulated secure aggregation for dashboard visualization (Eq.6-7).

Not real encryption — simulates the workflow with latency and status tracking
for the dashboard's threat model visualization.
"""

import time
import random
import torch


class SimulatedSecureAggregator:
    """Simulates homomorphic encryption workflow for FL model aggregation.

    Args:
        latency_range_ms: (min, max) simulated encryption/decryption latency.
        seed: Random seed for latency simulation.
    """

    def __init__(
        self,
        latency_range_ms: tuple[int, int] = (50, 200),
        seed: int = 42,
    ):
        self.latency_range_ms = latency_range_ms
        self.rng = random.Random(seed)
        self._status = "idle"
        self._last_latency_ms = 0
        self._encrypted_size_bytes = 0

    def encrypt(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        """Simulate encryption of model parameters (Eq.6)."""
        self._status = "encrypting"
        latency = self.rng.randint(*self.latency_range_ms) if self.latency_range_ms[1] > 0 else 0
        time.sleep(latency / 1000.0)
        self._last_latency_ms = latency
        self._encrypted_size_bytes = sum(p.numel() * 4 for p in params)
        self._status = "encrypted"
        # Return unchanged — simulation only
        return [p.clone() for p in params]

    def decrypt(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        """Simulate decryption of model parameters (Eq.7)."""
        self._status = "decrypting"
        latency = self.rng.randint(*self.latency_range_ms) if self.latency_range_ms[1] > 0 else 0
        time.sleep(latency / 1000.0)
        self._last_latency_ms += latency
        self._status = "decrypted"
        return [p.clone() for p in params]

    def aggregate(
        self,
        client_params: list[list[torch.Tensor]],
        weights: list[float],
    ) -> list[torch.Tensor]:
        """Simulate encrypted aggregation: encrypt → aggregate → decrypt."""
        # Encrypt all client params
        encrypted = [self.encrypt(cp) for cp in client_params]

        # Weighted average (done "on encrypted data" — simulated)
        n_params = len(encrypted[0])
        aggregated = []
        for param_idx in range(n_params):
            weighted_sum = sum(
                w * encrypted[client_idx][param_idx]
                for client_idx, w in enumerate(weights)
            )
            aggregated.append(weighted_sum)

        # Decrypt result
        return self.decrypt(aggregated)

    def get_report(self) -> dict:
        """Return status for dashboard encryption widget."""
        return {
            "status": self._status,
            "latency_ms": self._last_latency_ms,
            "encrypted_size_bytes": self._encrypted_size_bytes,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_secure_agg.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pfl_hcare/privacy/secure_aggregation.py tests/test_secure_agg.py
git commit -m "feat: add simulated secure aggregation for dashboard visualization"
```

### Task 12: Metrics collector

**Files:**
- Create: `pfl_hcare/metrics/collector.py`
- Create: `tests/test_collector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collector.py
from pfl_hcare.metrics.collector import MetricsCollector


def test_collector_records_round():
    mc = MetricsCollector()
    mc.record_round(
        round_num=1,
        method="pfl_hcare",
        global_accuracy=0.85,
        global_loss=0.42,
        epsilon_spent=0.3,
        bytes_original=10000,
        bytes_quantized=2500,
        clients_selected=[0, 2, 4],
        per_client_accuracy=[0.82, 0.88, 0.84],
        per_client_gradient_norm=[0.12, 0.45, 0.33],
        encryption_latency_ms=127,
        round_time_ms=3400,
    )
    history = mc.get_history()
    assert len(history) == 1
    assert history[0]["round"] == 1
    assert history[0]["metrics"]["global_accuracy"] == 0.85


def test_collector_multiple_rounds():
    mc = MetricsCollector()
    for i in range(5):
        mc.record_round(
            round_num=i,
            method="fedavg",
            global_accuracy=0.80 + i * 0.02,
            global_loss=0.5 - i * 0.05,
        )
    history = mc.get_history()
    assert len(history) == 5


def test_collector_callbacks():
    received = []
    mc = MetricsCollector()
    mc.on_round(lambda data: received.append(data))
    mc.record_round(round_num=0, method="test", global_accuracy=0.9)
    assert len(received) == 1
    assert received[0]["metrics"]["global_accuracy"] == 0.9


def test_collector_to_json():
    mc = MetricsCollector()
    mc.record_round(round_num=0, method="test", global_accuracy=0.9)
    json_str = mc.to_json()
    assert '"global_accuracy": 0.9' in json_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collector.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

```python
# pfl_hcare/metrics/collector.py
"""Metrics collection and callback system for FL training rounds."""

import json
from typing import Any, Callable


class MetricsCollector:
    """Collects per-round metrics and notifies listeners via callbacks.

    Used by FL strategies to record metrics, and by FastAPI to stream to dashboard.
    """

    def __init__(self):
        self._history: list[dict] = []
        self._callbacks: list[Callable[[dict], None]] = []

    def record_round(self, round_num: int, method: str, **metrics: Any) -> None:
        """Record metrics for a single FL round."""
        entry = {
            "type": "round_update",
            "round": round_num,
            "method": method,
            "metrics": metrics,
        }
        self._history.append(entry)

        for cb in self._callbacks:
            cb(entry)

    def on_round(self, callback: Callable[[dict], None]) -> None:
        """Register a callback invoked after each round is recorded."""
        self._callbacks.append(callback)

    def get_history(self) -> list[dict]:
        """Return all recorded rounds."""
        return list(self._history)

    def get_latest(self) -> dict | None:
        """Return the most recent round entry."""
        return self._history[-1] if self._history else None

    def to_json(self) -> str:
        """Serialize history to JSON string."""
        return json.dumps(self._history, indent=2)

    def reset(self) -> None:
        """Clear history for a new run."""
        self._history.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collector.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pfl_hcare/metrics/collector.py tests/test_collector.py
git commit -m "feat: add metrics collector with callback system"
```

---

## Phase 5: Federated Learning Strategies

### Task 13: Aggregation utilities and Flower client

**Files:**
- Create: `pfl_hcare/fl/aggregation.py`
- Create: `pfl_hcare/fl/client.py`

- [ ] **Step 1: Write aggregation utilities**

```python
# pfl_hcare/fl/aggregation.py
"""Weighted federated aggregation utilities (Eq.1)."""

import torch


def weighted_average(
    client_params: list[list[torch.Tensor]],
    weights: list[float],
) -> list[torch.Tensor]:
    """Compute weighted average of client model parameters (Eq.1).

    w_global = sum(|D_i| / sum(|D_j|) * w_i)
    """
    total_weight = sum(weights)
    normalized = [w / total_weight for w in weights]

    n_params = len(client_params[0])
    averaged = []
    for param_idx in range(n_params):
        weighted_sum = sum(
            nw * client_params[client_idx][param_idx]
            for client_idx, nw in enumerate(normalized)
        )
        averaged.append(weighted_sum)

    return averaged
```

- [ ] **Step 2: Write the Flower client**

```python
# pfl_hcare/fl/client.py
"""Flower client implementation for PFL-HCare federated learning."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from typing import Optional

import flwr as fl
from flwr.common import NDArrays, Scalar

from pfl_hcare.maml.maml import MAMLWrapper
from pfl_hcare.privacy.differential_privacy import DPMechanism


class PFLClient(fl.client.NumPyClient):
    """Flower client supporting multiple FL strategies.

    Args:
        model: The neural network model.
        train_dataset: Local training data for this client.
        test_dataset: Local test data (optional).
        client_id: Unique identifier for this client.
        local_epochs: Number of local training epochs.
        batch_size: Local training batch size.
        lr: Local learning rate.
        strategy: One of "fedavg", "fedprox", "per_fedavg", "pfedme", "pfl_hcare".
        maml_wrapper: Optional MAML wrapper for personalized strategies.
        dp_mechanism: Optional DP mechanism for privacy-preserving strategies.
        mu: Proximal term weight for FedProx.
        lambd: Moreau envelope parameter for pFedMe.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Subset,
        test_dataset: Optional[Subset] = None,
        client_id: int = 0,
        local_epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.01,
        strategy: str = "fedavg",
        maml_wrapper: Optional[MAMLWrapper] = None,
        dp_mechanism: Optional[DPMechanism] = None,
        mu: float = 0.01,
        lambd: float = 15.0,
    ):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = (
            DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
        )
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.lr = lr
        self.strategy = strategy
        self.maml = maml_wrapper
        self.dp = dp_mechanism
        self.mu = mu
        self.lambd = lambd
        self.device = torch.device("cpu")
        self._gradient_norm = 0.0

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict]:
        self.set_parameters(parameters)
        self.model.to(self.device)

        if self.strategy in ("per_fedavg", "pfl_hcare") and self.maml:
            self._train_maml()
        elif self.strategy == "pfedme":
            self._train_pfedme(parameters)
        elif self.strategy == "fedprox":
            self._train_fedprox(parameters)
        else:
            self._train_standard()

        # Compute gradient norm for adaptive selection
        new_params = self.get_parameters({})
        flat_diff = torch.cat([
            torch.tensor(new_params[i] - parameters[i]).flatten()
            for i in range(len(parameters))
        ])
        self._gradient_norm = torch.norm(flat_diff).item()

        result_params = self.get_parameters({})

        # Apply DP if enabled
        if self.dp and self.strategy == "pfl_hcare":
            tensors = [torch.tensor(p) for p in result_params]
            tensors = self.dp.clip_gradients(tensors)
            tensors = self.dp.add_noise(tensors, sample_rate=1.0 / len(self.train_loader.dataset))
            result_params = [t.numpy() for t in tensors]

        return (
            result_params,
            len(self.train_loader.dataset),
            {"client_id": self.client_id, "gradient_norm": self._gradient_norm},
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        if self.test_loader is None:
            return 0.0, 0, {}

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                total_loss += nn.functional.cross_entropy(outputs, y, reduction="sum").item()
                correct += (outputs.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy, "client_id": self.client_id}

    def _train_standard(self) -> None:
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(self.model(X), y)
                loss.backward()
                optimizer.step()

    def _train_fedprox(self, global_params: NDArrays) -> None:
        self.model.train()
        global_tensors = [torch.tensor(p, device=self.device) for p in global_params]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(self.model(X), y)
                # Proximal term: (mu/2) * ||w - w_global||^2
                prox_term = sum(
                    torch.sum((p - gp) ** 2)
                    for p, gp in zip(self.model.parameters(), global_tensors)
                )
                loss += (self.mu / 2) * prox_term
                loss.backward()
                optimizer.step()

    def _train_maml(self) -> None:
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                # Split into support/query
                mid = len(X) // 2
                support_X, query_X = X[:mid], X[mid:]
                support_y, query_y = y[:mid], y[mid:]

                if len(support_X) == 0 or len(query_X) == 0:
                    continue

                optimizer.zero_grad()
                loss = self.maml.outer_loss(support_X, support_y, query_X, query_y)
                loss.backward()
                optimizer.step()

    def _train_pfedme(self, global_params: NDArrays) -> None:
        self.model.train()
        global_tensors = [torch.tensor(p, device=self.device) for p in global_params]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(self.model(X), y)
                # Moreau envelope: F_i(w) + (lambda/2) * ||w - theta_i||^2
                reg_term = sum(
                    torch.sum((p - gp) ** 2)
                    for p, gp in zip(self.model.parameters(), global_tensors)
                )
                loss += (self.lambd / 2) * reg_term
                loss.backward()
                optimizer.step()
```

- [ ] **Step 3: Commit**

```bash
git add pfl_hcare/fl/aggregation.py pfl_hcare/fl/client.py
git commit -m "feat: add FL aggregation utilities and multi-strategy Flower client"
```

### Task 14: FL strategies (all 5 methods)

**Files:**
- Create: `pfl_hcare/fl/strategies/fedavg.py`
- Create: `pfl_hcare/fl/strategies/fedprox.py`
- Create: `pfl_hcare/fl/strategies/per_fedavg.py`
- Create: `pfl_hcare/fl/strategies/pfedme.py`
- Create: `pfl_hcare/fl/strategies/pfl_hcare.py`
- Create: `tests/test_strategies.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_strategies.py
from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy
from pfl_hcare.fl.strategies.fedprox import FedProxStrategy
from pfl_hcare.fl.strategies.per_fedavg import PerFedAvgStrategy
from pfl_hcare.fl.strategies.pfedme import PFedMeStrategy
from pfl_hcare.fl.strategies.pfl_hcare import PFLHCareStrategy
from pfl_hcare.metrics.collector import MetricsCollector


def test_fedavg_strategy_creates():
    mc = MetricsCollector()
    strategy = FedAvgStrategy(metrics_collector=mc)
    assert strategy is not None


def test_fedprox_strategy_creates():
    mc = MetricsCollector()
    strategy = FedProxStrategy(metrics_collector=mc, mu=0.01)
    assert strategy is not None


def test_per_fedavg_strategy_creates():
    mc = MetricsCollector()
    strategy = PerFedAvgStrategy(metrics_collector=mc)
    assert strategy is not None


def test_pfedme_strategy_creates():
    mc = MetricsCollector()
    strategy = PFedMeStrategy(metrics_collector=mc, lambd=15.0)
    assert strategy is not None


def test_pfl_hcare_strategy_creates():
    mc = MetricsCollector()
    strategy = PFLHCareStrategy(
        metrics_collector=mc,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
        k_bits=8,
        adaptive_selection=True,
    )
    assert strategy is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_strategies.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write FedAvg strategy**

```python
# pfl_hcare/fl/strategies/fedavg.py
"""FedAvg strategy — vanilla federated averaging baseline."""

import flwr as fl
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from pfl_hcare.metrics.collector import MetricsCollector

import time
from typing import Optional


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg with metrics collection for dashboard."""

    def __init__(self, metrics_collector: MetricsCollector, **kwargs):
        super().__init__(**kwargs)
        self.mc = metrics_collector
        self.method_name = "fedavg"
        self._round_start = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        params, metrics = super().aggregate_fit(server_round, results, failures)
        round_time = (time.time() - self._round_start) * 1000

        client_info = []
        for _, fit_res in results:
            client_info.append(fit_res.metrics)

        self.mc.record_round(
            round_num=server_round,
            method=self.method_name,
            round_time_ms=round_time,
            clients_selected=[c.get("client_id", -1) for c in client_info],
            per_client_gradient_norm=[c.get("gradient_norm", 0.0) for c in client_info],
        )

        return params, metrics
```

- [ ] **Step 4: Write FedProx strategy**

```python
# pfl_hcare/fl/strategies/fedprox.py
"""FedProx strategy — FedAvg with proximal term."""

from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy
from pfl_hcare.metrics.collector import MetricsCollector


class FedProxStrategy(FedAvgStrategy):
    """FedProx: FedAvg + proximal regularization. mu is passed to clients."""

    def __init__(self, metrics_collector: MetricsCollector, mu: float = 0.01, **kwargs):
        super().__init__(metrics_collector=metrics_collector, **kwargs)
        self.mu = mu
        self.method_name = "fedprox"
```

- [ ] **Step 5: Write Per-FedAvg strategy**

```python
# pfl_hcare/fl/strategies/per_fedavg.py
"""Per-FedAvg strategy — MAML-based personalization without privacy."""

from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy
from pfl_hcare.metrics.collector import MetricsCollector


class PerFedAvgStrategy(FedAvgStrategy):
    """Per-FedAvg: MAML inner/outer loop, no DP, no quantization."""

    def __init__(self, metrics_collector: MetricsCollector, **kwargs):
        super().__init__(metrics_collector=metrics_collector, **kwargs)
        self.method_name = "per_fedavg"
```

- [ ] **Step 6: Write pFedMe strategy**

```python
# pfl_hcare/fl/strategies/pfedme.py
"""pFedMe strategy — Moreau envelope personalization."""

from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy
from pfl_hcare.metrics.collector import MetricsCollector


class PFedMeStrategy(FedAvgStrategy):
    """pFedMe: Moreau envelope F_i(w) + (lambda/2)||w - theta_i||^2."""

    def __init__(self, metrics_collector: MetricsCollector, lambd: float = 15.0, **kwargs):
        super().__init__(metrics_collector=metrics_collector, **kwargs)
        self.lambd = lambd
        self.method_name = "pfedme"
```

- [ ] **Step 7: Write PFL-HCare strategy**

```python
# pfl_hcare/fl/strategies/pfl_hcare.py
"""PFL-HCare strategy — full framework with MAML + DP + secure agg + quantization + adaptive selection."""

import time
import numpy as np
import torch
from typing import Optional

import flwr as fl
from flwr.common import Parameters, FitRes, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from pfl_hcare.metrics.collector import MetricsCollector
from pfl_hcare.privacy.differential_privacy import DPMechanism
from pfl_hcare.privacy.quantization import GradientQuantizer
from pfl_hcare.privacy.secure_aggregation import SimulatedSecureAggregator


class PFLHCareStrategy(fl.server.strategy.FedAvg):
    """Full PFL-HCare strategy with all components.

    - MAML personalization (handled client-side)
    - Differential privacy (noise added client-side, tracked here)
    - Simulated secure aggregation
    - Gradient quantization
    - Adaptive client selection (Eq.9)
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        noise_multiplier: float = 0.5,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        k_bits: int = 8,
        adaptive_selection: bool = True,
        min_participation_interval: int = 10,
        secure_agg_latency: tuple[int, int] = (50, 200),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mc = metrics_collector
        self.method_name = "pfl_hcare"

        self.dp = DPMechanism(noise_multiplier, max_grad_norm, delta)
        self.quantizer = GradientQuantizer(k_bits)
        self.secure_agg = SimulatedSecureAggregator(secure_agg_latency)

        self.adaptive_selection = adaptive_selection
        self.min_participation_interval = min_participation_interval
        self._client_gradient_norms: dict[int, float] = {}
        self._client_last_participation: dict[int, int] = {}
        self._round_start = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.time()
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        config = {"server_round": server_round}

        if self.adaptive_selection and server_round > 1 and self._client_gradient_norms:
            clients = client_manager.sample(
                num_clients=client_manager.num_available(), min_num_clients=min_num_clients,
            )
            selected = self._adaptive_select(clients, sample_size, server_round)
            return [(client, fl.common.FitIns(parameters, config)) for client in selected]

        return super().configure_fit(server_round, parameters, client_manager)

    def _adaptive_select(
        self, all_clients: list[ClientProxy], n_select: int, server_round: int,
    ) -> list[ClientProxy]:
        """Select clients proportional to gradient norms (Eq.9)."""
        # Force-include clients that haven't participated recently
        forced = []
        remaining = []
        for client in all_clients:
            cid = id(client)
            last_round = self._client_last_participation.get(cid, 0)
            if server_round - last_round >= self.min_participation_interval:
                forced.append(client)
            else:
                remaining.append(client)

        n_to_sample = max(0, n_select - len(forced))

        if n_to_sample > 0 and remaining:
            norms = np.array([
                self._client_gradient_norms.get(id(c), 1.0) for c in remaining
            ])
            probs = norms / norms.sum() if norms.sum() > 0 else np.ones(len(norms)) / len(norms)
            indices = np.random.choice(len(remaining), size=min(n_to_sample, len(remaining)), replace=False, p=probs)
            selected = forced + [remaining[i] for i in indices]
        else:
            selected = forced[:n_select]

        for c in selected:
            self._client_last_participation[id(c)] = server_round

        return selected

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        # Track gradient norms for adaptive selection
        client_info = []
        for client, fit_res in results:
            cid = fit_res.metrics.get("client_id", id(client))
            gnorm = fit_res.metrics.get("gradient_norm", 1.0)
            self._client_gradient_norms[id(client)] = gnorm
            client_info.append(fit_res.metrics)

        # Quantization bandwidth tracking
        if results:
            sample_params = [torch.tensor(p) for p in fl.common.parameters_to_ndarrays(results[0][1].parameters)]
            _, _ = self.quantizer.quantize(sample_params)
            bandwidth = self.quantizer.get_bandwidth_report()
        else:
            bandwidth = {"original_bytes": 0, "quantized_bytes": 0, "compression_ratio": 0, "savings_percent": 0}

        # Simulated secure aggregation report
        sa_report = self.secure_agg.get_report()

        # Standard FedAvg aggregation
        params, metrics = super().aggregate_fit(server_round, results, failures)

        round_time = (time.time() - self._round_start) * 1000

        self.mc.record_round(
            round_num=server_round,
            method=self.method_name,
            epsilon_spent=self.dp.get_epsilon(),
            bytes_original=bandwidth["original_bytes"],
            bytes_quantized=bandwidth["quantized_bytes"],
            compression_ratio=bandwidth["compression_ratio"],
            savings_percent=bandwidth["savings_percent"],
            encryption_latency_ms=sa_report["latency_ms"],
            encryption_status=sa_report["status"],
            clients_selected=[c.get("client_id", -1) for c in client_info],
            per_client_gradient_norm=[c.get("gradient_norm", 0.0) for c in client_info],
            round_time_ms=round_time,
        )

        return params, metrics
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_strategies.py -v`
Expected: All 5 tests PASS

- [ ] **Step 9: Commit**

```bash
git add pfl_hcare/fl/strategies/ tests/test_strategies.py
git commit -m "feat: add all 5 FL strategies (FedAvg, FedProx, Per-FedAvg, pFedMe, PFL-HCare)"
```

---

## Phase 6: FL Server & Local Runner

### Task 15: Flower server and local simulation runner

**Files:**
- Create: `pfl_hcare/fl/server.py`
- Create: `scripts/run_local.py`

- [ ] **Step 1: Write FL server**

```python
# pfl_hcare/fl/server.py
"""Flower server launcher for PFL-HCare federated learning."""

import yaml
import torch
import torch.nn as nn
import copy
from typing import Optional

import flwr as fl
from torch.utils.data import Subset

from pfl_hcare.models.health_classifier import HealthClassifier
from pfl_hcare.models.har_classifier import HARClassifier
from pfl_hcare.fl.client import PFLClient
from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy
from pfl_hcare.fl.strategies.fedprox import FedProxStrategy
from pfl_hcare.fl.strategies.per_fedavg import PerFedAvgStrategy
from pfl_hcare.fl.strategies.pfedme import PFedMeStrategy
from pfl_hcare.fl.strategies.pfl_hcare import PFLHCareStrategy
from pfl_hcare.maml.maml import MAMLWrapper
from pfl_hcare.privacy.differential_privacy import DPMechanism
from pfl_hcare.metrics.collector import MetricsCollector
from data.har_loader import HARDataset
from data.mimic_loader import MedicalDataset
from data.partition import DirichletPartitioner


def create_model(dataset_name: str) -> nn.Module:
    if dataset_name == "har":
        return HARClassifier(num_classes=6, accept_flat=True)
    else:
        return HealthClassifier(input_dim=13, num_classes=2)


def create_strategy(
    method: str,
    metrics_collector: MetricsCollector,
    config: dict,
) -> fl.server.strategy.Strategy:
    fraction_fit = config["training"].get("fraction_fit", 0.5)
    min_fit_clients = max(2, config["training"]["num_clients"] // 2)

    kwargs = dict(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=config["training"]["num_clients"],
    )

    if method == "fedavg":
        return FedAvgStrategy(metrics_collector=metrics_collector, **kwargs)
    elif method == "fedprox":
        return FedProxStrategy(metrics_collector=metrics_collector, mu=0.01, **kwargs)
    elif method == "per_fedavg":
        return PerFedAvgStrategy(metrics_collector=metrics_collector, **kwargs)
    elif method == "pfedme":
        return PFedMeStrategy(metrics_collector=metrics_collector, lambd=15.0, **kwargs)
    elif method == "pfl_hcare":
        return PFLHCareStrategy(
            metrics_collector=metrics_collector,
            noise_multiplier=config["privacy"]["noise_multiplier"],
            max_grad_norm=config["privacy"]["max_grad_norm"],
            delta=config["privacy"]["delta"],
            k_bits=config["quantization"]["k_bits"],
            adaptive_selection=config["client_selection"]["adaptive"],
            min_participation_interval=config["client_selection"]["min_participation_interval"],
            secure_agg_latency=tuple(config["secure_aggregation"]["latency_range_ms"]),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def load_dataset(dataset_name: str, root: str = "./datasets") -> tuple:
    if dataset_name == "har":
        train_ds = HARDataset(root=root, download=True, split="train")
        test_ds = HARDataset(root=root, download=True, split="test")
    else:
        train_ds = MedicalDataset(root=root, split="train")
        test_ds = MedicalDataset(root=root, split="test")
    return train_ds, test_ds


def run_simulation(
    config: dict,
    method: str = "pfl_hcare",
    metrics_collector: Optional[MetricsCollector] = None,
) -> MetricsCollector:
    """Run a federated learning simulation with the specified method."""
    if metrics_collector is None:
        metrics_collector = MetricsCollector()

    torch.manual_seed(config["training"]["seed"])

    dataset_name = config["dataset"]["name"]
    train_ds, test_ds = load_dataset(dataset_name)

    # Partition data
    partitioner = DirichletPartitioner(
        num_clients=config["training"]["num_clients"],
        alpha=config["dataset"]["partition_alpha"],
        seed=config["training"]["seed"],
    )
    partitions = partitioner.partition(train_ds)

    # Create strategy
    strategy = create_strategy(method, metrics_collector, config)

    # Create client function
    model_template = create_model(dataset_name)

    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        model = copy.deepcopy(model_template)

        maml_wrapper = None
        dp_mechanism = None

        if method in ("per_fedavg", "pfl_hcare"):
            maml_wrapper = MAMLWrapper(
                model,
                inner_lr=config["maml"]["inner_lr"],
                inner_steps=config["maml"]["inner_steps"],
                second_order=config["maml"]["second_order"],
            )

        if method == "pfl_hcare":
            dp_mechanism = DPMechanism(
                noise_multiplier=config["privacy"]["noise_multiplier"],
                max_grad_norm=config["privacy"]["max_grad_norm"],
                delta=config["privacy"]["delta"],
            )

        client_subset = Subset(train_ds, partitions[client_id])
        test_subset = Subset(test_ds, list(range(len(test_ds))))

        return PFLClient(
            model=model,
            train_dataset=client_subset,
            test_dataset=test_subset,
            client_id=client_id,
            local_epochs=config["training"]["local_epochs"],
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["learning_rate"],
            strategy=method,
            maml_wrapper=maml_wrapper,
            dp_mechanism=dp_mechanism,
        ).to_client()

    # Run simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config["training"]["num_clients"],
        config=fl.server.ServerConfig(num_rounds=config["training"]["num_rounds"]),
        strategy=strategy,
    )

    return metrics_collector
```

- [ ] **Step 2: Write local runner script**

```python
# scripts/run_local.py
"""Single-machine FL simulation launcher."""

import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfl_hcare.fl.server import run_simulation
from pfl_hcare.metrics.collector import MetricsCollector


def main():
    parser = argparse.ArgumentParser(description="Run PFL-HCare FL simulation")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--method", default="pfl_hcare", choices=["fedavg", "fedprox", "per_fedavg", "pfedme", "pfl_hcare"])
    parser.add_argument("--rounds", type=int, default=None, help="Override num_rounds")
    parser.add_argument("--clients", type=int, default=None, help="Override num_clients")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.rounds:
        config["training"]["num_rounds"] = args.rounds
    if args.clients:
        config["training"]["num_clients"] = args.clients

    mc = MetricsCollector()
    mc.on_round(lambda data: print(f"Round {data['round']}: {data['metrics'].get('global_accuracy', 'N/A')}"))

    print(f"Starting {args.method} simulation with {config['training']['num_clients']} clients for {config['training']['num_rounds']} rounds")
    run_simulation(config, method=args.method, metrics_collector=mc)

    print(f"\nCompleted. {len(mc.get_history())} rounds recorded.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add pfl_hcare/fl/server.py scripts/run_local.py
git commit -m "feat: add FL server launcher and local simulation runner"
```

---

## Phase 7: FastAPI Backend

### Task 16: Database and FastAPI app

**Files:**
- Create: `server/db.py`
- Create: `server/main.py`

- [ ] **Step 1: Write SQLite persistence layer**

```python
# server/db.py
"""SQLite persistence for FL run results."""

import json
import aiosqlite
from typing import Optional


DB_PATH = "results.db"


async def init_db(db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                method TEXT NOT NULL,
                dataset TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS round_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                round_num INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        await db.commit()


async def create_run(method: str, dataset: str, config: dict, db_path: str = DB_PATH) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO runs (method, dataset, config) VALUES (?, ?, ?)",
            (method, dataset, json.dumps(config)),
        )
        await db.commit()
        return cursor.lastrowid


async def save_round(run_id: int, round_num: int, metrics: dict, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO round_metrics (run_id, round_num, metrics_json) VALUES (?, ?, ?)",
            (run_id, round_num, json.dumps(metrics)),
        )
        await db.commit()


async def get_run_metrics(run_id: int, db_path: str = DB_PATH) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT round_num, metrics_json FROM round_metrics WHERE run_id = ? ORDER BY round_num",
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [{"round": row["round_num"], "metrics": json.loads(row["metrics_json"])} for row in rows]


async def list_runs(db_path: str = DB_PATH) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT id, method, dataset, config, created_at FROM runs ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [
            {"id": row["id"], "method": row["method"], "dataset": row["dataset"],
             "config": json.loads(row["config"]), "created_at": row["created_at"]}
            for row in rows
        ]
```

- [ ] **Step 2: Write FastAPI main app**

```python
# server/main.py
"""FastAPI application for PFL-HCare dashboard backend."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.db import init_db
from server.routes.training import router as training_router
from server.routes.metrics import router as metrics_router
from server.routes.datasets import router as datasets_router
from server.ws.live import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="PFL-HCare Dashboard API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(metrics_router, prefix="/api/metrics", tags=["metrics"])
app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
app.include_router(ws_router)
```

- [ ] **Step 3: Commit**

```bash
git add server/db.py server/main.py
git commit -m "feat: add FastAPI app with SQLite persistence"
```

### Task 17: API routes and WebSocket

**Files:**
- Create: `server/routes/training.py`
- Create: `server/routes/metrics.py`
- Create: `server/routes/datasets.py`
- Create: `server/ws/live.py`
- Create: `server/orchestrator.py`

- [ ] **Step 1: Write training routes**

```python
# server/routes/training.py
"""Training start/stop/status endpoints."""

import asyncio
import yaml
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Global state for current training run
_current_task: Optional[asyncio.Task] = None
_current_status = {"status": "idle", "method": None, "round": 0, "total_rounds": 0}


class TrainingConfig(BaseModel):
    method: str = "pfl_hcare"
    dataset: str = "har"
    num_clients: int = 10
    num_rounds: int = 200
    noise_multiplier: float = 0.5
    k_bits: int = 8
    partition_alpha: float = 0.5
    learning_rate: float = 0.01


@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    global _current_task, _current_status

    if _current_status["status"] == "running":
        return {"error": "Training already running"}

    _current_status = {
        "status": "running",
        "method": config.method,
        "round": 0,
        "total_rounds": config.num_rounds,
    }

    background_tasks.add_task(_run_training, config)
    return {"status": "started", "config": config.model_dump()}


@router.post("/stop")
async def stop_training():
    global _current_task, _current_status
    if _current_task and not _current_task.done():
        _current_task.cancel()
    _current_status["status"] = "stopped"
    return {"status": "stopped"}


@router.get("/status")
async def get_status():
    return _current_status


async def _run_training(config: TrainingConfig):
    global _current_status
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from pfl_hcare.fl.server import run_simulation
    from pfl_hcare.metrics.collector import MetricsCollector
    from server.ws.live import broadcast_metric
    from server.db import create_run, save_round

    # Load base config and override
    with open("configs/default.yaml") as f:
        base_config = yaml.safe_load(f)

    base_config["training"]["num_clients"] = config.num_clients
    base_config["training"]["num_rounds"] = config.num_rounds
    base_config["training"]["learning_rate"] = config.learning_rate
    base_config["privacy"]["noise_multiplier"] = config.noise_multiplier
    base_config["quantization"]["k_bits"] = config.k_bits
    base_config["dataset"]["name"] = config.dataset
    base_config["dataset"]["partition_alpha"] = config.partition_alpha

    run_id = await create_run(config.method, config.dataset, base_config)

    mc = MetricsCollector()

    async def on_round(data):
        _current_status["round"] = data["round"]
        await save_round(run_id, data["round"], data["metrics"])
        await broadcast_metric(data)

    mc.on_round(lambda data: asyncio.get_event_loop().create_task(on_round(data)))

    try:
        run_simulation(base_config, method=config.method, metrics_collector=mc)
        _current_status["status"] = "completed"
    except Exception as e:
        _current_status["status"] = f"error: {str(e)}"
```

- [ ] **Step 2: Write metrics routes**

```python
# server/routes/metrics.py
"""Historical metrics endpoints."""

from fastapi import APIRouter
from server.db import get_run_metrics, list_runs

router = APIRouter()


@router.get("/runs")
async def get_runs():
    return await list_runs()


@router.get("/{run_id}")
async def get_metrics(run_id: int):
    return await get_run_metrics(run_id)
```

- [ ] **Step 3: Write datasets routes**

```python
# server/routes/datasets.py
"""Dataset info and partition preview endpoints."""

import os
from fastapi import APIRouter
from pydantic import BaseModel
from data.partition import DirichletPartitioner
from data.synthetic_generator import SyntheticMedicalGenerator

router = APIRouter()


class PartitionRequest(BaseModel):
    num_clients: int = 10
    alpha: float = 0.5
    dataset: str = "har"
    seed: int = 42


@router.get("/info")
async def dataset_info():
    datasets = {
        "har": {"available": True, "name": "UCI HAR", "samples": 10299, "features": 561, "classes": 6},
        "mimic": {
            "available": os.path.exists("./datasets/mimic3/processed.csv"),
            "name": "MIMIC-III",
            "fallback": "synthetic",
        },
    }
    # Check which medical tier is active
    if os.path.exists("./datasets/mimic3/processed.csv"):
        datasets["mimic"]["active_tier"] = "mimic3_full"
    elif os.path.exists("./datasets/mimic3_demo/processed.csv"):
        datasets["mimic"]["active_tier"] = "mimic3_demo"
    elif os.path.exists("./datasets/heart_disease/processed.cleveland.data"):
        datasets["mimic"]["active_tier"] = "heart_disease"
    else:
        datasets["mimic"]["active_tier"] = "synthetic"

    return datasets


@router.post("/partition-preview")
async def partition_preview(req: PartitionRequest):
    # Use synthetic data for preview to avoid loading full dataset
    gen = SyntheticMedicalGenerator(n_samples=500, seed=req.seed)
    X, y = gen.generate()

    import torch
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))

    partitioner = DirichletPartitioner(req.num_clients, req.alpha, req.seed)
    partitions = partitioner.partition(ds)
    labels = [ds[i][1].item() for i in range(len(ds))]
    n_classes = len(set(labels))

    summary = partitioner.get_distribution_summary(partitions, labels, n_classes)
    score = partitioner.heterogeneity_score(partitions, labels, n_classes)

    return {"partitions": summary, "heterogeneity_score": score}
```

- [ ] **Step 4: Write WebSocket live streaming**

```python
# server/ws/live.py
"""WebSocket endpoint for live metric streaming."""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

_connected_clients: list[WebSocket] = []


@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connected_clients.append(websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connected_clients.remove(websocket)


async def broadcast_metric(data: dict) -> None:
    """Broadcast a metric update to all connected dashboard clients."""
    message = json.dumps(data)
    disconnected = []
    for ws in _connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _connected_clients.remove(ws)
```

- [ ] **Step 5: Write orchestrator for comparison runs**

```python
# server/orchestrator.py
"""Orchestrates sequential comparison runs across multiple FL methods."""

import yaml
from pfl_hcare.fl.server import run_simulation
from pfl_hcare.metrics.collector import MetricsCollector


METHODS = ["fedavg", "fedprox", "per_fedavg", "pfedme", "pfl_hcare"]


async def run_comparison(config_path: str = "configs/comparison.yaml", on_round=None):
    """Run all methods sequentially and collect results."""
    with open(config_path) as f:
        comp_config = yaml.safe_load(f)

    with open("configs/default.yaml") as f:
        base_config = yaml.safe_load(f)

    methods = comp_config["comparison_run"]["methods"]
    rounds = comp_config["comparison_run"]["rounds"]
    clients = comp_config["comparison_run"]["clients"]
    seeds = comp_config["comparison_run"]["seeds"]

    base_config["training"]["num_rounds"] = rounds
    base_config["training"]["num_clients"] = clients

    all_results = {}

    for method in methods:
        method_results = []
        for seed in seeds:
            base_config["training"]["seed"] = seed
            mc = MetricsCollector()
            if on_round:
                mc.on_round(on_round)

            run_simulation(base_config, method=method, metrics_collector=mc)
            method_results.append(mc.get_history())

        all_results[method] = method_results

    return all_results
```

- [ ] **Step 6: Commit**

```bash
git add server/routes/ server/ws/ server/orchestrator.py
git commit -m "feat: add API routes, WebSocket streaming, and comparison orchestrator"
```

---

## Phase 8: React Dashboard

### Task 18: Scaffold React project

**Files:**
- Create: `client/` directory with Vite + React + TypeScript + Tailwind

- [ ] **Step 1: Initialize Vite React project**

Run:
```bash
cd /Users/tisharunwal/Desktop/Personalized_Federated_Learning
npm create vite@latest client -- --template react-ts
cd client
npm install
npm install react-router-dom recharts d3 @types/d3 framer-motion tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Configure Tailwind**

Replace `client/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
```

Replace `client/src/index.css`:

```css
@import "tailwindcss";

@theme {
  --color-primary: #3b82f6;
  --color-accent: #06b6d4;
  --font-sans: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body {
  @apply bg-slate-900 text-slate-100;
}
```

- [ ] **Step 3: Commit**

```bash
git add client/
git commit -m "feat: scaffold React dashboard with Vite + Tailwind"
```

### Task 19: TypeScript types and WebSocket hook

**Files:**
- Create: `client/src/types/metrics.ts`
- Create: `client/src/hooks/useWebSocket.ts`
- Create: `client/src/hooks/useTrainingState.ts`
- Create: `client/src/utils/format.ts`

- [ ] **Step 1: Write TypeScript interfaces**

```typescript
// client/src/types/metrics.ts
export interface RoundMetrics {
  global_accuracy?: number;
  global_loss?: number;
  epsilon_spent?: number;
  bytes_original?: number;
  bytes_quantized?: number;
  compression_ratio?: number;
  savings_percent?: number;
  clients_selected?: number[];
  per_client_accuracy?: number[];
  per_client_gradient_norm?: number[];
  encryption_latency_ms?: number;
  encryption_status?: string;
  round_time_ms?: number;
}

export interface RoundUpdate {
  type: 'round_update';
  round: number;
  method: string;
  metrics: RoundMetrics;
}

export interface TrainingConfig {
  method: string;
  dataset: string;
  num_clients: number;
  num_rounds: number;
  noise_multiplier: number;
  k_bits: number;
  partition_alpha: number;
  learning_rate: number;
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'completed' | 'stopped' | string;
  method: string | null;
  round: number;
  total_rounds: number;
}

export interface PartitionSummary {
  client_id: number;
  n_samples: number;
  class_distribution: number[];
}

export type MethodName = 'fedavg' | 'fedprox' | 'per_fedavg' | 'pfedme' | 'pfl_hcare';

export const METHOD_COLORS: Record<MethodName, string> = {
  pfl_hcare: '#3b82f6',
  pfedme: '#a855f7',
  per_fedavg: '#fb923c',
  fedprox: '#9ca3af',
  fedavg: '#6b7280',
};

export const METHOD_LABELS: Record<MethodName, string> = {
  pfl_hcare: 'PFL-HCare',
  pfedme: 'pFedMe',
  per_fedavg: 'Per-FedAvg',
  fedprox: 'FedProx',
  fedavg: 'FedAvg',
};
```

- [ ] **Step 2: Write WebSocket hook**

```typescript
// client/src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback, useState } from 'react';
import { RoundUpdate } from '../types/metrics';

export function useWebSocket(url: string = '/ws/live') {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<RoundUpdate | null>(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}${url}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      // Reconnect after 2 seconds
      setTimeout(connect, 2000);
    };
    ws.onmessage = (event) => {
      const data: RoundUpdate = JSON.parse(event.data);
      setLastMessage(data);
    };

    wsRef.current = ws;
  }, [url]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { connected, lastMessage };
}
```

- [ ] **Step 3: Write training state hook**

```typescript
// client/src/hooks/useTrainingState.ts
import { useState, useEffect, useCallback } from 'react';
import { RoundUpdate, RoundMetrics, TrainingConfig, TrainingStatus } from '../types/metrics';
import { useWebSocket } from './useWebSocket';

interface TrainingState {
  rounds: RoundUpdate[];
  status: TrainingStatus;
  connected: boolean;
  startTraining: (config: TrainingConfig) => Promise<void>;
  stopTraining: () => Promise<void>;
}

export function useTrainingState(): TrainingState {
  const { connected, lastMessage } = useWebSocket();
  const [rounds, setRounds] = useState<RoundUpdate[]>([]);
  const [status, setStatus] = useState<TrainingStatus>({
    status: 'idle', method: null, round: 0, total_rounds: 0,
  });

  useEffect(() => {
    if (lastMessage) {
      setRounds(prev => [...prev, lastMessage]);
      setStatus(prev => ({ ...prev, round: lastMessage.round }));
    }
  }, [lastMessage]);

  const startTraining = useCallback(async (config: TrainingConfig) => {
    setRounds([]);
    const res = await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    const data = await res.json();
    setStatus({ status: 'running', method: config.method, round: 0, total_rounds: config.num_rounds });
  }, []);

  const stopTraining = useCallback(async () => {
    await fetch('/api/training/stop', { method: 'POST' });
    setStatus(prev => ({ ...prev, status: 'stopped' }));
  }, []);

  // Poll status periodically
  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch('/api/training/status');
      const data: TrainingStatus = await res.json();
      setStatus(data);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return { rounds, status, connected, startTraining, stopTraining };
}
```

- [ ] **Step 4: Write format utilities**

```typescript
// client/src/utils/format.ts
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatEpsilon(epsilon: number): string {
  return `ε = ${epsilon.toFixed(2)}`;
}

export function accuracyColor(accuracy: number): string {
  if (accuracy >= 0.9) return '#22c55e';  // green
  if (accuracy >= 0.7) return '#eab308';  // yellow
  return '#ef4444';  // red
}
```

- [ ] **Step 5: Commit**

```bash
git add client/src/types/ client/src/hooks/ client/src/utils/
git commit -m "feat: add TypeScript types, WebSocket hook, and training state management"
```

### Task 20: Dashboard layout (Sidebar, Header, ControlRibbon)

**Files:**
- Create: `client/src/components/layout/Sidebar.tsx`
- Create: `client/src/components/layout/Header.tsx`
- Create: `client/src/components/layout/ControlRibbon.tsx`
- Modify: `client/src/App.tsx`

- [ ] **Step 1: Write Sidebar**

```tsx
// client/src/components/layout/Sidebar.tsx
import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/', label: 'Overview', icon: '◉' },
  { path: '/convergence', label: 'Convergence', icon: '◎' },
  { path: '/privacy', label: 'Privacy', icon: '◎' },
  { path: '/communication', label: 'Communication', icon: '◎' },
  { path: '/comparison', label: 'Comparison', icon: '◎' },
];

export function Sidebar() {
  return (
    <nav className="w-16 bg-slate-800 flex flex-col items-center py-4 gap-2 border-r border-slate-700">
      {navItems.map((item) => (
        <NavLink
          key={item.path}
          to={item.path}
          className={({ isActive }) =>
            `w-10 h-10 rounded-lg flex items-center justify-center text-sm transition-colors ${
              isActive ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-700'
            }`
          }
          title={item.label}
        >
          {item.icon}
        </NavLink>
      ))}
    </nav>
  );
}
```

- [ ] **Step 2: Write Header**

```tsx
// client/src/components/layout/Header.tsx
import { TrainingStatus } from '../../types/metrics';

interface HeaderProps {
  status: TrainingStatus;
  connected: boolean;
}

export function Header({ status, connected }: HeaderProps) {
  const progress = status.total_rounds > 0
    ? (status.round / status.total_rounds) * 100
    : 0;

  return (
    <header className="h-14 bg-slate-800 border-b border-slate-700 flex items-center px-6 gap-4">
      <h1 className="text-lg font-semibold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
        PFL-HCare Dashboard
      </h1>

      <div className="flex items-center gap-2 ml-auto">
        <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
        <span className="text-xs text-slate-400">
          {connected ? 'Live' : 'Disconnected'}
        </span>
      </div>

      {status.status === 'running' && (
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-300">
            Round {status.round}/{status.total_rounds}
          </span>
          <div className="w-48 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </header>
  );
}
```

- [ ] **Step 3: Write ControlRibbon**

```tsx
// client/src/components/layout/ControlRibbon.tsx
import { useState } from 'react';
import { TrainingConfig, MethodName, METHOD_LABELS } from '../../types/metrics';

interface ControlRibbonProps {
  onStart: (config: TrainingConfig) => void;
  onStop: () => void;
  isRunning: boolean;
}

export function ControlRibbon({ onStart, onStop, isRunning }: ControlRibbonProps) {
  const [config, setConfig] = useState<TrainingConfig>({
    method: 'pfl_hcare',
    dataset: 'har',
    num_clients: 10,
    num_rounds: 50,
    noise_multiplier: 0.5,
    k_bits: 8,
    partition_alpha: 0.5,
    learning_rate: 0.01,
  });

  const update = (key: keyof TrainingConfig, value: string | number) =>
    setConfig(prev => ({ ...prev, [key]: value }));

  return (
    <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex flex-wrap items-center gap-4 text-sm">
      <label className="flex items-center gap-2">
        <span className="text-slate-400">Dataset</span>
        <select
          value={config.dataset}
          onChange={e => update('dataset', e.target.value)}
          className="bg-slate-700 rounded px-2 py-1 text-slate-200"
          disabled={isRunning}
        >
          <option value="har">UCI HAR</option>
          <option value="mimic">Medical</option>
        </select>
      </label>

      <label className="flex items-center gap-2">
        <span className="text-slate-400">Method</span>
        <select
          value={config.method}
          onChange={e => update('method', e.target.value)}
          className="bg-slate-700 rounded px-2 py-1 text-slate-200"
          disabled={isRunning}
        >
          {(Object.keys(METHOD_LABELS) as MethodName[]).map(m => (
            <option key={m} value={m}>{METHOD_LABELS[m]}</option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2">
        <span className="text-slate-400">Clients</span>
        <input
          type="number" min={2} max={20} value={config.num_clients}
          onChange={e => update('num_clients', parseInt(e.target.value))}
          className="bg-slate-700 rounded px-2 py-1 w-16 text-slate-200"
          disabled={isRunning}
        />
      </label>

      <label className="flex items-center gap-2">
        <span className="text-slate-400">Rounds</span>
        <input
          type="number" min={10} max={500} value={config.num_rounds}
          onChange={e => update('num_rounds', parseInt(e.target.value))}
          className="bg-slate-700 rounded px-2 py-1 w-20 text-slate-200"
          disabled={isRunning}
        />
      </label>

      <label className="flex items-center gap-2">
        <span className="text-slate-400">Noise (sigma)</span>
        <input
          type="range" min={0.1} max={2.0} step={0.1} value={config.noise_multiplier}
          onChange={e => update('noise_multiplier', parseFloat(e.target.value))}
          className="w-20"
          disabled={isRunning}
        />
        <span className="text-slate-300 font-mono w-8">{config.noise_multiplier}</span>
      </label>

      <label className="flex items-center gap-2">
        <span className="text-slate-400">k-bits</span>
        <select
          value={config.k_bits}
          onChange={e => update('k_bits', parseInt(e.target.value))}
          className="bg-slate-700 rounded px-2 py-1 text-slate-200"
          disabled={isRunning}
        >
          {[2, 4, 8, 16].map(k => <option key={k} value={k}>{k}</option>)}
        </select>
      </label>

      <div className="ml-auto flex gap-2">
        {!isRunning ? (
          <button
            onClick={() => onStart(config)}
            className="bg-green-600 hover:bg-green-500 text-white px-4 py-1.5 rounded-lg font-medium transition-colors"
          >
            Start Training
          </button>
        ) : (
          <button
            onClick={onStop}
            className="bg-red-600 hover:bg-red-500 text-white px-4 py-1.5 rounded-lg font-medium transition-colors"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Write App.tsx with routing**

```tsx
// client/src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { ControlRibbon } from './components/layout/ControlRibbon';
import { useTrainingState } from './hooks/useTrainingState';
import { OverviewView } from './components/views/OverviewView';
import { ConvergenceView } from './components/views/ConvergenceView';
import { PrivacyView } from './components/views/PrivacyView';
import { CommunicationView } from './components/views/CommunicationView';
import { ComparisonView } from './components/views/ComparisonView';

export default function App() {
  const { rounds, status, connected, startTraining, stopTraining } = useTrainingState();
  const isRunning = status.status === 'running';

  return (
    <BrowserRouter>
      <div className="h-screen flex">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header status={status} connected={connected} />
          <ControlRibbon onStart={startTraining} onStop={stopTraining} isRunning={isRunning} />
          <main className="flex-1 overflow-auto p-6">
            <Routes>
              <Route path="/" element={<OverviewView rounds={rounds} status={status} />} />
              <Route path="/convergence" element={<ConvergenceView rounds={rounds} />} />
              <Route path="/privacy" element={<PrivacyView rounds={rounds} />} />
              <Route path="/communication" element={<CommunicationView rounds={rounds} />} />
              <Route path="/comparison" element={<ComparisonView rounds={rounds} />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add client/src/components/layout/ client/src/App.tsx
git commit -m "feat: add dashboard layout with sidebar, header, and control ribbon"
```

### Task 21: Dashboard views (Overview, Convergence, Privacy, Communication, Comparison)

**Files:**
- Create: `client/src/components/views/OverviewView.tsx`
- Create: `client/src/components/views/ConvergenceView.tsx`
- Create: `client/src/components/views/PrivacyView.tsx`
- Create: `client/src/components/views/CommunicationView.tsx`
- Create: `client/src/components/views/ComparisonView.tsx`
- Create: `client/src/components/widgets/KpiCard.tsx`
- Create: `client/src/components/widgets/ActivityFeed.tsx`
- Create: `client/src/components/charts/ConvergenceChart.tsx`
- Create: `client/src/components/charts/PrivacyGauge.tsx`
- Create: `client/src/components/charts/BandwidthChart.tsx`
- Create: `client/src/components/charts/HeatmapChart.tsx`
- Create: `client/src/components/charts/SpeedBars.tsx`

- [ ] **Step 1: Write KpiCard widget**

```tsx
// client/src/components/widgets/KpiCard.tsx
import { motion } from 'framer-motion';

interface KpiCardProps {
  label: string;
  value: string;
  trend?: number;
  color?: string;
}

export function KpiCard({ label, value, trend, color = '#3b82f6' }: KpiCardProps) {
  return (
    <motion.div
      className="bg-slate-800 rounded-xl p-4 border border-slate-700"
      animate={{ borderColor: trend ? color : '#334155' }}
      transition={{ duration: 0.3 }}
    >
      <p className="text-sm text-slate-400">{label}</p>
      <motion.p
        className="text-2xl font-mono font-bold mt-1"
        style={{ color }}
        key={value}
        initial={{ opacity: 0.5 }}
        animate={{ opacity: 1 }}
      >
        {value}
      </motion.p>
      {trend !== undefined && (
        <p className={`text-xs mt-1 ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {trend >= 0 ? '+' : ''}{trend.toFixed(2)}
        </p>
      )}
    </motion.div>
  );
}
```

- [ ] **Step 2: Write ActivityFeed widget**

```tsx
// client/src/components/widgets/ActivityFeed.tsx
import { RoundUpdate } from '../../types/metrics';

interface ActivityFeedProps {
  rounds: RoundUpdate[];
  maxItems?: number;
}

export function ActivityFeed({ rounds, maxItems = 20 }: ActivityFeedProps) {
  const recent = rounds.slice(-maxItems).reverse();

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 h-full overflow-y-auto">
      <h3 className="text-sm font-semibold text-slate-300 mb-3">Activity Feed</h3>
      <div className="space-y-2">
        {recent.map((r, i) => (
          <div key={i} className="text-xs text-slate-400 flex gap-2">
            <span className="text-blue-400">R{r.round}</span>
            <span>
              acc: {((r.metrics.global_accuracy ?? 0) * 100).toFixed(1)}%
              {r.metrics.epsilon_spent !== undefined && ` | eps: ${r.metrics.epsilon_spent.toFixed(2)}`}
            </span>
          </div>
        ))}
        {recent.length === 0 && (
          <p className="text-xs text-slate-500">No activity yet. Start a training run.</p>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write ConvergenceChart**

```tsx
// client/src/components/charts/ConvergenceChart.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { RoundUpdate, METHOD_COLORS, METHOD_LABELS, MethodName } from '../../types/metrics';

interface ConvergenceChartProps {
  rounds: RoundUpdate[];
}

export function ConvergenceChart({ rounds }: ConvergenceChartProps) {
  const data = rounds.map(r => ({
    round: r.round,
    accuracy: (r.metrics.global_accuracy ?? 0) * 100,
    method: r.method,
  }));

  // Group by method
  const methods = [...new Set(data.map(d => d.method))];
  const chartData: Record<string, number | undefined>[] = [];
  const roundNums = [...new Set(data.map(d => d.round))].sort((a, b) => a - b);

  for (const round of roundNums) {
    const entry: Record<string, number | undefined> = { round };
    for (const method of methods) {
      const point = data.find(d => d.round === round && d.method === method);
      entry[method] = point?.accuracy;
    }
    chartData.push(entry);
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="round" stroke="#94a3b8" fontSize={12} />
        <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} unit="%" />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
          labelStyle={{ color: '#e2e8f0' }}
        />
        <Legend />
        {methods.map(method => (
          <Line
            key={method}
            type="monotone"
            dataKey={method}
            name={METHOD_LABELS[method as MethodName] ?? method}
            stroke={METHOD_COLORS[method as MethodName] ?? '#888'}
            strokeWidth={method === 'pfl_hcare' ? 3 : 1.5}
            strokeDasharray={method === 'pfl_hcare' ? undefined : '5 5'}
            dot={false}
            animationDuration={300}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 4: Write PrivacyGauge**

```tsx
// client/src/components/charts/PrivacyGauge.tsx
interface PrivacyGaugeProps {
  epsilonSpent: number;
  targetEpsilon: number;
}

export function PrivacyGauge({ epsilonSpent, targetEpsilon }: PrivacyGaugeProps) {
  const percent = Math.min((epsilonSpent / targetEpsilon) * 100, 100);
  const color = percent < 50 ? '#22c55e' : percent < 80 ? '#eab308' : '#ef4444';

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 text-center">
      <h3 className="text-sm text-slate-400 mb-4">Privacy Budget</h3>
      <div className="relative w-32 h-32 mx-auto">
        <svg viewBox="0 0 36 36" className="w-full h-full transform -rotate-90">
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none" stroke="#334155" strokeWidth="3"
          />
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none" stroke={color} strokeWidth="3"
            strokeDasharray={`${percent}, 100`}
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-xl font-mono font-bold" style={{ color }}>
            {epsilonSpent.toFixed(1)}
          </span>
          <span className="text-xs text-slate-400">/ {targetEpsilon}</span>
        </div>
      </div>
      <p className="text-xs text-slate-400 mt-2">{percent.toFixed(0)}% budget used</p>
    </div>
  );
}
```

- [ ] **Step 5: Write BandwidthChart**

```tsx
// client/src/components/charts/BandwidthChart.tsx
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { RoundUpdate } from '../../types/metrics';

interface BandwidthChartProps {
  rounds: RoundUpdate[];
}

export function BandwidthChart({ rounds }: BandwidthChartProps) {
  const data = rounds
    .filter(r => r.metrics.bytes_original)
    .map(r => ({
      round: r.round,
      original: (r.metrics.bytes_original ?? 0) / 1024,
      quantized: (r.metrics.bytes_quantized ?? 0) / 1024,
    }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="round" stroke="#94a3b8" fontSize={12} />
        <YAxis stroke="#94a3b8" fontSize={12} unit=" KB" />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
        />
        <Legend />
        <Bar dataKey="original" name="Original" fill="#6b7280" />
        <Bar dataKey="quantized" name="Quantized" fill="#3b82f6" />
      </BarChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 6: Write all 5 view components**

```tsx
// client/src/components/views/OverviewView.tsx
import { RoundUpdate, TrainingStatus } from '../../types/metrics';
import { KpiCard } from '../widgets/KpiCard';
import { ActivityFeed } from '../widgets/ActivityFeed';
import { formatPercent, formatBytes, formatEpsilon } from '../../utils/format';

interface Props { rounds: RoundUpdate[]; status: TrainingStatus; }

export function OverviewView({ rounds, status }: Props) {
  const latest = rounds[rounds.length - 1];
  const prev = rounds.length > 1 ? rounds[rounds.length - 2] : null;

  const acc = latest?.metrics.global_accuracy ?? 0;
  const loss = latest?.metrics.global_loss ?? 0;
  const eps = latest?.metrics.epsilon_spent ?? 0;
  const saved = latest?.metrics.savings_percent ?? 0;

  const accTrend = prev ? acc - (prev.metrics.global_accuracy ?? 0) : undefined;
  const lossTrend = prev ? loss - (prev.metrics.global_loss ?? 0) : undefined;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        <KpiCard label="Accuracy" value={formatPercent(acc)} trend={accTrend} color="#22c55e" />
        <KpiCard label="Loss" value={loss.toFixed(4)} trend={lossTrend} color="#ef4444" />
        <KpiCard label="Privacy Budget" value={formatEpsilon(eps)} color="#a855f7" />
        <KpiCard label="Bandwidth Saved" value={`${saved.toFixed(1)}%`} color="#06b6d4" />
      </div>
      <div className="grid grid-cols-2 gap-6 h-96">
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex items-center justify-center">
          <p className="text-slate-500">Network Topology (D3) — coming in next iteration</p>
        </div>
        <ActivityFeed rounds={rounds} />
      </div>
    </div>
  );
}
```

```tsx
// client/src/components/views/ConvergenceView.tsx
import { RoundUpdate } from '../../types/metrics';
import { ConvergenceChart } from '../charts/ConvergenceChart';

interface Props { rounds: RoundUpdate[]; }

export function ConvergenceView({ rounds }: Props) {
  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Convergence & Accuracy</h2>
      <div className="h-[500px] bg-slate-800 rounded-xl border border-slate-700 p-4">
        <ConvergenceChart rounds={rounds} />
      </div>
    </div>
  );
}
```

```tsx
// client/src/components/views/PrivacyView.tsx
import { RoundUpdate } from '../../types/metrics';
import { PrivacyGauge } from '../charts/PrivacyGauge';

interface Props { rounds: RoundUpdate[]; }

export function PrivacyView({ rounds }: Props) {
  const latest = rounds[rounds.length - 1];
  const eps = latest?.metrics.epsilon_spent ?? 0;
  const encStatus = latest?.metrics.encryption_status ?? 'idle';
  const encLatency = latest?.metrics.encryption_latency_ms ?? 0;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Privacy & Security</h2>
      <div className="grid grid-cols-3 gap-6">
        <PrivacyGauge epsilonSpent={eps} targetEpsilon={10.0} />
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 col-span-2">
          <h3 className="text-sm text-slate-400 mb-4">Encryption Status</h3>
          <div className="flex items-center gap-4">
            <span className="text-4xl">{encStatus === 'encrypted' ? '🔒' : '🔓'}</span>
            <div>
              <p className="text-lg font-mono text-slate-200">{encStatus.toUpperCase()}</p>
              <p className="text-sm text-slate-400">Latency: {encLatency}ms</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
```

```tsx
// client/src/components/views/CommunicationView.tsx
import { RoundUpdate } from '../../types/metrics';
import { BandwidthChart } from '../charts/BandwidthChart';
import { formatBytes } from '../../utils/format';

interface Props { rounds: RoundUpdate[]; }

export function CommunicationView({ rounds }: Props) {
  const totalOriginal = rounds.reduce((s, r) => s + (r.metrics.bytes_original ?? 0), 0);
  const totalQuantized = rounds.reduce((s, r) => s + (r.metrics.bytes_quantized ?? 0), 0);
  const totalSaved = totalOriginal - totalQuantized;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Communication & Scalability</h2>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 text-center">
          <p className="text-sm text-slate-400">Total Original</p>
          <p className="text-xl font-mono text-slate-200">{formatBytes(totalOriginal)}</p>
        </div>
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 text-center">
          <p className="text-sm text-slate-400">Total Quantized</p>
          <p className="text-xl font-mono text-blue-400">{formatBytes(totalQuantized)}</p>
        </div>
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 text-center">
          <p className="text-sm text-slate-400">Total Saved</p>
          <p className="text-xl font-mono text-green-400">{formatBytes(totalSaved)}</p>
        </div>
      </div>
      <div className="h-[400px] bg-slate-800 rounded-xl border border-slate-700 p-4">
        <BandwidthChart rounds={rounds} />
      </div>
    </div>
  );
}
```

```tsx
// client/src/components/views/ComparisonView.tsx
import { RoundUpdate, METHOD_LABELS, MethodName } from '../../types/metrics';

interface Props { rounds: RoundUpdate[]; }

export function ComparisonView({ rounds }: Props) {
  // Group rounds by method, get final accuracy
  const methodResults: Record<string, { accuracy: number; rounds: number }> = {};
  for (const r of rounds) {
    methodResults[r.method] = {
      accuracy: r.metrics.global_accuracy ?? 0,
      rounds: r.round,
    };
  }

  const features: Record<string, Record<string, boolean>> = {
    fedavg:      { personalized: false, dp: false, secure_agg: false, quantization: false, adaptive: false },
    fedprox:     { personalized: false, dp: false, secure_agg: false, quantization: false, adaptive: false },
    per_fedavg:  { personalized: true,  dp: false, secure_agg: false, quantization: false, adaptive: false },
    pfedme:      { personalized: true,  dp: true,  secure_agg: false, quantization: false, adaptive: false },
    pfl_hcare:   { personalized: true,  dp: true,  secure_agg: true,  quantization: true,  adaptive: true  },
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Experiment Log & Comparison</h2>

      {/* Feature Matrix */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="px-4 py-3 text-left text-slate-400">Method</th>
              <th className="px-4 py-3 text-center text-slate-400">Personalized</th>
              <th className="px-4 py-3 text-center text-slate-400">Diff. Privacy</th>
              <th className="px-4 py-3 text-center text-slate-400">Secure Agg.</th>
              <th className="px-4 py-3 text-center text-slate-400">Quantization</th>
              <th className="px-4 py-3 text-center text-slate-400">Adaptive Sel.</th>
              <th className="px-4 py-3 text-center text-slate-400">Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {(Object.keys(features) as MethodName[]).map(method => (
              <tr
                key={method}
                className={`border-b border-slate-700/50 ${method === 'pfl_hcare' ? 'bg-blue-900/20' : ''}`}
              >
                <td className="px-4 py-3 font-medium">{METHOD_LABELS[method]}</td>
                {Object.values(features[method]).map((v, i) => (
                  <td key={i} className="px-4 py-3 text-center">
                    {v ? <span className="text-green-400">✓</span> : <span className="text-slate-600">✗</span>}
                  </td>
                ))}
                <td className="px-4 py-3 text-center font-mono">
                  {methodResults[method]
                    ? `${(methodResults[method].accuracy * 100).toFixed(1)}%`
                    : '—'
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Commit**

```bash
git add client/src/components/
git commit -m "feat: add all dashboard views, charts, and widgets"
```

---

## Phase 9: Docker Setup

### Task 22: Docker configuration

**Files:**
- Create: `docker/Dockerfile.fl-server`
- Create: `docker/Dockerfile.fl-client`
- Create: `docker/Dockerfile.api`
- Create: `docker/Dockerfile.dashboard`
- Create: `docker/docker-compose.yml`

- [ ] **Step 1: Write Dockerfiles**

```dockerfile
# docker/Dockerfile.fl-server
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pfl_hcare/ pfl_hcare/
COPY data/ data/
COPY configs/ configs/
COPY scripts/ scripts/
CMD ["python", "scripts/run_local.py", "--method", "pfl_hcare"]
```

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pfl_hcare/ pfl_hcare/
COPY data/ data/
COPY server/ server/
COPY configs/ configs/
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# docker/Dockerfile.dashboard
FROM node:20-slim AS build
WORKDIR /app
COPY client/package*.json ./
RUN npm ci
COPY client/ .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
```

- [ ] **Step 2: Write docker-compose.yml**

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ../configs:/app/configs
      - ../datasets:/app/datasets
    networks:
      - pfl-network

  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "3000:3000"
    depends_on:
      - api
    networks:
      - pfl-network

networks:
  pfl-network:
    driver: bridge
```

- [ ] **Step 3: Write nginx config**

```nginx
# docker/nginx.conf
server {
    listen 3000;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://api:8000/api/;
    }

    location /ws/ {
        proxy_pass http://api:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] **Step 4: Commit**

```bash
git add docker/
git commit -m "feat: add Docker setup with compose for multi-container simulation"
```

---

## Phase 10: Integration & Polish

### Task 23: README and final integration

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# PFL-HCare: Personalized Federated Learning for IoT-Driven Smart Healthcare

Implementation of the PFL-HCare framework from the ICICI-2025 paper, featuring a real-time web dashboard for demonstration and comparison of federated learning methods.

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- (Optional) Docker Desktop

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download datasets
python scripts/download_data.py

# Install dashboard dependencies
cd client && npm install && cd ..
```

### Run (Local Mode)

```bash
# Terminal 1: Start FastAPI backend
uvicorn server.main:app --reload --port 8000

# Terminal 2: Start React dashboard
cd client && npm run dev

# Terminal 3: Run FL simulation
python scripts/run_local.py --method pfl_hcare --rounds 50 --clients 5
```

Open http://localhost:5173 to view the dashboard.

### Run (Docker Mode)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Open http://localhost:3000 to view the dashboard.

## Methods Compared

| Method | Personalized | Diff. Privacy | Secure Agg. | Quantization | Adaptive Selection |
|--------|:---:|:---:|:---:|:---:|:---:|
| FedAvg | - | - | - | - | - |
| FedProx | ~ | - | - | - | - |
| Per-FedAvg | Yes | ~ | - | - | - |
| pFedMe | Yes | Yes | - | - | - |
| **PFL-HCare** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

## Project Structure

- `pfl_hcare/` — Core ML library (models, FL strategies, MAML, privacy, metrics)
- `data/` — Dataset loaders (UCI HAR, MIMIC-III with fallback chain, synthetic)
- `server/` — FastAPI backend (REST + WebSocket)
- `client/` — React dashboard (Recharts, D3, Framer Motion)
- `docker/` — Docker configuration
- `configs/` — YAML experiment configs
- `scripts/` — CLI tools (run, download, export)

## Configuration

Edit `configs/default.yaml` to change hyperparameters, or use CLI flags:

```bash
python scripts/run_local.py --method pfl_hcare --rounds 200 --clients 10
```

## Paper Reference

> "Personalized Federated Learning for Privacy-Preserving and Scalable IoT-Driven Smart Healthcare"
> ICICI-2025, IEEE Xplore
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```

### Task 24: End-to-end smoke test

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write end-to-end test**

```python
# tests/test_e2e.py
"""End-to-end smoke test: runs a small FL simulation and verifies metrics are collected."""

import yaml
from pfl_hcare.fl.server import run_simulation
from pfl_hcare.metrics.collector import MetricsCollector


def test_e2e_fedavg_smoke():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    config["training"]["num_clients"] = 3
    config["training"]["num_rounds"] = 2
    config["training"]["local_epochs"] = 1
    config["dataset"]["name"] = "mimic"  # Will use synthetic fallback

    mc = MetricsCollector()
    run_simulation(config, method="fedavg", metrics_collector=mc)
    history = mc.get_history()
    assert len(history) > 0


def test_e2e_pfl_hcare_smoke():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    config["training"]["num_clients"] = 3
    config["training"]["num_rounds"] = 2
    config["training"]["local_epochs"] = 1
    config["dataset"]["name"] = "mimic"
    config["maml"]["second_order"] = False
    config["maml"]["inner_steps"] = 1

    mc = MetricsCollector()
    run_simulation(config, method="pfl_hcare", metrics_collector=mc)
    history = mc.get_history()
    assert len(history) > 0
```

- [ ] **Step 2: Run e2e test**

Run: `python -m pytest tests/test_e2e.py -v --timeout=120`
Expected: Both tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end smoke tests for FedAvg and PFL-HCare"
```
