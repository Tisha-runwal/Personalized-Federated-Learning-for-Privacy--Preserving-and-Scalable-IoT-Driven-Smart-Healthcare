import torch
from data.mimic_loader import MedicalDataset

def test_medical_dataset_fallback_to_synthetic():
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
    assert x.shape[0] == 13

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
