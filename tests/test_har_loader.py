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
    assert x.dim() == 1
    assert x.shape[0] == 561
    assert y.dim() == 0

def test_har_dataset_classes():
    ds = HARDataset(root="./datasets", download=True, split="train")
    labels = set()
    for i in range(min(100, len(ds))):
        _, y = ds[i]
        labels.add(y.item())
    assert labels.issubset({0, 1, 2, 3, 4, 5})

def test_har_dataset_length():
    ds = HARDataset(root="./datasets", download=True, split="train")
    assert len(ds) > 5000
