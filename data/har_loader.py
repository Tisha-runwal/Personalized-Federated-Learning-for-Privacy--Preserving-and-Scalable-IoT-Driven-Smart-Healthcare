"""UCI HAR Dataset loader for federated learning experiments."""
import os, numpy as np, torch
from torch.utils.data import Dataset
from scripts.download_data import download_har

class HARDataset(Dataset):
    CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

    def __init__(self, root: str = "./datasets", download: bool = True, split: str = "train"):
        assert split in ("train", "test")
        har_dir = os.path.join(root, "UCI HAR Dataset")
        if download and not os.path.exists(har_dir):
            download_har(root)
        split_dir = os.path.join(har_dir, split if split == "train" else "test")
        X_path = os.path.join(split_dir, f"X_{split}.txt")
        y_path = os.path.join(split_dir, f"y_{split}.txt")
        self.X = torch.tensor(np.loadtxt(X_path, dtype=np.float32))
        self.y = torch.tensor(np.loadtxt(y_path, dtype=np.int64).flatten() - 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
