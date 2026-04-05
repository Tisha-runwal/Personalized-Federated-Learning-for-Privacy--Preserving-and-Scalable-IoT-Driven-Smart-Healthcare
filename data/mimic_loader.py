"""Medical dataset loader with 4-tier fallback chain."""
import os, logging, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.synthetic_generator import SyntheticMedicalGenerator

logger = logging.getLogger(__name__)

class MedicalDataset(Dataset):
    def __init__(self, root: str = "./datasets", split: str = "train", test_fraction: float = 0.3, seed: int = 42, n_synthetic: int = 5000):
        assert split in ("train", "test")
        self.root = root
        self.seed = seed
        self.active_tier = "unknown"
        X, y = self._load_with_fallback(n_synthetic)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if split == "train":
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.int64)

    def _load_with_fallback(self, n_synthetic):
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
        # Tier 3: Heart Disease UCI — skip if too small for FL (< 500 samples)
        heart_path = os.path.join(self.root, "heart_disease", "processed.cleveland.data")
        if os.path.exists(heart_path):
            X, y = self._load_heart_disease(heart_path)
            if len(X) >= 500:
                logger.info("Loading Heart Disease UCI (Tier 3) — %d samples", len(X))
                self.active_tier = "heart_disease"
                return X, y
            else:
                logger.info("Heart Disease too small (%d samples) for FL — falling back to synthetic", len(X))
        # Tier 4: Synthetic (always works, configurable size)
        logger.info("Loading Synthetic Medical Data (Tier 4) — %d samples", n_synthetic)
        self.active_tier = "synthetic"
        gen = SyntheticMedicalGenerator(n_samples=n_synthetic, seed=self.seed)
        return gen.generate()

    def _load_csv(self, path):
        df = pd.read_csv(path)
        y = df.iloc[:, -1].values.astype(np.int64)
        X = df.iloc[:, :-1].values.astype(np.float32)
        if X.shape[1] < 13:
            X = np.pad(X, ((0, 0), (0, 13 - X.shape[1])))
        elif X.shape[1] > 13:
            X = X[:, :13]
        return X, y

    def _load_heart_disease(self, path):
        df = pd.read_csv(path, header=None, na_values="?")
        df = df.dropna()
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = (df.iloc[:, -1].values > 0).astype(np.int64)
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
