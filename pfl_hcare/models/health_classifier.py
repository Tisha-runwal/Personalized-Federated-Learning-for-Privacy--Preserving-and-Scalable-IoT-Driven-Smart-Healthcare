"""MLP classifier for medical/health prediction tasks."""
import torch
import torch.nn as nn

class HealthClassifier(nn.Module):
    """3-layer MLP for health prediction. ~15K parameters."""
    def __init__(self, input_dim: int = 13, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
