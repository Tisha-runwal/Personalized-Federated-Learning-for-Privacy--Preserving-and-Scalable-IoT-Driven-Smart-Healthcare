"""1D-CNN classifier for Human Activity Recognition."""
import torch
import torch.nn as nn

class HARClassifier(nn.Module):
    """1D-CNN for HAR sensor data. ~52K parameters."""
    def __init__(self, num_classes: int = 6, in_channels: int = 9, accept_flat: bool = False):
        super().__init__()
        self.accept_flat = accept_flat
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.accept_flat and x.dim() == 2:
            x = x[:, :558].reshape(-1, self.in_channels, 62)
        x = self.features(x)
        x = x.mean(dim=2)
        return self.classifier(x)
