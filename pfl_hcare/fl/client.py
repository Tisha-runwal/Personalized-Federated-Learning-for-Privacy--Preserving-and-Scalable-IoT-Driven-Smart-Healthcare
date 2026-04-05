"""Flower NumPyClient supporting all 5 FL strategies."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from flwr.client import NumPyClient

from pfl_hcare.privacy.differential_privacy import DPMechanism


class PFLClient(NumPyClient):
    """Federated client supporting fedavg, fedprox, per_fedavg, pfedme, pfl_hcare."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_dataset,
        test_dataset,
        strategy: str = "fedavg",
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: str | None = None,
        # FedProx
        mu: float = 0.01,
        # pFedMe
        lambd: float = 15.0,
        # MAML wrapper (for per_fedavg / pfl_hcare)
        maml_wrapper=None,
        # DP (for pfl_hcare)
        dp_mechanism: DPMechanism | None = None,
    ):
        self.client_id = client_id
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.strategy = strategy
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = mu
        self.lambd = lambd
        self.maml_wrapper = maml_wrapper
        self.dp_mechanism = dp_mechanism

        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        param_keys = list(state_dict.keys())
        # Only update keys that correspond to parameters() (exclude buffers)
        param_names = [n for n, _ in self.model.named_parameters()]
        for name, arr in zip(param_names, parameters):
            state_dict[name] = torch.tensor(arr)
        self.model.load_state_dict(state_dict)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        self.set_parameters(parameters)

        old_params = [p.detach().clone() for p in self.model.parameters()]

        if self.strategy == "fedavg":
            self._train_standard()
        elif self.strategy == "fedprox":
            self._train_fedprox()
        elif self.strategy == "per_fedavg":
            self._train_maml()
        elif self.strategy == "pfedme":
            self._train_pfedme()
        elif self.strategy == "pfl_hcare":
            self._train_maml()
            # Apply DP after training
            if self.dp_mechanism is not None:
                new_params = [p.detach().clone() for p in self.model.parameters()]
                delta_params = [n - o for n, o in zip(new_params, old_params)]
                clipped = self.dp_mechanism.clip_gradients(delta_params)
                n_samples = len(self.train_dataset)
                noisy = self.dp_mechanism.add_noise(clipped, sample_rate=min(self.batch_size / max(n_samples, 1), 1.0))
                # Apply noisy delta back to model
                with torch.no_grad():
                    for p, o, d in zip(self.model.parameters(), old_params, noisy):
                        p.copy_(o + d)
        else:
            self._train_standard()

        new_params = [p.detach().clone() for p in self.model.parameters()]

        # Gradient norm: L2 norm of (new_params - old_params)
        delta = torch.cat([(n - o).flatten() for n, o in zip(new_params, old_params)])
        grad_norm = delta.norm(p=2).item()

        updated_parameters = [p.cpu().numpy() for p in new_params]
        n_samples = len(self.train_dataset)
        metrics = {"grad_norm": grad_norm, "client_id": self.client_id}
        return updated_parameters, n_samples, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = criterion(logits, y)
                total_loss += loss.item() * len(y)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy}

    # ------------------------------------------------------------------
    # Training routines
    # ------------------------------------------------------------------

    def _get_loader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def _train_standard(self) -> None:
        """Standard FedAvg local training."""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = self._get_loader()
        for _ in range(self.local_epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()

    def _train_fedprox(self) -> None:
        """FedProx: adds proximal term mu/2 * ||w - w0||^2."""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = self._get_loader()
        global_params = [p.detach().clone() for p in self.model.parameters()]
        for _ in range(self.local_epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                # Proximal term
                prox = sum(
                    ((p - gp) ** 2).sum()
                    for p, gp in zip(self.model.parameters(), global_params)
                )
                loss = loss + (self.mu / 2.0) * prox
                loss.backward()
                optimizer.step()

    def _train_maml(self) -> None:
        """MAML inner/outer loop personalization (Eq.3-4).

        Splits each batch into support/query sets. Runs MAML inner loop
        adaptation on support set, computes outer loss on query set, and
        backpropagates through the entire computation graph.
        """
        if self.maml_wrapper is None:
            self._train_standard()
            return

        self.model.train()
        loader = self._get_loader()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for _ in range(self.local_epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                if len(X) < 4:
                    continue

                # Split batch into support and query sets
                half = max(2, len(X) // 2)
                support_X, query_X = X[:half], X[half:]
                support_y, query_y = y[:half], y[half:]

                optimizer.zero_grad()
                # MAML outer_loss: inner-loop adapt on support, evaluate on query
                # This computes gradients and assigns them to model.parameters()
                loss = self.maml_wrapper.outer_loss(
                    support_X, support_y, query_X, query_y
                )
                # Clip gradients to prevent MAML-induced explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

    def _train_pfedme(self) -> None:
        """pFedMe: personalized model via Moreau envelope (lambda=15)."""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        loader = self._get_loader()
        # Global params (server params, fixed during local update)
        global_params = [p.detach().clone() for p in self.model.parameters()]
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        for _ in range(self.local_epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                # Moreau envelope regularisation: lambda/2 * ||w - w_global||^2
                moreau = sum(
                    ((p - gp) ** 2).sum()
                    for p, gp in zip(self.model.parameters(), global_params)
                )
                loss = loss + (self.lambd / 2.0) * moreau
                loss.backward()
                optimizer.step()
