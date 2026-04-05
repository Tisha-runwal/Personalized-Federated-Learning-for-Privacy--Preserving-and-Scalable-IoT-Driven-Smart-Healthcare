"""Differential privacy mechanism for federated learning (Eq.5)."""
import math, torch

class DPMechanism:
    def __init__(self, noise_multiplier: float = 0.5, max_grad_norm: float = 1.0, delta: float = 1e-5):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self._steps = 0
        self._sample_rates: list[float] = []

    def clip_gradients(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        flat = torch.cat([p.flatten() for p in params])
        total_norm = torch.norm(flat)
        clip_factor = min(1.0, self.max_grad_norm / (total_norm.item() + 1e-8))
        return [p * clip_factor for p in params]

    def add_noise(self, params: list[torch.Tensor], sample_rate: float = 1.0, track: bool = True) -> list[torch.Tensor]:
        if track:
            self._steps += 1
            self._sample_rates.append(sample_rate)
        if self.noise_multiplier == 0.0:
            return params
        sigma = self.noise_multiplier * self.max_grad_norm
        return [p + torch.normal(mean=0.0, std=sigma, size=p.shape, device=p.device) for p in params]

    def get_epsilon(self) -> float:
        if self._steps == 0 or self.noise_multiplier == 0.0:
            return 0.0
        avg_rate = sum(self._sample_rates) / len(self._sample_rates) if self._sample_rates else 1.0
        rdp_epsilon = math.sqrt(2.0 * self._steps * avg_rate * math.log(1.0 / self.delta))
        rdp_epsilon /= self.noise_multiplier
        return rdp_epsilon

    def get_privacy_report(self) -> dict:
        return {"epsilon_spent": self.get_epsilon(), "delta": self.delta, "noise_multiplier": self.noise_multiplier, "max_grad_norm": self.max_grad_norm, "steps": self._steps}

    def reset(self) -> None:
        self._steps = 0
        self._sample_rates = []
