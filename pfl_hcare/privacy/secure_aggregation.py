"""Simulated secure aggregation for dashboard visualization (Eq.6-7)."""
import time, random, torch

class SimulatedSecureAggregator:
    def __init__(self, latency_range_ms: tuple[int, int] = (50, 200), seed: int = 42):
        self.latency_range_ms = latency_range_ms
        self.rng = random.Random(seed)
        self._status = "idle"
        self._last_latency_ms = 0
        self._encrypted_size_bytes = 0

    def encrypt(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        self._status = "encrypting"
        latency = self.rng.randint(*self.latency_range_ms) if self.latency_range_ms[1] > 0 else 0
        time.sleep(latency / 1000.0)
        self._last_latency_ms = latency
        self._encrypted_size_bytes = sum(p.numel() * 4 for p in params)
        self._status = "encrypted"
        return [p.clone() for p in params]

    def decrypt(self, params: list[torch.Tensor]) -> list[torch.Tensor]:
        self._status = "decrypting"
        latency = self.rng.randint(*self.latency_range_ms) if self.latency_range_ms[1] > 0 else 0
        time.sleep(latency / 1000.0)
        self._last_latency_ms += latency
        self._status = "decrypted"
        return [p.clone() for p in params]

    def aggregate(self, client_params: list[list[torch.Tensor]], weights: list[float]) -> list[torch.Tensor]:
        encrypted = [self.encrypt(cp) for cp in client_params]
        n_params = len(encrypted[0])
        aggregated = []
        for param_idx in range(n_params):
            weighted_sum = sum(w * encrypted[ci][param_idx] for ci, w in enumerate(weights))
            aggregated.append(weighted_sum)
        return self.decrypt(aggregated)

    def get_report(self) -> dict:
        return {"status": self._status, "latency_ms": self._last_latency_ms, "encrypted_size_bytes": self._encrypted_size_bytes}
