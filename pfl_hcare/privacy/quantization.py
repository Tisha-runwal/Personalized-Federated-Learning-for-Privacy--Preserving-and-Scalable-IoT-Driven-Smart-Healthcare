"""k-bit gradient quantization for communication-efficient FL (Eq.8)."""
import torch

class GradientQuantizer:
    def __init__(self, k_bits: int = 8):
        assert k_bits in (2, 4, 8, 16), f"k_bits must be 2, 4, 8, or 16, got {k_bits}"
        self.k_bits = k_bits
        self._last_original_bytes = 0
        self._last_quantized_bytes = 0

    def quantize(self, params: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[dict]]:
        max_val = 2 ** self.k_bits - 1
        quantized, meta = [], []
        original_bytes = quantized_bytes = 0
        for p in params:
            original_bytes += p.numel() * 4
            w_min, w_max = p.min().item(), p.max().item()
            scale = w_max - w_min if w_max != w_min else 1.0
            q = torch.round((p - w_min) / scale * max_val).clamp(0, max_val)
            q = q.to(torch.uint8) if self.k_bits <= 8 else q.to(torch.int32)
            quantized.append(q)
            meta.append({"w_min": w_min, "w_max": w_max, "shape": p.shape})
            quantized_bytes += p.numel() * (self.k_bits / 8)
        self._last_original_bytes = original_bytes
        self._last_quantized_bytes = quantized_bytes
        return quantized, meta

    def dequantize(self, quantized: list[torch.Tensor], meta: list[dict]) -> list[torch.Tensor]:
        max_val = 2 ** self.k_bits - 1
        params = []
        for q, m in zip(quantized, meta):
            scale = m["w_max"] - m["w_min"] if m["w_max"] != m["w_min"] else 1.0
            p = q.float() / max_val * scale + m["w_min"]
            params.append(p.reshape(m["shape"]))
        return params

    def get_bandwidth_report(self) -> dict:
        ratio = self._last_original_bytes / self._last_quantized_bytes if self._last_quantized_bytes > 0 else 0.0
        return {"original_bytes": self._last_original_bytes, "quantized_bytes": self._last_quantized_bytes, "compression_ratio": ratio, "savings_percent": (1.0 - 1.0 / ratio) * 100 if ratio > 0 else 0.0, "k_bits": self.k_bits}
