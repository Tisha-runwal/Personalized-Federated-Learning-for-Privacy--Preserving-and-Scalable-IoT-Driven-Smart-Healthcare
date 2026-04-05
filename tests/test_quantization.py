import torch
from pfl_hcare.privacy.quantization import GradientQuantizer

def test_quantize_dequantize_recovers_approximate_values():
    q = GradientQuantizer(k_bits=8)
    params = [torch.randn(100)]
    quantized, meta = q.quantize(params)
    dequantized = q.dequantize(quantized, meta)
    assert torch.allclose(params[0], dequantized[0], atol=0.05)

def test_quantize_reduces_size():
    q = GradientQuantizer(k_bits=8)
    params = [torch.randn(1000)]
    original_bytes = sum(p.numel() * 4 for p in params)
    quantized, meta = q.quantize(params)
    quantized_bytes = sum(p.numel() * (q.k_bits / 8) for p in quantized)
    assert quantized_bytes < original_bytes

def test_quantize_2bit():
    q = GradientQuantizer(k_bits=2)
    params = [torch.randn(100)]
    quantized, meta = q.quantize(params)
    for qp in quantized:
        assert qp.min() >= 0
        assert qp.max() <= 3

def test_quantize_16bit():
    q = GradientQuantizer(k_bits=16)
    params = [torch.randn(100)]
    quantized, meta = q.quantize(params)
    dequantized = q.dequantize(quantized, meta)
    assert torch.allclose(params[0], dequantized[0], atol=0.001)

def test_bandwidth_report():
    q = GradientQuantizer(k_bits=8)
    q.quantize([torch.randn(1000)])
    report = q.get_bandwidth_report()
    assert report["original_bytes"] > 0
    assert report["quantized_bytes"] > 0
    assert report["compression_ratio"] > 1.0
