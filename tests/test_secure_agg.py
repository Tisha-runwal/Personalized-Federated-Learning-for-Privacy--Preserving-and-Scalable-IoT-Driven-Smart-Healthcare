import torch
from pfl_hcare.privacy.secure_aggregation import SimulatedSecureAggregator

def test_encrypt_decrypt_preserves_values():
    sa = SimulatedSecureAggregator(latency_range_ms=(0, 0))
    params = [torch.randn(10, 10)]
    encrypted = sa.encrypt(params)
    decrypted = sa.decrypt(encrypted)
    for orig, dec in zip(params, decrypted):
        assert torch.equal(orig, dec)

def test_encrypt_produces_metadata():
    sa = SimulatedSecureAggregator(latency_range_ms=(0, 1))
    sa.encrypt([torch.randn(10)])
    report = sa.get_report()
    assert report["status"] == "encrypted"
    assert report["latency_ms"] >= 0

def test_aggregate_multiple_clients():
    sa = SimulatedSecureAggregator(latency_range_ms=(0, 0))
    client_params = [[torch.ones(5) * 1.0], [torch.ones(5) * 3.0]]
    result = sa.aggregate(client_params, [0.5, 0.5])
    assert torch.allclose(result[0], torch.ones(5) * 2.0)
