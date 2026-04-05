import torch
from pfl_hcare.privacy.differential_privacy import DPMechanism

def test_dp_adds_noise():
    dp = DPMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
    params = [torch.ones(10, 10)]
    noisy_params = dp.add_noise(params)
    assert not torch.equal(params[0], noisy_params[0])

def test_dp_clips_gradients():
    dp = DPMechanism(noise_multiplier=0.0, max_grad_norm=1.0, delta=1e-5)
    params = [torch.ones(100) * 100.0]
    clipped = dp.clip_gradients(params)
    assert torch.norm(clipped[0]) <= 1.0 + 1e-6

def test_dp_noise_scale_increases_with_multiplier():
    params = [torch.zeros(1000)]
    dp_low = DPMechanism(noise_multiplier=0.1, max_grad_norm=1.0, delta=1e-5)
    dp_high = DPMechanism(noise_multiplier=2.0, max_grad_norm=1.0, delta=1e-5)
    noisy_low = dp_low.add_noise([p.clone() for p in params])
    noisy_high = dp_high.add_noise([p.clone() for p in params])
    assert noisy_high[0].std().item() > noisy_low[0].std().item()

def test_dp_epsilon_tracking():
    dp = DPMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
    assert dp.get_epsilon() == 0.0
    for _ in range(10):
        dp.add_noise([torch.randn(10)], sample_rate=0.1)
    assert dp.get_epsilon() > 0.0

def test_dp_zero_noise():
    dp = DPMechanism(noise_multiplier=0.0, max_grad_norm=1.0, delta=1e-5)
    params = [torch.ones(10)]
    clipped = dp.clip_gradients(params)
    noisy = dp.add_noise(clipped, track=False)
    assert torch.allclose(noisy[0], clipped[0])
