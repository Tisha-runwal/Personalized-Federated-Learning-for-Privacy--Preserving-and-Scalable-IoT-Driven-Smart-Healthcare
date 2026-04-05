import torch
from pfl_hcare.models.health_classifier import HealthClassifier
from pfl_hcare.models.har_classifier import HARClassifier

def test_health_classifier_forward_shape():
    model = HealthClassifier(input_dim=13, num_classes=2)
    x = torch.randn(8, 13)
    assert model(x).shape == (8, 2)

def test_health_classifier_param_count():
    model = HealthClassifier(input_dim=13, num_classes=2)
    assert sum(p.numel() for p in model.parameters()) < 20000

def test_health_classifier_gradient_flow():
    model = HealthClassifier(input_dim=13, num_classes=2)
    x = torch.randn(4, 13)
    y = torch.randint(0, 2, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

def test_har_classifier_forward_shape():
    model = HARClassifier(num_classes=6)
    x = torch.randn(8, 9, 128)
    assert model(x).shape == (8, 6)

def test_har_classifier_param_count():
    model = HARClassifier(num_classes=6)
    assert sum(p.numel() for p in model.parameters()) < 100000

def test_har_classifier_flat_input():
    model = HARClassifier(num_classes=6, accept_flat=True)
    x = torch.randn(8, 561)
    assert model(x).shape == (8, 6)
