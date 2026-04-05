import torch
from pfl_hcare.maml.maml import MAMLWrapper
from pfl_hcare.models.health_classifier import HealthClassifier

def test_maml_inner_loop_updates_params():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=3, second_order=False)
    X = torch.randn(16, 13)
    y = torch.randint(0, 2, (16,))
    original_params = [p.clone() for p in model.parameters()]
    adapted_params = maml.inner_loop(X, y)
    for orig, adapted in zip(original_params, adapted_params):
        assert not torch.equal(orig, adapted)

def test_maml_inner_loop_returns_list_of_tensors():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=3, second_order=False)
    X = torch.randn(8, 13)
    y = torch.randint(0, 2, (8,))
    adapted_params = maml.inner_loop(X, y)
    model_params = list(model.parameters())
    assert len(adapted_params) == len(model_params)
    for ap, mp in zip(adapted_params, model_params):
        assert ap.shape == mp.shape

def test_maml_outer_loss_computes_gradient():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=1, second_order=False)
    loss = maml.outer_loss(torch.randn(8, 13), torch.randint(0, 2, (8,)), torch.randn(8, 13), torch.randint(0, 2, (8,)))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

def test_maml_second_order_mode():
    model = HealthClassifier(input_dim=13, num_classes=2)
    maml = MAMLWrapper(model, inner_lr=0.01, inner_steps=1, second_order=True)
    loss = maml.outer_loss(torch.randn(8, 13), torch.randint(0, 2, (8,)), torch.randn(8, 13), torch.randint(0, 2, (8,)))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
