"""Model-Agnostic Meta-Learning (MAML) for personalized federated learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MAMLWrapper:
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, inner_steps: int = 5, second_order: bool = False):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.second_order = second_order

    def _functional_forward(self, x: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass using given params via torch.func.functional_call (eval mode)."""
        named_params = dict(zip([n for n, _ in self.model.named_parameters()], params))
        was_training = self.model.training
        self.model.eval()
        try:
            out = torch.func.functional_call(self.model, named_params, (x,))
        finally:
            if was_training:
                self.model.train()
        return out

    def inner_loop(self, support_X: torch.Tensor, support_y: torch.Tensor, params: Optional[list[torch.Tensor]] = None) -> list[torch.Tensor]:
        if params is None:
            params = [p.clone().requires_grad_(True) for p in self.model.parameters()]
        else:
            params = [p.requires_grad_(True) if not p.requires_grad else p for p in params]

        for _ in range(self.inner_steps):
            logits = self._functional_forward(support_X, params)
            loss = F.cross_entropy(logits, support_y)
            grads = torch.autograd.grad(loss, params, create_graph=self.second_order, allow_unused=True)
            params = [
                p - self.inner_lr * g if g is not None else p
                for p, g in zip(params, grads)
            ]
        return params

    def outer_loss(self, support_X, support_y, query_X, query_y) -> torch.Tensor:
        # Start from model parameters with requires_grad to connect computation graph
        init_params = [p.clone().requires_grad_(True) for p in self.model.parameters()]
        adapted_params = self.inner_loop(support_X, support_y, params=init_params)
        # Use retain_graph so model.parameters() can receive gradients
        logits = self._functional_forward(query_X, adapted_params)
        loss = F.cross_entropy(logits, query_y)
        # Propagate gradients to init_params (which are clones of model.parameters())
        grads_init = torch.autograd.grad(loss, init_params, allow_unused=True, retain_graph=True)
        # Assign those gradients to the actual model parameters
        for p, g in zip(self.model.parameters(), grads_init):
            if g is not None:
                p.grad = g if p.grad is None else p.grad + g
        return loss
