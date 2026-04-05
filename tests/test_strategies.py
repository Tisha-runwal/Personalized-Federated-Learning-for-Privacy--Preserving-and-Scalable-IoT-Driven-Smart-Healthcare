"""Tests for all 5 FL strategy implementations."""
from pfl_hcare.fl.strategies.fedavg import FedAvgStrategy
from pfl_hcare.fl.strategies.fedprox import FedProxStrategy
from pfl_hcare.fl.strategies.per_fedavg import PerFedAvgStrategy
from pfl_hcare.fl.strategies.pfedme import PFedMeStrategy
from pfl_hcare.fl.strategies.pfl_hcare import PFLHCareStrategy
from pfl_hcare.metrics.collector import MetricsCollector


def test_fedavg_strategy_creates():
    mc = MetricsCollector()
    strategy = FedAvgStrategy(metrics_collector=mc)
    assert strategy is not None


def test_fedprox_strategy_creates():
    mc = MetricsCollector()
    strategy = FedProxStrategy(metrics_collector=mc, mu=0.01)
    assert strategy is not None


def test_per_fedavg_strategy_creates():
    mc = MetricsCollector()
    strategy = PerFedAvgStrategy(metrics_collector=mc)
    assert strategy is not None


def test_pfedme_strategy_creates():
    mc = MetricsCollector()
    strategy = PFedMeStrategy(metrics_collector=mc, lambd=15.0)
    assert strategy is not None


def test_pfl_hcare_strategy_creates():
    mc = MetricsCollector()
    strategy = PFLHCareStrategy(
        metrics_collector=mc,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
        k_bits=8,
        adaptive_selection=True,
    )
    assert strategy is not None
