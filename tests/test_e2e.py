"""End-to-end smoke test: runs a small FL simulation and verifies metrics are collected."""
import yaml
from pfl_hcare.fl.server import run_simulation
from pfl_hcare.metrics.collector import MetricsCollector


def test_e2e_fedavg_smoke():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["training"]["num_clients"] = 3
    config["training"]["num_rounds"] = 2
    config["training"]["local_epochs"] = 1
    config["dataset"]["name"] = "mimic"
    mc = MetricsCollector()
    run_simulation(config, method="fedavg", metrics_collector=mc)
    history = mc.get_history()
    assert len(history) > 0


def test_e2e_pfl_hcare_smoke():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    config["training"]["num_clients"] = 3
    config["training"]["num_rounds"] = 2
    config["training"]["local_epochs"] = 1
    config["dataset"]["name"] = "mimic"
    config["maml"]["second_order"] = False
    config["maml"]["inner_steps"] = 1
    mc = MetricsCollector()
    run_simulation(config, method="pfl_hcare", metrics_collector=mc)
    history = mc.get_history()
    assert len(history) > 0
