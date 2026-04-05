from pfl_hcare.metrics.collector import MetricsCollector

def test_collector_records_round():
    mc = MetricsCollector()
    mc.record_round(round_num=1, method="pfl_hcare", global_accuracy=0.85, global_loss=0.42)
    history = mc.get_history()
    assert len(history) == 1
    assert history[0]["round"] == 1
    assert history[0]["metrics"]["global_accuracy"] == 0.85

def test_collector_multiple_rounds():
    mc = MetricsCollector()
    for i in range(5):
        mc.record_round(round_num=i, method="fedavg", global_accuracy=0.80 + i * 0.02)
    assert len(mc.get_history()) == 5

def test_collector_callbacks():
    received = []
    mc = MetricsCollector()
    mc.on_round(lambda data: received.append(data))
    mc.record_round(round_num=0, method="test", global_accuracy=0.9)
    assert len(received) == 1
    assert received[0]["metrics"]["global_accuracy"] == 0.9

def test_collector_to_json():
    mc = MetricsCollector()
    mc.record_round(round_num=0, method="test", global_accuracy=0.9)
    json_str = mc.to_json()
    assert '"global_accuracy": 0.9' in json_str
