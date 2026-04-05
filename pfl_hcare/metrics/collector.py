"""Metrics collection and callback system for FL training rounds."""
import json
from typing import Any, Callable

class MetricsCollector:
    def __init__(self):
        self._history: list[dict] = []
        self._callbacks: list[Callable[[dict], None]] = []

    def record_round(self, round_num: int, method: str, **metrics: Any) -> None:
        entry = {"type": "round_update", "round": round_num, "method": method, "metrics": metrics}
        self._history.append(entry)
        for cb in self._callbacks:
            cb(entry)

    def on_round(self, callback: Callable[[dict], None]) -> None:
        self._callbacks.append(callback)

    def get_history(self) -> list[dict]:
        return list(self._history)

    def get_latest(self) -> dict | None:
        return self._history[-1] if self._history else None

    def to_json(self) -> str:
        return json.dumps(self._history, indent=2)

    def reset(self) -> None:
        self._history.clear()
