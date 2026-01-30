"""Performance metrics collection for the chatbot."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List


class MetricsCollector:
    """Collect and report performance metrics."""

    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._query_count = 0
        self._total_confidence = 0.0

    def time_operation(self, name: str):
        """Context manager to time an operation."""
        return _Timer(self, name)

    def record_query(self, confidence: float):
        """Record a query and its confidence score."""
        self._query_count += 1
        self._total_confidence += confidence
        self._counters["queries"] += 1

        if confidence >= 0.90:
            self._counters["high_confidence"] += 1
        elif confidence >= 0.70:
            self._counters["medium_confidence"] += 1
        else:
            self._counters["low_confidence"] += 1

    def increment(self, counter_name: str, amount: int = 1):
        """Increment a counter."""
        self._counters[counter_name] += amount

    def get_summary(self) -> Dict[str, Any]:
        """Return metrics summary."""
        summary: Dict[str, Any] = {
            "counters": dict(self._counters),
        }

        if self._query_count > 0:
            summary["average_confidence"] = round(
                self._total_confidence / self._query_count, 3
            )

        for name, timings in self._timings.items():
            if timings:
                summary[f"timing_{name}"] = {
                    "count": len(timings),
                    "avg_ms": round(sum(timings) / len(timings) * 1000, 2),
                    "min_ms": round(min(timings) * 1000, 2),
                    "max_ms": round(max(timings) * 1000, 2),
                }

        return summary


class _Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        self.collector._timings[self.name].append(elapsed)
