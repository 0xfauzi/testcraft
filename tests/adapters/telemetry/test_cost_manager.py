from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from testcraft.adapters.telemetry.cost_manager import CostEntry, CostManager


def _make_entry(
    *,
    timestamp: datetime,
    cost: float,
    tokens: int,
    service: str = "openai",
    operation: str = "responses",
) -> CostEntry:
    return CostEntry(
        id=str(uuid4()),
        timestamp=timestamp,
        service=service,
        operation=operation,
        cost=cost,
        tokens_used=tokens,
        api_calls=1,
    )


def test_get_summary_respects_time_filters(tmp_path: Path) -> None:
    manager = CostManager(config={}, telemetry=None, storage_path=tmp_path)

    now = datetime.now()
    manager.cost_entries = [
        _make_entry(timestamp=now - timedelta(minutes=30), cost=0.5, tokens=100),
        _make_entry(timestamp=now - timedelta(minutes=10), cost=1.0, tokens=200),
        _make_entry(timestamp=now, cost=2.0, tokens=300),
    ]

    summary_all = manager.get_summary()
    assert summary_all["total_cost"] == 3.5
    assert summary_all["usage_stats"]["total_tokens"] == 600

    recent_start = now - timedelta(minutes=15)
    summary_recent = manager.get_summary(start_time=recent_start)
    assert summary_recent["total_cost"] == 3.0
    assert summary_recent["usage_stats"]["total_tokens"] == 500

    window_start = now - timedelta(minutes=25)
    window_end = now - timedelta(minutes=5)
    summary_window = manager.get_summary(start_time=window_start, end_time=window_end)
    assert summary_window["total_cost"] == 1.0
    assert summary_window["usage_stats"]["total_tokens"] == 200
