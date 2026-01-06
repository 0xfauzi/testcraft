from pathlib import Path

from testcraft.adapters.io.state_json import StateJsonAdapter


def test_state_adapter_reentrant_update_get(tmp_path: Path):
    adapter = StateJsonAdapter(project_root=tmp_path)

    # Prime a nested structure
    adapter.update_state("idempotent_decisions.sample", {"value": 1})

    # update_state internally uses _get_state_unlocked while holding the lock
    # This should complete without deadlock and correctly append
    res = adapter.update_state(
        "generation_log",
        {"operation": "write", "meta": {"x": 1}},
        merge_strategy="append",
    )
    assert res["success"] is True

    # Ensure get_state works and returns a list for generation_log
    log = adapter.get_state("generation_log", [])
    assert isinstance(log, list)
    assert len(log) >= 1
