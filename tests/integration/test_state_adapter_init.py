from pathlib import Path

from testcraft.adapters.io.state_json import StateJsonAdapter


def test_state_adapter_initializes_without_deadlock(tmp_path: Path):
    project_root = tmp_path
    state_path = project_root / ".testcraft_state.json"

    # Ensure no pre-existing state file
    assert not state_path.exists()

    adapter = StateJsonAdapter(project_root=project_root)

    # Should have created the state file and returned promptly (no hang)
    assert state_path.exists()

    # Exercise an append update to ensure nested get/update flow works
    result = adapter.update_state(
        "generation_log",
        {"operation": "init", "details": {}},
        merge_strategy="append",
    )
    assert result["success"] is True

    # Persist and verify file is writable
    persist = adapter.persist_state()
    assert persist["success"] is True
