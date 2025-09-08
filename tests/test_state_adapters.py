"""
Tests for state adapter implementations.

This module tests the JSON state adapter including state management,
persistence, coverage history, and generation logging functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from testcraft.adapters.io.safety import SafetyError
from testcraft.adapters.io.state_json import StateJsonAdapter, StateJsonError


class TestStateJsonAdapter:
    """Test the JSON state adapter implementation."""

    def setup_method(self):
        """Set up test fixtures for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adapter = StateJsonAdapter(project_root=self.temp_dir)

        # Clean up any existing state file
        if self.adapter.state_file_path.exists():
            self.adapter.state_file_path.unlink()

    def teardown_method(self):
        """Clean up after each test."""
        # Clean up temp directory
        if self.temp_dir.exists():
            for file in self.temp_dir.rglob("*"):
                if file.is_file():
                    file.unlink()
            self.temp_dir.rmdir()

    def test_initialization_creates_default_structure(self):
        """Test that initialization creates the default state structure."""
        adapter = StateJsonAdapter(project_root=self.temp_dir)

        # Check that default structure exists
        assert adapter.get_state("coverage_history") == {}
        assert adapter.get_state("generation_log") == []
        assert adapter.get_state("idempotent_decisions") == {}
        assert adapter.get_state("file_states") == {}

        metadata = adapter.get_state("metadata")
        assert isinstance(metadata, dict)
        assert "created_at" in metadata
        assert "version" in metadata
        assert metadata["version"] == "1.0.0"

    def test_initialization_with_existing_file(self):
        """Test initialization when state file already exists."""
        # Create a pre-existing state file
        existing_state = {
            "coverage_history": {"test.py": ["some_data"]},
            "generation_log": [{"operation": "test", "timestamp": "2023-01-01"}],
            "custom_key": "custom_value",
        }

        state_file = self.temp_dir / ".testcraft_state.json"
        with open(state_file, "w") as f:
            json.dump(existing_state, f)

        adapter = StateJsonAdapter(project_root=self.temp_dir)

        # Check that existing data is preserved
        assert adapter.get_state("coverage_history") == {"test.py": ["some_data"]}
        assert adapter.get_state("custom_key") == "custom_value"

        # Check that default structure is still added
        assert isinstance(adapter.get_state("idempotent_decisions"), dict)
        assert isinstance(adapter.get_state("file_states"), dict)

    def test_get_state_with_default_value(self):
        """Test getting state with default values."""
        result = self.adapter.get_state("nonexistent_key", "default_value")
        assert result == "default_value"

        result = self.adapter.get_state("also_nonexistent", None)
        assert result is None

    def test_get_state_nested_keys(self):
        """Test getting state with dot notation for nested keys."""
        # Set up nested state
        self.adapter._state_cache["level1"] = {"level2": {"level3": "deep_value"}}

        result = self.adapter.get_state("level1.level2.level3")
        assert result == "deep_value"

        # Test non-existent nested key
        result = self.adapter.get_state("level1.nonexistent.key", "default")
        assert result == "default"

    def test_update_state_replace_strategy(self):
        """Test updating state with replace strategy."""
        result = self.adapter.update_state("test_key", "new_value")

        assert result["success"] is True
        assert result["previous_value"] is None
        assert result["new_value"] == "new_value"
        assert "timestamp" in result["update_metadata"]

        # Verify state was actually updated
        assert self.adapter.get_state("test_key") == "new_value"

    def test_update_state_merge_strategy(self):
        """Test updating state with merge strategy."""
        # Set initial dict value
        self.adapter.update_state("test_dict", {"key1": "value1"})

        # Update with merge strategy
        result = self.adapter.update_state(
            "test_dict", {"key2": "value2"}, merge_strategy="merge"
        )

        assert result["success"] is True
        expected_value = {"key1": "value1", "key2": "value2"}
        assert result["new_value"] == expected_value
        assert self.adapter.get_state("test_dict") == expected_value

    def test_update_state_append_strategy(self):
        """Test updating state with append strategy."""
        # Set initial list value
        self.adapter.update_state("test_list", ["item1"])

        # Append single item
        result = self.adapter.update_state(
            "test_list", "item2", merge_strategy="append"
        )

        assert result["success"] is True
        assert result["new_value"] == ["item1", "item2"]

        # Append multiple items
        self.adapter.update_state(
            "test_list", ["item3", "item4"], merge_strategy="append"
        )

        assert self.adapter.get_state("test_list") == [
            "item1",
            "item2",
            "item3",
            "item4",
        ]

    def test_update_state_nested_keys(self):
        """Test updating nested state using dot notation."""
        result = self.adapter.update_state("level1.level2.key", "nested_value")

        assert result["success"] is True
        assert self.adapter.get_state("level1.level2.key") == "nested_value"

        # Verify structure was created
        level1 = self.adapter.get_state("level1")
        assert isinstance(level1, dict)
        assert level1["level2"]["key"] == "nested_value"

    def test_get_all_state_without_prefix(self):
        """Test getting all state without prefix filter."""
        # Set up some test data
        self.adapter.update_state("key1", "value1")
        self.adapter.update_state("key2", "value2")

        all_state = self.adapter.get_all_state()

        assert isinstance(all_state, dict)
        assert "key1" in all_state
        assert "key2" in all_state
        assert "coverage_history" in all_state  # Default structure

    def test_get_all_state_with_prefix(self):
        """Test getting state filtered by prefix."""
        # Set up nested test data
        self.adapter.update_state("prefix.key1", "value1")
        self.adapter.update_state("prefix.key2", "value2")
        self.adapter.update_state("other.key3", "value3")

        prefix_state = self.adapter.get_all_state("prefix")

        assert "prefix.key1" in prefix_state
        assert "prefix.key2" in prefix_state
        assert "other.key3" not in prefix_state
        assert prefix_state["prefix.key1"] == "value1"
        assert prefix_state["prefix.key2"] == "value2"

    def test_get_all_state_nonexistent_prefix(self):
        """Test getting state with non-existent prefix."""
        result = self.adapter.get_all_state("nonexistent")
        assert result == {}

    def test_clear_state_specific_key(self):
        """Test clearing a specific state key."""
        self.adapter.update_state("test_key", "test_value")
        self.adapter.update_state("keep_key", "keep_value")

        result = self.adapter.clear_state("test_key")

        assert result["success"] is True
        assert "test_key" in result["cleared_keys"]
        assert self.adapter.get_state("test_key") is None
        assert self.adapter.get_state("keep_key") == "keep_value"

    def test_clear_state_all(self):
        """Test clearing all state."""
        self.adapter.update_state("test_key1", "value1")
        self.adapter.update_state("test_key2", "value2")

        result = self.adapter.clear_state()

        assert result["success"] is True
        assert len(result["cleared_keys"]) > 0

        # Default structure should be restored
        assert self.adapter.get_state("coverage_history") == {}
        assert self.adapter.get_state("generation_log") == []

        # Custom keys should be gone
        assert self.adapter.get_state("test_key1") is None
        assert self.adapter.get_state("test_key2") is None

    def test_clear_state_nonexistent_key(self):
        """Test clearing a non-existent key."""
        result = self.adapter.clear_state("nonexistent_key")

        assert result["success"] is True
        assert result["cleared_keys"] == []
        assert "Key not found" in result["clear_metadata"]["reason"]

    def test_persist_state_success(self):
        """Test successful state persistence."""
        self.adapter.update_state("test_key", "test_value")

        result = self.adapter.persist_state()

        assert result["success"] is True
        assert result["persistence_location"] == str(self.adapter.state_file_path)
        assert self.adapter.state_file_path.exists()

        # Verify file contents
        with open(self.adapter.state_file_path) as f:
            data = json.load(f)

        assert data["test_key"] == "test_value"
        assert "coverage_history" in data

    def test_persist_state_no_changes(self):
        """Test persistence when no changes were made."""
        # Create fresh adapter and persist immediately
        result = self.adapter.persist_state()

        # Should still succeed but indicate no changes
        assert result["success"] is True
        assert result["persisted_keys"] == []
        assert "No changes to persist" in result["persistence_metadata"]["reason"]

    def test_persist_state_specific_key(self):
        """Test persisting only a specific key."""
        self.adapter.update_state("key1", "value1")
        self.adapter.update_state("key2", "value2")

        result = self.adapter.persist_state("key1")

        assert result["success"] is True
        assert "key1" in result["persisted_keys"]

    def test_load_state_success(self):
        """Test successful state loading."""
        # Create state data
        test_data = {
            "test_key": "test_value",
            "coverage_history": {"file.py": ["data"]},
        }

        with open(self.adapter.state_file_path, "w") as f:
            json.dump(test_data, f)

        result = self.adapter.load_state()

        assert result["success"] is True
        assert "test_key" in result["loaded_keys"]
        assert result["loaded_values"]["test_key"] == "test_value"
        assert self.adapter.get_state("test_key") == "test_value"

    def test_load_state_file_not_exists(self):
        """Test loading when state file doesn't exist."""
        # Ensure file doesn't exist
        if self.adapter.state_file_path.exists():
            self.adapter.state_file_path.unlink()

        result = self.adapter.load_state()

        assert result["success"] is False
        assert result["loaded_keys"] == []
        assert "does not exist" in result["load_metadata"]["reason"]

    def test_load_state_specific_key(self):
        """Test loading a specific state key."""
        test_data = {"key1": "value1", "key2": "value2"}

        with open(self.adapter.state_file_path, "w") as f:
            json.dump(test_data, f)

        # Load the state first
        self.adapter._load_state_from_file()

        result = self.adapter.load_state("key1")

        assert result["success"] is True
        assert result["loaded_keys"] == ["key1"]
        assert result["loaded_values"]["key1"] == "value1"
        assert "key2" not in result["loaded_values"]

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_persist_state_failure(self, mock_open):
        """Test handling of persistence failures."""
        self.adapter.update_state("test_key", "test_value")

        with pytest.raises(StateJsonError, match="Failed to persist state"):
            self.adapter.persist_state()

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_load_state_failure(self, mock_open):
        """Test handling of loading failures."""
        # Create file first so it exists but fails to read
        self.adapter.state_file_path.touch()

        with pytest.raises(StateJsonError, match="Failed to load state"):
            self.adapter.load_state()

    def test_get_state_failure(self):
        """Test handling of get_state failures."""
        # This is hard to trigger with normal usage, so we'll simulate
        # by corrupting the internal state
        self.adapter._state_cache = "not a dict"

        with pytest.raises(StateJsonError, match="Failed to get state"):
            self.adapter.get_state("any_key")

    def test_update_state_failure(self):
        """Test handling of update_state failures."""
        # Simulate failure by making cache non-dict
        self.adapter._state_cache = "not a dict"

        with pytest.raises(StateJsonError, match="Failed to update state"):
            self.adapter.update_state("key", "value")

    def test_add_coverage_entry(self):
        """Test adding coverage history entries."""
        coverage_data = {
            "line_coverage": 0.85,
            "branch_coverage": 0.78,
            "missing_lines": [10, 15, 20],
        }

        result = self.adapter.add_coverage_entry("test_file.py", coverage_data)

        assert result["success"] is True

        # Verify coverage history was updated
        history = self.adapter.get_coverage_history("test_file.py")
        assert len(history) == 1
        assert history[0]["coverage_data"] == coverage_data
        assert "timestamp" in history[0]

    def test_add_generation_log_entry(self):
        """Test adding generation log entries."""
        operation_details = {
            "file_path": "test_file.py",
            "status": "success",
            "test_count": 5,
        }

        result = self.adapter.add_generation_log_entry(
            "generate_tests", operation_details
        )

        assert result["success"] is True

        # Verify log entry was added
        log = self.adapter.get_generation_log()
        assert len(log) == 1
        assert log[0]["operation"] == "generate_tests"
        assert log[0]["details"] == operation_details
        assert "timestamp" in log[0]

    def test_set_idempotent_decision(self):
        """Test setting idempotent decisions."""
        decision_data = {
            "decision": "skip_generation",
            "reason": "file_unchanged",
            "file_hash": "abc123",
        }

        result = self.adapter.set_idempotent_decision("test_file.py", decision_data)

        assert result["success"] is True

        # Verify decision was set
        decision = self.adapter.get_state("idempotent_decisions.test_file.py")
        assert decision["decision"] == "skip_generation"
        assert decision["reason"] == "file_unchanged"
        assert "timestamp" in decision

    def test_get_coverage_history_all_files(self):
        """Test getting coverage history for all files."""
        # Add coverage for multiple files
        self.adapter.add_coverage_entry("file1.py", {"coverage": 0.8})
        self.adapter.add_coverage_entry("file2.py", {"coverage": 0.9})

        history = self.adapter.get_coverage_history()

        assert isinstance(history, dict)
        assert "file1.py" in history
        assert "file2.py" in history

    def test_get_coverage_history_specific_file(self):
        """Test getting coverage history for a specific file."""
        self.adapter.add_coverage_entry("file1.py", {"coverage": 0.8})
        self.adapter.add_coverage_entry("file2.py", {"coverage": 0.9})

        history = self.adapter.get_coverage_history("file1.py")

        assert len(history) == 1
        assert history[0]["coverage_data"]["coverage"] == 0.8

    def test_get_coverage_history_nonexistent_file(self):
        """Test getting coverage history for non-existent file."""
        history = self.adapter.get_coverage_history("nonexistent.py")
        assert history == []

    def test_get_generation_log_with_limit(self):
        """Test getting generation log with entry limit."""
        # Add multiple log entries
        for i in range(5):
            self.adapter.add_generation_log_entry(f"operation_{i}", {"index": i})

        # Get only last 3 entries
        log = self.adapter.get_generation_log(limit=3)

        assert len(log) == 3
        assert log[0]["operation"] == "operation_2"  # Should be the 3rd from end
        assert log[2]["operation"] == "operation_4"  # Should be the last

    def test_get_generation_log_no_limit(self):
        """Test getting all generation log entries."""
        # Add multiple log entries
        for i in range(3):
            self.adapter.add_generation_log_entry(f"operation_{i}", {"index": i})

        log = self.adapter.get_generation_log()

        assert len(log) == 3
        assert log[0]["operation"] == "operation_0"
        assert log[2]["operation"] == "operation_2"

    def test_should_regenerate_file_new_file(self):
        """Test regeneration decision for new file."""
        result = self.adapter.should_regenerate_file("new_file.py", "hash123")
        assert result is True

    def test_should_regenerate_file_unchanged(self):
        """Test regeneration decision for unchanged file."""
        # Set file state
        self.adapter.update_state("file_states.existing_file.py", {"hash": "hash123"})

        result = self.adapter.should_regenerate_file("existing_file.py", "hash123")
        assert result is False

    def test_should_regenerate_file_changed(self):
        """Test regeneration decision for changed file."""
        # Set file state with old hash
        self.adapter.update_state("file_states.existing_file.py", {"hash": "old_hash"})

        result = self.adapter.should_regenerate_file("existing_file.py", "new_hash")
        assert result is True

    def test_custom_state_file_name(self):
        """Test using a custom state file name."""
        custom_name = "custom_state.json"
        adapter = StateJsonAdapter(project_root=self.temp_dir, state_file=custom_name)

        assert adapter.state_file_name == custom_name
        assert adapter.state_file_path.name == custom_name

        # Test that it works normally
        adapter.update_state("test", "value")
        adapter.persist_state()

        assert (self.temp_dir / custom_name).exists()

    def test_json_serialization_with_datetime(self):
        """Test that datetime objects are properly serialized."""
        now = datetime.utcnow()
        self.adapter.update_state("datetime_test", now)

        result = self.adapter.persist_state()
        assert result["success"] is True

        # Reload and verify
        self.adapter.load_state()
        value = self.adapter.get_state("datetime_test")
        # Should be serialized as string
        assert isinstance(value, str)


class TestStateJsonAdapterSafety:
    """Test safety and security aspects of the state adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adapter = StateJsonAdapter(project_root=self.temp_dir)

    def teardown_method(self):
        """Clean up after tests."""
        if self.temp_dir.exists():
            for file in self.temp_dir.rglob("*"):
                if file.is_file():
                    file.unlink()
            self.temp_dir.rmdir()

    @patch("testcraft.adapters.io.safety.SafetyPolicies.validate_file_path")
    def test_persist_state_validates_path(self, mock_validate):
        """Test that state persistence validates file paths."""
        mock_validate.side_effect = SafetyError("Path validation failed")

        with pytest.raises(StateJsonError, match="Failed to persist state"):
            self.adapter.persist_state()

        mock_validate.assert_called_once()

    def test_large_state_handling(self):
        """Test handling of large state data."""
        # Create moderately large state
        large_data = {"key" + str(i): "value" * 100 for i in range(1000)}

        for key, value in large_data.items():
            self.adapter.update_state(key, value)

        # Should be able to persist and load
        result = self.adapter.persist_state()
        assert result["success"] is True

        # Reload to verify
        new_adapter = StateJsonAdapter(project_root=self.temp_dir)
        for key, expected_value in large_data.items():
            assert new_adapter.get_state(key) == expected_value


class TestStateJsonAdapterIntegration:
    """Integration tests for the state adapter."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adapter = StateJsonAdapter(project_root=self.temp_dir)

    def teardown_method(self):
        """Clean up after integration tests."""
        if self.temp_dir.exists():
            for file in self.temp_dir.rglob("*"):
                if file.is_file():
                    file.unlink()
            self.temp_dir.rmdir()

    def test_full_workflow_simulation(self):
        """Test a complete workflow simulation."""
        # Simulate a test generation session

        # 1. Initial coverage measurement
        initial_coverage = {
            "line_coverage": 0.60,
            "branch_coverage": 0.45,
            "missing_lines": [10, 15, 20, 25, 30],
        }
        self.adapter.add_coverage_entry("main.py", initial_coverage)

        # 2. Generate tests
        self.adapter.add_generation_log_entry(
            "generate_tests",
            {"file_path": "main.py", "tests_generated": 3, "status": "success"},
        )

        # 3. Set idempotent decision
        self.adapter.set_idempotent_decision(
            "main.py",
            {
                "decision": "generated",
                "file_hash": "abc123",
                "test_file": "test_main.py",
            },
        )

        # 4. Update file state
        self.adapter.update_state(
            "file_states.main.py",
            {
                "hash": "abc123",
                "last_processed": datetime.utcnow().isoformat(),
                "test_file": "test_main.py",
            },
        )

        # 5. Post-generation coverage
        final_coverage = {
            "line_coverage": 0.90,
            "branch_coverage": 0.85,
            "missing_lines": [25],
        }
        self.adapter.add_coverage_entry("main.py", final_coverage)

        # 6. Persist all changes
        result = self.adapter.persist_state()
        assert result["success"] is True

        # 7. Simulate new session - create new adapter
        new_adapter = StateJsonAdapter(project_root=self.temp_dir)

        # 8. Verify all data is preserved
        coverage_history = new_adapter.get_coverage_history("main.py")
        assert len(coverage_history) == 2
        assert coverage_history[0]["coverage_data"]["line_coverage"] == 0.60
        assert coverage_history[1]["coverage_data"]["line_coverage"] == 0.90

        generation_log = new_adapter.get_generation_log()
        assert len(generation_log) == 1
        assert generation_log[0]["operation"] == "generate_tests"

        decision = new_adapter.get_state("idempotent_decisions.main.py")
        assert decision["decision"] == "generated"
        assert decision["file_hash"] == "abc123"

        # 9. Test idempotent decision making
        should_regen = new_adapter.should_regenerate_file("main.py", "abc123")
        assert should_regen is False  # Same hash, should not regenerate

        should_regen = new_adapter.should_regenerate_file("main.py", "new_hash")
        assert should_regen is True  # Different hash, should regenerate
