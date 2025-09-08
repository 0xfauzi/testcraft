"""
State adapter that manages application state using JSON storage.

This adapter provides project-scoped JSON state storage for managing
coverage history, generation logs, and idempotent decisions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .safety import SafetyError, SafetyPolicies


class StateJsonError(Exception):
    """Exception raised when JSON state operations fail."""

    pass


class StateJsonAdapter:
    """
    State adapter that manages application state using JSON storage.

    This adapter implements project-scoped JSON state storage with methods for
    initializing, updating, and querying state data including coverage history
    and generation logs.
    """

    def __init__(self, project_root: Path | None = None, state_file: str | None = None):
        """
        Initialize the JSON state adapter.

        Args:
            project_root: Project root path for state file location
            state_file: Optional custom state file name (default: .testcraft_state.json)
        """
        self.project_root = project_root or Path.cwd()
        self.state_file_name = state_file or ".testcraft_state.json"
        self.state_file_path = self.project_root / self.state_file_name
        self.logger = logging.getLogger(__name__)

        # In-memory state cache
        self._state_cache: dict[str, Any] = {}
        self._cache_dirty = False

        # Initialize state if file doesn't exist
        if not self.state_file_path.exists():
            self._initialize_state()
        else:
            self._load_state_from_file()

    def get_state(
        self, state_key: str, default_value: Any | None = None, **kwargs: Any
    ) -> Any:
        """
        Get the current value of a state key.

        Args:
            state_key: Key identifying the state to retrieve
            default_value: Default value to return if key doesn't exist
            **kwargs: Additional retrieval parameters

        Returns:
            Current value of the state key, or default_value if not found

        Raises:
            StateJsonError: If state retrieval fails
        """
        try:
            self.logger.debug(f"Getting state for key: {state_key}")

            # Navigate nested keys using dot notation
            keys = state_key.split(".")
            current = self._state_cache

            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    self.logger.debug(f"State key not found: {state_key}")
                    return default_value

            return current

        except Exception as e:
            raise StateJsonError(
                f"Failed to get state for key '{state_key}': {e}"
            ) from e

    def update_state(
        self,
        state_key: str,
        new_value: Any,
        merge_strategy: str = "replace",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Update the value of a state key.

        Args:
            state_key: Key identifying the state to update
            new_value: New value to set for the state key
            merge_strategy: Strategy for merging with existing values
                          ("replace", "merge", "append")
            **kwargs: Additional update parameters

        Returns:
            Dictionary containing:
                - 'success': Whether the update succeeded
                - 'previous_value': Previous value of the state key
                - 'new_value': New value that was set
                - 'update_metadata': Additional metadata about the update

        Raises:
            StateJsonError: If state update fails
        """
        try:
            self.logger.debug(f"Updating state for key: {state_key}")

            # Get previous value
            previous_value = self.get_state(state_key)

            # Navigate to parent container and key
            keys = state_key.split(".")
            current = self._state_cache

            # Navigate to the parent container
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            final_key = keys[-1]

            # Apply merge strategy
            if merge_strategy == "replace":
                current[final_key] = new_value
            elif (
                merge_strategy == "merge"
                and isinstance(previous_value, dict)
                and isinstance(new_value, dict)
            ):
                if final_key not in current:
                    current[final_key] = {}
                current[final_key].update(new_value)
            elif merge_strategy == "append" and isinstance(previous_value, list):
                if final_key not in current:
                    current[final_key] = []
                if isinstance(new_value, list):
                    current[final_key].extend(new_value)
                else:
                    current[final_key].append(new_value)
            else:
                # Fallback to replace for unsupported merge strategies
                current[final_key] = new_value

            # Mark cache as dirty
            self._cache_dirty = True

            # Add timestamp to metadata
            update_metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "merge_strategy": merge_strategy,
                "state_key": state_key,
            }

            return {
                "success": True,
                "previous_value": previous_value,
                "new_value": current[final_key],
                "update_metadata": update_metadata,
            }

        except Exception as e:
            raise StateJsonError(
                f"Failed to update state for key '{state_key}': {e}"
            ) from e

    def get_all_state(
        self, state_prefix: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Get all state values, optionally filtered by prefix.

        Args:
            state_prefix: Optional prefix to filter state keys
            **kwargs: Additional retrieval parameters

        Returns:
            Dictionary mapping state keys to their values

        Raises:
            StateJsonError: If state retrieval fails
        """
        try:
            self.logger.debug(f"Getting all state with prefix: {state_prefix}")

            if state_prefix is None:
                return dict(self._state_cache)

            # Filter by prefix
            filtered_state = {}
            prefix_parts = state_prefix.split(".")
            current = self._state_cache

            # Navigate to the prefix location
            for part in prefix_parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return {}  # Prefix doesn't exist

            # If current is a dict, return its contents with full keys
            if isinstance(current, dict):

                def _build_filtered_dict(
                    data: dict[str, Any], prefix: str
                ) -> dict[str, Any]:
                    result = {}
                    for key, value in data.items():
                        full_key = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, dict):
                            result.update(_build_filtered_dict(value, full_key))
                        else:
                            result[full_key] = value
                    return result

                filtered_state = _build_filtered_dict(current, state_prefix)
            else:
                # If it's not a dict, return the value with the prefix as key
                filtered_state[state_prefix] = current

            return filtered_state

        except Exception as e:
            raise StateJsonError(
                f"Failed to get all state with prefix '{state_prefix}': {e}"
            ) from e

    def clear_state(
        self, state_key: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Clear state values, optionally for a specific key.

        Args:
            state_key: Optional specific key to clear (clears all if None)
            **kwargs: Additional clearing parameters

        Returns:
            Dictionary containing:
                - 'success': Whether the clearing succeeded
                - 'cleared_keys': List of keys that were cleared
                - 'clear_metadata': Additional metadata about the clearing

        Raises:
            StateJsonError: If state clearing fails
        """
        try:
            self.logger.debug(f"Clearing state for key: {state_key}")

            cleared_keys = []

            if state_key is None:
                # Clear all state
                cleared_keys = list(self._state_cache.keys())
                self._state_cache.clear()
                self._initialize_default_structure()
            else:
                # Clear specific key
                keys = state_key.split(".")
                current = self._state_cache

                # Navigate to parent
                for key in keys[:-1]:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        # Key doesn't exist, nothing to clear
                        return {
                            "success": True,
                            "cleared_keys": [],
                            "clear_metadata": {
                                "timestamp": datetime.utcnow().isoformat(),
                                "reason": "Key not found",
                            },
                        }

                final_key = keys[-1]
                if isinstance(current, dict) and final_key in current:
                    del current[final_key]
                    cleared_keys.append(state_key)

            # Mark cache as dirty
            self._cache_dirty = True

            clear_metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "cleared_count": len(cleared_keys),
            }

            return {
                "success": True,
                "cleared_keys": cleared_keys,
                "clear_metadata": clear_metadata,
            }

        except Exception as e:
            raise StateJsonError(
                f"Failed to clear state for key '{state_key}': {e}"
            ) from e

    def persist_state(
        self, state_key: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Persist state to storage.

        Args:
            state_key: Optional specific key to persist (persists all if None)
            **kwargs: Additional persistence parameters

        Returns:
            Dictionary containing:
                - 'success': Whether persistence succeeded
                - 'persisted_keys': List of keys that were persisted
                - 'persistence_location': Location where state was persisted
                - 'persistence_metadata': Additional metadata about persistence

        Raises:
            StateJsonError: If state persistence fails
        """
        try:
            self.logger.debug(f"Persisting state for key: {state_key}")

            if not self._cache_dirty and state_key is None:
                # No changes to persist
                return {
                    "success": True,
                    "persisted_keys": [],
                    "persistence_location": str(self.state_file_path),
                    "persistence_metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "reason": "No changes to persist",
                    },
                }

            # Validate file path
            SafetyPolicies.validate_file_path(self.state_file_path, self.project_root)

            # Ensure directory exists
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data to persist
            data_to_persist = self._state_cache
            persisted_keys = list(self._state_cache.keys())

            if state_key is not None:
                # Only persist specific key
                state_value = self.get_state(state_key)
                if state_value is not None:
                    # Create a minimal state structure with just this key
                    keys = state_key.split(".")
                    data_to_persist = {}
                    current = data_to_persist

                    for _i, key in enumerate(keys[:-1]):
                        current[key] = {}
                        current = current[key]

                    current[keys[-1]] = state_value
                    persisted_keys = [state_key]
                else:
                    persisted_keys = []

            # Write to file with proper JSON formatting
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_persist, f, indent=2, sort_keys=True, default=str)

            # Clear dirty flag
            self._cache_dirty = False

            persistence_metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "file_size": self.state_file_path.stat().st_size,
                "key_count": len(persisted_keys),
            }

            return {
                "success": True,
                "persisted_keys": persisted_keys,
                "persistence_location": str(self.state_file_path),
                "persistence_metadata": persistence_metadata,
            }

        except (OSError, SafetyError, json.JSONDecodeError) as e:
            raise StateJsonError(f"Failed to persist state: {e}") from e

    def load_state(self, state_key: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """
        Load state from storage.

        Args:
            state_key: Optional specific key to load (loads all if None)
            **kwargs: Additional loading parameters

        Returns:
            Dictionary containing:
                - 'success': Whether loading succeeded
                - 'loaded_keys': List of keys that were loaded
                - 'loaded_values': Dictionary of loaded state values
                - 'load_metadata': Additional metadata about loading

        Raises:
            StateJsonError: If state loading fails
        """
        try:
            self.logger.debug(f"Loading state for key: {state_key}")

            if not self.state_file_path.exists():
                return {
                    "success": False,
                    "loaded_keys": [],
                    "loaded_values": {},
                    "load_metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "reason": "State file does not exist",
                    },
                }

            # Load from file
            loaded_data = self._load_state_from_file()

            if state_key is None:
                # Load all state
                self._state_cache = loaded_data
                loaded_keys = list(loaded_data.keys())
                loaded_values = dict(loaded_data)
            else:
                # Load specific key
                loaded_value = self.get_state(state_key)
                if loaded_value is not None:
                    loaded_keys = [state_key]
                    loaded_values = {state_key: loaded_value}
                else:
                    loaded_keys = []
                    loaded_values = {}

            load_metadata = {
                "timestamp": datetime.utcnow().isoformat(),
                "file_size": self.state_file_path.stat().st_size,
                "key_count": len(loaded_keys),
            }

            return {
                "success": True,
                "loaded_keys": loaded_keys,
                "loaded_values": loaded_values,
                "load_metadata": load_metadata,
            }

        except (OSError, json.JSONDecodeError) as e:
            raise StateJsonError(f"Failed to load state: {e}") from e

    def _initialize_state(self) -> None:
        """Initialize default state structure."""
        self.logger.debug("Initializing default state structure")

        self._state_cache = {}
        self._initialize_default_structure()
        self._cache_dirty = True

        # Persist initial state
        self.persist_state()

    def _initialize_default_structure(self) -> None:
        """Initialize the default state structure with required sections."""
        default_structure = {
            "coverage_history": {},
            "generation_log": [],
            "idempotent_decisions": {},
            "file_states": {},
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat(),
                "version": "1.0.0",
            },
        }

        # Merge with existing cache, preserving any existing data
        for key, value in default_structure.items():
            if key not in self._state_cache:
                self._state_cache[key] = value

    def _load_state_from_file(self) -> dict[str, Any]:
        """Load state from the JSON file."""
        try:
            with open(self.state_file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Ensure default structure exists
            self._state_cache = data
            self._initialize_default_structure()
            self._cache_dirty = False

            return data

        except (OSError, json.JSONDecodeError) as e:
            raise StateJsonError(f"Failed to load state from file: {e}") from e

    # Convenience methods for common state operations

    def add_coverage_entry(
        self, file_path: str, coverage_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Add a coverage entry to the coverage history."""
        timestamp = datetime.utcnow().isoformat()
        coverage_entry = {"timestamp": timestamp, "coverage_data": coverage_data}

        return self.update_state(
            f"coverage_history.{file_path}", coverage_entry, merge_strategy="append"
        )

    def add_generation_log_entry(
        self, operation: str, details: dict[str, Any]
    ) -> dict[str, Any]:
        """Add an entry to the generation log."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "details": details,
        }

        return self.update_state("generation_log", log_entry, merge_strategy="append")

    def set_idempotent_decision(
        self, decision_key: str, decision_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Set an idempotent decision."""
        return self.update_state(
            f"idempotent_decisions.{decision_key}",
            {"timestamp": datetime.utcnow().isoformat(), **decision_data},
        )

    def get_coverage_history(self, file_path: str | None = None) -> dict[str, Any]:
        """Get coverage history, optionally for a specific file."""
        if file_path:
            return self.get_state(f"coverage_history.{file_path}", [])
        return self.get_state("coverage_history", {})

    def get_generation_log(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get generation log entries, optionally limited to recent entries."""
        log = self.get_state("generation_log", [])
        if limit:
            return log[-limit:]
        return log

    def should_regenerate_file(self, file_path: str, current_hash: str) -> bool:
        """Determine if a file should be regenerated based on state."""
        file_state = self.get_state(f"file_states.{file_path}")
        if not file_state:
            return True

        return file_state.get("hash") != current_hash
