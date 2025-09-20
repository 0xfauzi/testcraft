"""
Centralized state management for the TestCraft Textual UI application.

This module provides a robust state management system with:
- Reactive state updates with subscriptions
- State persistence to disk
- Undo/redo capabilities
- History tracking
"""

import json
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


class StateManager:
    """
    Centralized state management for the TUI application.

    Provides reactive state management with subscriptions, persistence,
    and history tracking for undo/redo operations.
    """

    def __init__(self, persist_path: Path | None = None):
        """
        Initialize the state manager.

        Args:
            persist_path: Optional path to persist state to disk
        """
        self._state: dict[str, Any] = {}
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[dict] = []
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._persist_path = persist_path
        self._max_history_size = 100
        self._batch_updates = False
        self._batch_changes: list[dict] = []

        # Load persisted state if available
        self._load_state()

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get state value with optional default.

        Args:
            key: The state key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The state value or default
        """
        # Support nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            value = self._state
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, default)
                else:
                    return default
            return value

        return self._state.get(key, default)

    def set_state(self, key: str, value: Any, notify: bool = True) -> None:
        """
        Set state and notify subscribers.

        Args:
            key: The state key to set
            value: The new value
            notify: Whether to notify subscribers
        """
        old_value = self.get_state(key)

        # Support nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            current = self._state
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self._state[key] = value

        # Track history
        change = {
            "key": key,
            "old": old_value,
            "new": value,
            "timestamp": datetime.now().isoformat(),
        }

        if self._batch_updates:
            self._batch_changes.append(change)
        else:
            self._add_to_history(change)

            # Notify subscribers
            if notify:
                self._notify_subscribers(key, value, old_value)

            # Persist state
            self._persist_state()

    def update_state(self, updates: dict[str, Any], notify: bool = True) -> None:
        """
        Update multiple state values at once.

        Args:
            updates: Dictionary of key-value pairs to update
            notify: Whether to notify subscribers
        """
        with self.batch_update():
            for key, value in updates.items():
                self.set_state(key, value, notify)

    def subscribe(self, key: str, callback: Callable) -> Callable:
        """
        Subscribe to state changes.

        Args:
            key: State key to watch (supports wildcards with *)
            callback: Function to call when state changes

        Returns:
            Unsubscribe function
        """
        self._subscribers[key].append(callback)

        # Return unsubscribe function
        def unsubscribe():
            if callback in self._subscribers[key]:
                self._subscribers[key].remove(callback)

        return unsubscribe

    def subscribe_many(self, subscriptions: dict[str, Callable]) -> list[Callable]:
        """
        Subscribe to multiple state keys at once.

        Args:
            subscriptions: Dictionary of key-callback pairs

        Returns:
            List of unsubscribe functions
        """
        return [
            self.subscribe(key, callback) for key, callback in subscriptions.items()
        ]

    @contextmanager
    def batch_update(self):
        """
        Context manager for batching state updates.

        Useful for making multiple changes that should be treated as a single
        transaction for history and notification purposes.
        """
        self._batch_updates = True
        self._batch_changes = []

        try:
            yield
        finally:
            self._batch_updates = False

            if self._batch_changes:
                # Add batch as single history entry
                self._add_to_history(
                    {
                        "type": "batch",
                        "changes": self._batch_changes,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Notify subscribers for each change
                for change in self._batch_changes:
                    self._notify_subscribers(
                        change["key"], change["new"], change["old"]
                    )

                # Persist state once
                self._persist_state()

    def undo(self) -> bool:
        """
        Undo the last state change.

        Returns:
            True if undo was successful, False if nothing to undo
        """
        if not self._undo_stack:
            return False

        change = self._undo_stack.pop()
        self._redo_stack.append(change)

        # Apply the undo
        if change.get("type") == "batch":
            # Undo batch changes in reverse order
            for sub_change in reversed(change["changes"]):
                self._apply_change(sub_change["key"], sub_change["old"])
        else:
            self._apply_change(change["key"], change["old"])

        self._persist_state()
        return True

    def redo(self) -> bool:
        """
        Redo a previously undone change.

        Returns:
            True if redo was successful, False if nothing to redo
        """
        if not self._redo_stack:
            return False

        change = self._redo_stack.pop()
        self._undo_stack.append(change)

        # Apply the redo
        if change.get("type") == "batch":
            # Redo batch changes in original order
            for sub_change in change["changes"]:
                self._apply_change(sub_change["key"], sub_change["new"])
        else:
            self._apply_change(change["key"], change["new"])

        self._persist_state()
        return True

    def get_history(self, limit: int = 10) -> list[dict]:
        """
        Get recent state change history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of recent history entries
        """
        return self._history[-limit:]

    def clear_history(self) -> None:
        """Clear all history and undo/redo stacks."""
        self._history.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()

    def reset_state(self, new_state: dict[str, Any] | None = None) -> None:
        """
        Reset the entire state.

        Args:
            new_state: Optional new state to set
        """
        old_state = self._state.copy()
        self._state = new_state or {}

        # Add to history
        self._add_to_history(
            {
                "type": "reset",
                "old": old_state,
                "new": self._state,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Notify all subscribers
        for key in set(old_state.keys()) | set(self._state.keys()):
            self._notify_subscribers(key, self._state.get(key), old_state.get(key))

        self._persist_state()

    def _add_to_history(self, change: dict) -> None:
        """Add a change to history and manage stack size."""
        self._history.append(change)
        self._undo_stack.append(change)
        self._redo_stack.clear()  # Clear redo stack on new change

        # Limit history size
        if len(self._history) > self._max_history_size:
            self._history = self._history[-self._max_history_size :]
        if len(self._undo_stack) > self._max_history_size:
            self._undo_stack = self._undo_stack[-self._max_history_size :]

    def _apply_change(self, key: str, value: Any) -> None:
        """Apply a state change without history tracking."""
        if "." in key:
            parts = key.split(".")
            current = self._state
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self._state[key] = value

    def _notify_subscribers(self, key: str, new_value: Any, old_value: Any) -> None:
        """Notify all relevant subscribers of a state change."""
        # Notify exact key subscribers
        for callback in self._subscribers.get(key, []):
            try:
                callback(new_value, old_value)
            except Exception as e:
                print(f"Error in state subscriber for {key}: {e}")

        # Notify wildcard subscribers
        for pattern, callbacks in self._subscribers.items():
            if "*" in pattern:
                # Simple wildcard matching
                if self._matches_pattern(key, pattern):
                    for callback in callbacks:
                        try:
                            callback(new_value, old_value)
                        except Exception as e:
                            print(f"Error in wildcard subscriber for {pattern}: {e}")

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if a key matches a wildcard pattern."""
        if pattern == "*":
            return True

        # Convert pattern to regex-like matching
        import re

        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return re.match(f"^{regex_pattern}$", key) is not None

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self._persist_path and self._persist_path.exists():
            try:
                with open(self._persist_path) as f:
                    data = json.load(f)
                    self._state = data.get("state", {})
                    # Optionally load history
                    if "history" in data:
                        self._history = data["history"][-self._max_history_size :]
            except Exception as e:
                print(f"Error loading state: {e}")

    def _persist_state(self) -> None:
        """Persist state to disk."""
        if self._persist_path:
            try:
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._persist_path, "w") as f:
                    json.dump(
                        {
                            "state": self._state,
                            "history": self._history[
                                -10:
                            ],  # Save last 10 history entries
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                        default=str,
                    )
            except Exception as e:
                print(f"Error persisting state: {e}")

    def get_state_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current state.

        Returns:
            Dictionary with state statistics
        """
        return {
            "total_keys": len(self._state),
            "subscribers": len(self._subscribers),
            "history_size": len(self._history),
            "undo_available": len(self._undo_stack),
            "redo_available": len(self._redo_stack),
            "persisted": bool(self._persist_path),
        }
