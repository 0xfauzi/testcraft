"""
State Port interface definition.

This module defines the interface for state management operations,
including getting and updating application state.
"""

from typing import Any

from typing_extensions import Protocol


class StatePort(Protocol):
    """
    Interface for state management operations.

    This protocol defines the contract for managing application state,
    including state retrieval, updates, and persistence.
    """

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
            StateError: If state retrieval fails
        """
        ...

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
            **kwargs: Additional update parameters

        Returns:
            Dictionary containing:
                - 'success': Whether the update succeeded
                - 'previous_value': Previous value of the state key
                - 'new_value': New value that was set
                - 'update_metadata': Additional metadata about the update

        Raises:
            StateError: If state update fails
        """
        ...

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
            StateError: If state retrieval fails
        """
        ...

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
            StateError: If state clearing fails
        """
        ...

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
            StateError: If state persistence fails
        """
        ...

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
            StateError: If state loading fails
        """
        ...
