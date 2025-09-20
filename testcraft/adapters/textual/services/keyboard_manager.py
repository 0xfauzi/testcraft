"""
Keyboard navigation and shortcut management for TestCraft Textual UI.

This module provides:
- Vi-mode navigation support
- Customizable keyboard shortcuts
- Context-aware key bindings
- Navigation mode management
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from textual.binding import Binding


class NavigationMode(Enum):
    """Available navigation modes for the application."""

    NORMAL = "normal"
    INSERT = "insert"
    COMMAND = "command"
    VISUAL = "visual"
    SEARCH = "search"


@dataclass
class KeyBinding:
    """Represents a keyboard binding."""

    key: str
    action: str
    description: str
    mode: NavigationMode = NavigationMode.NORMAL
    show: bool = True
    priority: int = 0


class KeyboardManager:
    """
    Manages keyboard navigation and shortcuts.

    Provides modal navigation (vi-style), customizable shortcuts,
    and context-aware key handling.
    """

    # Common key patterns
    VI_NAVIGATION = {
        "h": "move_left",
        "j": "move_down",
        "k": "move_up",
        "l": "move_right",
        "g": "go_top",
        "G": "go_bottom",
        "ctrl+d": "page_down",
        "ctrl+u": "page_up",
        "ctrl+f": "page_down",
        "ctrl+b": "page_up",
        "0": "go_line_start",
        "$": "go_line_end",
        "w": "next_word",
        "b": "previous_word",
        "/": "search_forward",
        "?": "search_backward",
        "n": "next_match",
        "N": "previous_match",
        "i": "enter_insert_mode",
        "v": "enter_visual_mode",
        ":": "enter_command_mode",
        "escape": "enter_normal_mode",
    }

    GLOBAL_SHORTCUTS = {
        "ctrl+p": ("command_palette", "Open command palette"),
        "ctrl+/": ("show_shortcuts", "Show keyboard shortcuts"),
        "ctrl+tab": ("next_screen", "Next screen"),
        "ctrl+shift+tab": ("previous_screen", "Previous screen"),
        "ctrl+s": ("save", "Save"),
        "ctrl+z": ("undo", "Undo"),
        "ctrl+y": ("redo", "Redo"),
        "ctrl+r": ("refresh", "Refresh"),
        "ctrl+q": ("quit", "Quit application"),
        "f1": ("help", "Show help"),
        "f11": ("fullscreen", "Toggle fullscreen"),
    }

    def __init__(self, vi_mode_enabled: bool = True):
        """
        Initialize the keyboard manager.

        Args:
            vi_mode_enabled: Whether to enable vi-style navigation
        """
        self.mode = NavigationMode.NORMAL
        self.vi_mode_enabled = vi_mode_enabled
        self.key_bindings: dict[NavigationMode, dict[str, KeyBinding]] = {
            mode: {} for mode in NavigationMode
        }
        self.mode_stack: list[NavigationMode] = []
        self.key_buffer: list[str] = []
        self.repeat_count: int = 0
        self._callbacks: dict[str, Callable] = {}
        self._disabled_keys: set[str] = set()

        # Initialize default bindings
        self._initialize_default_bindings()

    def _initialize_default_bindings(self) -> None:
        """Set up default key bindings."""
        # Add vi navigation bindings if enabled
        if self.vi_mode_enabled:
            for key, action in self.VI_NAVIGATION.items():
                self.register_binding(
                    key=key,
                    action=action,
                    description=action.replace("_", " ").title(),
                    mode=NavigationMode.NORMAL,
                )

        # Add global shortcuts
        for key, (action, description) in self.GLOBAL_SHORTCUTS.items():
            for mode in NavigationMode:
                self.register_binding(
                    key=key,
                    action=action,
                    description=description,
                    mode=mode,
                    priority=10,  # Higher priority for global shortcuts
                )

    def register_binding(
        self,
        key: str,
        action: str,
        description: str = "",
        mode: NavigationMode = NavigationMode.NORMAL,
        show: bool = True,
        priority: int = 0,
    ) -> None:
        """
        Register a keyboard binding.

        Args:
            key: Key combination (e.g., "ctrl+s", "j", "shift+tab")
            action: Action identifier or method name
            description: Human-readable description
            mode: Navigation mode this binding applies to
            show: Whether to show in help/footer
            priority: Priority for conflicting bindings (higher wins)
        """
        binding = KeyBinding(key, action, description, mode, show, priority)

        if mode not in self.key_bindings:
            self.key_bindings[mode] = {}

        # Check for conflicts and resolve by priority
        if key in self.key_bindings[mode]:
            existing = self.key_bindings[mode][key]
            if binding.priority >= existing.priority:
                self.key_bindings[mode][key] = binding
        else:
            self.key_bindings[mode][key] = binding

    def unregister_binding(self, key: str, mode: NavigationMode | None = None) -> None:
        """
        Remove a keyboard binding.

        Args:
            key: Key combination to unregister
            mode: Specific mode to remove from (None for all modes)
        """
        if mode:
            self.key_bindings[mode].pop(key, None)
        else:
            for mode_bindings in self.key_bindings.values():
                mode_bindings.pop(key, None)

    def set_mode(self, mode: NavigationMode) -> None:
        """
        Switch to a different navigation mode.

        Args:
            mode: The mode to switch to
        """
        if mode != self.mode:
            self.mode_stack.append(self.mode)
            self.mode = mode
            self.key_buffer.clear()
            self.repeat_count = 0

    def get_mode(self) -> NavigationMode:
        """Get the current navigation mode."""
        return self.mode

    def push_mode(self, mode: NavigationMode) -> None:
        """
        Push a new mode onto the stack.

        Args:
            mode: Mode to push
        """
        self.mode_stack.append(self.mode)
        self.mode = mode

    def pop_mode(self) -> NavigationMode:
        """
        Pop the previous mode from the stack.

        Returns:
            The previous mode
        """
        if self.mode_stack:
            self.mode = self.mode_stack.pop()
        return self.mode

    def handle_key(self, key: str) -> str | None:
        """
        Process a keyboard input based on current mode.

        Args:
            key: The key pressed

        Returns:
            The action to perform, or None if no binding
        """
        # Check if key is disabled
        if key in self._disabled_keys:
            return None

        # Handle escape specially - always returns to normal mode
        if key == "escape" and self.mode != NavigationMode.NORMAL:
            self.set_mode(NavigationMode.NORMAL)
            return "enter_normal_mode"

        # Handle numeric prefix for repeat counts
        if self.mode == NavigationMode.NORMAL and key.isdigit() and key != "0":
            self.repeat_count = self.repeat_count * 10 + int(key)
            return None

        # Add to key buffer for multi-key sequences
        self.key_buffer.append(key)

        # Check for multi-key sequences
        key_sequence = "".join(self.key_buffer)

        # Look for binding in current mode
        binding = self.key_bindings[self.mode].get(key_sequence)

        if binding:
            self.key_buffer.clear()
            action = binding.action

            # Apply repeat count if applicable
            if self.repeat_count > 0:
                action = f"{action}:{self.repeat_count}"
                self.repeat_count = 0

            return action

        # Check if this could be the start of a multi-key sequence
        for registered_key in self.key_bindings[self.mode]:
            if registered_key.startswith(key_sequence):
                return None  # Wait for more keys

        # No match found, clear buffer
        self.key_buffer.clear()
        self.repeat_count = 0
        return None

    def get_bindings_for_mode(
        self, mode: NavigationMode | None = None
    ) -> list[Binding]:
        """
        Get Textual bindings for the specified mode.

        Args:
            mode: Mode to get bindings for (current mode if None)

        Returns:
            List of Textual Binding objects
        """
        target_mode = mode or self.mode
        bindings = []

        for binding in self.key_bindings[target_mode].values():
            if binding.show:
                bindings.append(
                    Binding(
                        key=binding.key,
                        action=binding.action,
                        description=binding.description,
                        show=binding.show,
                    )
                )

        # Sort by priority
        bindings.sort(
            key=lambda b: self.key_bindings[target_mode][b.key].priority, reverse=True
        )
        return bindings

    def get_all_bindings(self) -> dict[NavigationMode, list[Binding]]:
        """Get all bindings organized by mode."""
        return {mode: self.get_bindings_for_mode(mode) for mode in NavigationMode}

    def disable_key(self, key: str) -> None:
        """
        Temporarily disable a key.

        Args:
            key: Key to disable
        """
        self._disabled_keys.add(key)

    def enable_key(self, key: str) -> None:
        """
        Re-enable a disabled key.

        Args:
            key: Key to enable
        """
        self._disabled_keys.discard(key)

    def is_key_disabled(self, key: str) -> bool:
        """
        Check if a key is disabled.

        Args:
            key: Key to check

        Returns:
            True if disabled
        """
        return key in self._disabled_keys

    def register_callback(self, action: str, callback: Callable) -> None:
        """
        Register a callback for an action.

        Args:
            action: Action identifier
            callback: Function to call
        """
        self._callbacks[action] = callback

    def execute_action(self, action: str) -> bool:
        """
        Execute a registered action.

        Args:
            action: Action to execute

        Returns:
            True if action was executed
        """
        # Check for repeat count
        if ":" in action:
            action, count = action.split(":")
            count = int(count)
        else:
            count = 1

        if action in self._callbacks:
            for _ in range(count):
                self._callbacks[action]()
            return True

        return False

    def get_mode_indicator(self) -> str:
        """
        Get a string indicator for the current mode.

        Returns:
            Mode indicator string
        """
        indicators = {
            NavigationMode.NORMAL: "-- NORMAL --",
            NavigationMode.INSERT: "-- INSERT --",
            NavigationMode.VISUAL: "-- VISUAL --",
            NavigationMode.COMMAND: ":",
            NavigationMode.SEARCH: "/",
        }
        return indicators.get(self.mode, "")

    def export_bindings(self) -> dict:
        """
        Export current bindings configuration.

        Returns:
            Dictionary representation of bindings
        """
        export = {}
        for mode in NavigationMode:
            export[mode.value] = [
                {
                    "key": binding.key,
                    "action": binding.action,
                    "description": binding.description,
                    "show": binding.show,
                    "priority": binding.priority,
                }
                for binding in self.key_bindings[mode].values()
            ]
        return export

    def import_bindings(self, config: dict) -> None:
        """
        Import bindings from configuration.

        Args:
            config: Dictionary of bindings to import
        """
        for mode_str, bindings in config.items():
            mode = NavigationMode(mode_str)
            for binding_data in bindings:
                self.register_binding(
                    key=binding_data["key"],
                    action=binding_data["action"],
                    description=binding_data.get("description", ""),
                    mode=mode,
                    show=binding_data.get("show", True),
                    priority=binding_data.get("priority", 0),
                )
