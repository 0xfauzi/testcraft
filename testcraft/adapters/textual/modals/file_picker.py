"""
File picker dialog for TestCraft Textual UI.

Provides a dialog for selecting files or directories with
navigation and filtering capabilities.
"""

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, DirectoryTree, Input, Static

from .base import BaseModal


class FilePickerDialog(BaseModal):
    """
    Modal dialog for file/directory selection.

    Provides a file browser with navigation, filtering, and
    multi-selection support.
    """

    DEFAULT_CSS = """
    FilePickerDialog .modal-container {
        width: 80%;
        height: 80%;
        max-width: 100;
        max-height: 40;
    }
    
    FilePickerDialog .file-browser {
        height: 100%;
        border: solid $border;
        margin: 1 0;
        background: $surface;
    }
    
    FilePickerDialog .path-bar {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $border;
    }
    
    FilePickerDialog .path-input {
        width: 100%;
        margin: 0;
    }
    
    FilePickerDialog .filter-bar {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $border;
    }
    
    FilePickerDialog .filter-input {
        width: 100%;
        margin: 0;
    }
    
    FilePickerDialog DirectoryTree {
        background: transparent;
        padding: 1;
    }
    
    FilePickerDialog DirectoryTree:focus {
        border: none;
    }
    
    FilePickerDialog .selected-file {
        color: $primary;
        text-style: bold;
    }
    
    FilePickerDialog .file-info {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $border;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("enter", "select", "Select", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+l", "focus_path", "Edit Path", show=False),
        Binding("ctrl+f", "focus_filter", "Filter", show=False),
        Binding("backspace", "go_up", "Go Up", show=False),
        Binding("tab", "toggle_selection", "Toggle", show=False),
    ]

    def __init__(
        self,
        title: str = "Select File",
        initial_path: Path = None,
        file_filter: str = "*",
        select_directory: bool = False,
        multiple: bool = False,
        show_hidden: bool = False,
        **kwargs,
    ):
        """
        Initialize the file picker dialog.

        Args:
            title: Title for the dialog
            initial_path: Initial directory to show
            file_filter: File filter pattern (e.g., "*.py", "*.txt")
            select_directory: Whether to select directories instead of files
            multiple: Whether to allow multiple selection
            show_hidden: Whether to show hidden files
            **kwargs: Additional arguments passed to BaseModal
        """
        super().__init__(title=title, **kwargs)
        self.current_path = initial_path or Path.cwd()
        self.file_filter = file_filter
        self.select_directory = select_directory
        self.multiple = multiple
        self.show_hidden = show_hidden
        self.selected_paths: set[Path] = set()
        self._tree: DirectoryTree | None = None
        self._path_input: Input | None = None
        self._filter_input: Input | None = None
        self._info_label: Static | None = None

    def compose_content(self) -> ComposeResult:
        """Compose the file picker content."""
        with Vertical():
            # Path bar
            with Horizontal(classes="path-bar"):
                self._path_input = Input(
                    value=str(self.current_path),
                    placeholder="Enter path...",
                    id="path-input",
                    classes="path-input",
                )
                yield self._path_input

            # File browser
            with ScrollableContainer(classes="file-browser"):
                self._tree = DirectoryTree(path=str(self.current_path), id="file-tree")
                yield self._tree

            # Filter bar
            with Horizontal(classes="filter-bar"):
                self._filter_input = Input(
                    value=self.file_filter,
                    placeholder="Filter (e.g., *.py)...",
                    id="filter-input",
                    classes="filter-input",
                )
                yield self._filter_input

            # Info bar
            self._info_label = Static(self._get_info_text(), classes="file-info")
            yield self._info_label

    def compose_footer(self) -> ComposeResult:
        """Compose the dialog footer with buttons."""
        with Horizontal(classes="modal-buttons"):
            select_text = "Select Directory" if self.select_directory else "Select"
            yield Button(select_text, id="select", variant="primary")
            yield Button("Cancel", id="cancel", variant="default")

            if self.multiple:
                yield Button("Clear Selection", id="clear", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "select":
            self.action_select()
        elif event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "clear":
            self.clear_selection()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """Handle file selection in the tree."""
        path = Path(event.path)

        if self.select_directory:
            # For directory selection, select the parent directory
            if path.is_file():
                path = path.parent

        if self.multiple:
            # Toggle selection for multiple mode
            if path in self.selected_paths:
                self.selected_paths.remove(path)
            else:
                self.selected_paths.add(path)
            self._update_info()
        else:
            # Single selection mode
            self.selected_paths = {path}
            self._update_info()

            # Auto-submit if not in multiple mode
            if not self.select_directory or path.is_dir():
                self.action_select()

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        """Handle directory selection in the tree."""
        path = Path(event.path)

        if self.select_directory:
            if self.multiple:
                # Toggle selection for multiple mode
                if path in self.selected_paths:
                    self.selected_paths.remove(path)
                else:
                    self.selected_paths.add(path)
                self._update_info()
            else:
                # Single selection mode
                self.selected_paths = {path}
                self._update_info()
                self.action_select()
        else:
            # Navigate into directory for file selection
            self.current_path = path
            self._path_input.value = str(path)
            self._tree.path = str(path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "path-input":
            # Navigate to entered path
            path = Path(event.value).expanduser().resolve()
            if path.exists():
                self.current_path = path if path.is_dir() else path.parent
                self._tree.path = str(self.current_path)
                self._path_input.value = str(self.current_path)
        elif event.input.id == "filter-input":
            # Apply filter
            self.file_filter = event.value
            self._apply_filter()

    def _apply_filter(self) -> None:
        """Apply the current file filter to the tree."""
        # Note: DirectoryTree doesn't have built-in filtering,
        # so this would need custom implementation or tree refresh
        pass

    def _update_info(self) -> None:
        """Update the info label with selection information."""
        if self._info_label:
            self._info_label.update(self._get_info_text())

    def _get_info_text(self) -> str:
        """Get the current info text."""
        if not self.selected_paths:
            if self.select_directory:
                return "Select a directory"
            else:
                return f"Select a file (filter: {self.file_filter})"
        elif len(self.selected_paths) == 1:
            path = list(self.selected_paths)[0]
            return f"Selected: {path.name}"
        else:
            return f"Selected: {len(self.selected_paths)} items"

    def clear_selection(self) -> None:
        """Clear all selections."""
        self.selected_paths.clear()
        self._update_info()

    def action_select(self) -> None:
        """Handle the select action."""
        if self.selected_paths:
            if self.multiple:
                self.dismiss(list(self.selected_paths))
            else:
                self.dismiss(
                    list(self.selected_paths)[0] if self.selected_paths else None
                )
        else:
            # No selection, use current directory if selecting directories
            if self.select_directory:
                self.dismiss(self.current_path)

    def action_cancel(self) -> None:
        """Handle the cancel action."""
        self.dismiss(None)

    def action_go_up(self) -> None:
        """Navigate to parent directory."""
        if self.current_path.parent != self.current_path:
            self.current_path = self.current_path.parent
            self._tree.path = str(self.current_path)
            self._path_input.value = str(self.current_path)

    def action_focus_path(self) -> None:
        """Focus the path input."""
        self._path_input.focus()

    def action_focus_filter(self) -> None:
        """Focus the filter input."""
        self._filter_input.focus()

    def action_toggle_selection(self) -> None:
        """Toggle selection of current item in multiple mode."""
        if self.multiple and self._tree:
            # Get currently highlighted node
            # This would need access to tree's current selection
            pass

    @classmethod
    def pick_file(
        cls,
        app,
        title: str = "Select File",
        initial_path: Path = None,
        file_filter: str = "*",
        multiple: bool = False,
        callback: callable = None,
    ) -> "FilePickerDialog":
        """
        Convenience method to show a file picker dialog.

        Args:
            app: The Textual app instance
            title: Dialog title
            initial_path: Initial directory
            file_filter: File filter pattern
            multiple: Whether to allow multiple selection
            callback: Optional callback function

        Returns:
            The dialog instance
        """
        dialog = cls(
            title=title,
            initial_path=initial_path,
            file_filter=file_filter,
            select_directory=False,
            multiple=multiple,
        )

        if callback:
            dialog.set_callback(callback)

        app.push_screen(dialog)
        return dialog

    @classmethod
    def pick_directory(
        cls,
        app,
        title: str = "Select Directory",
        initial_path: Path = None,
        callback: callable = None,
    ) -> "FilePickerDialog":
        """
        Convenience method to show a directory picker dialog.

        Args:
            app: The Textual app instance
            title: Dialog title
            initial_path: Initial directory
            callback: Optional callback function

        Returns:
            The dialog instance
        """
        dialog = cls(
            title=title,
            initial_path=initial_path,
            select_directory=True,
            multiple=False,
        )

        if callback:
            dialog.set_callback(callback)

        app.push_screen(dialog)
        return dialog
