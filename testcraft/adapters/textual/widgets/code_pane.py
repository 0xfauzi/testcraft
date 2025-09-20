"""
CodePane widget for displaying code snippets.

Provides syntax-highlighted code display with scrolling,
line numbers, and modal viewing capabilities.
"""

from rich.panel import Panel
from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Static


class CodePane(Static):
    """
    Widget for displaying syntax-highlighted code.

    Can display code inline or in a modal for focused viewing.
    """

    def __init__(
        self,
        code: str = "",
        language: str = "python",
        theme: str = "monokai",
        line_numbers: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_class("code-pane")

        self._code = code
        self._language = language
        self._theme = theme
        self._line_numbers = line_numbers

        self._update_display()

    def _update_display(self) -> None:
        """Update the code display."""
        if not self._code.strip():
            self.update("No code to display")
            return

        # Create syntax-highlighted code
        syntax = Syntax(
            self._code,
            lexer=self._language,
            theme=self._theme,
            line_numbers=self._line_numbers,
            word_wrap=True,
            background_color="default",
        )

        # Wrap in a panel
        panel = Panel(
            syntax,
            title=f"Code ({self._language})",
            title_align="left",
            border_style="blue",
        )

        self.update(panel)

    def set_code(
        self, code: str, language: str | None = None, theme: str | None = None
    ) -> None:
        """Set new code content."""
        self._code = code
        if language is not None:
            self._language = language
        if theme is not None:
            self._theme = theme

        self._update_display()

    def clear_code(self) -> None:
        """Clear the displayed code."""
        self._code = ""
        self._update_display()

    def show_modal(self) -> None:
        """Show the code in a modal screen for focused viewing."""
        if self.app and self._code.strip():
            modal = CodeViewerModal(
                self._code, self._language, self._theme, self._line_numbers
            )
            self.app.push_screen(modal)


class CodeViewerModal(ModalScreen):
    """
    Modal screen for viewing code in a larger, focused view.

    Provides a full-screen code viewer with close button and
    keyboard navigation.
    """

    BINDINGS = [
        ("escape,q", "close", "Close"),
        ("ctrl+c", "close", "Close"),
    ]

    def __init__(
        self,
        code: str,
        language: str = "python",
        theme: str = "monokai",
        line_numbers: bool = True,
    ):
        super().__init__()
        self._code = code
        self._language = language
        self._theme = theme
        self._line_numbers = line_numbers

    def compose(self) -> ComposeResult:
        """Compose the modal content."""
        yield Header()
        yield Footer()

        with Vertical(id="modal-content"):
            # Title and close button
            with Horizontal(id="modal-header"):
                yield Static(f"Code Viewer - {self._language}", id="modal-title")
                yield Button("Close", id="close-button", variant="error")

            # Code display in scrollable container
            with VerticalScroll(id="code-container"):
                syntax = Syntax(
                    self._code,
                    lexer=self._language,
                    theme=self._theme,
                    line_numbers=self._line_numbers,
                    word_wrap=False,  # Don't wrap in modal for better readability
                    background_color="default",
                )
                yield Static(syntax, id="modal-code")

    def action_close(self) -> None:
        """Close the modal."""
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close-button":
            self.dismiss()


class InlineCodeSnippet(Static):
    """
    A compact widget for displaying small code snippets inline.

    Useful for showing single lines or small blocks of code
    without taking up too much space.
    """

    def __init__(self, code: str = "", language: str = "python", **kwargs):
        super().__init__(**kwargs)
        self.add_class("inline-code")

        self._code = code
        self._language = language

        self._update_display()

    def _update_display(self) -> None:
        """Update the inline code display."""
        if not self._code.strip():
            self.update("")
            return

        # For inline display, use simpler formatting
        # Truncate if too long
        display_code = self._code.strip()
        if len(display_code) > 100:
            display_code = display_code[:97] + "..."

        # Create simple syntax highlighting
        syntax = Syntax(
            display_code,
            lexer=self._language,
            theme="default",
            line_numbers=False,
            word_wrap=False,
            background_color="default",
        )

        self.update(syntax)

    def set_code(self, code: str, language: str | None = None) -> None:
        """Set new code content."""
        self._code = code
        if language is not None:
            self._language = language

        self._update_display()


# CSS for the modal (would typically be in the main theme file)
MODAL_CSS = """
#modal-content {
    dock: fill;
    background: $surface;
    border: thick $accent;
    padding: 1;
}

#modal-header {
    dock: top;
    height: 3;
    background: $primary;
    color: white;
    padding: 0 1;
}

#modal-title {
    text-style: bold;
    width: 1fr;
    content-align: center middle;
}

#close-button {
    width: 10;
    margin-left: 1;
}

#code-container {
    background: $surface;
    border: solid $secondary;
    margin: 1 0;
}

#modal-code {
    padding: 1;
}
"""
