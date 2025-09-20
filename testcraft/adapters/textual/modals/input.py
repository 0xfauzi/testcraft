"""
Input dialog for TestCraft Textual UI.

Provides a dialog for getting text input from the user with
optional validation.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.validation import Validator
from textual.widgets import Button, Input, Static

from .base import BaseModal


class InputDialog(BaseModal):
    """
    Modal dialog for text input.

    Provides a customizable input dialog with validation support
    and helpful prompts.
    """

    DEFAULT_CSS = """
    InputDialog .modal-prompt {
        margin: 1 0;
        color: $text;
    }

    InputDialog .modal-input {
        margin: 1 0;
        width: 100%;
    }

    InputDialog .modal-help {
        margin-top: 0.5;
        color: $text-muted;
        font-size: 90%;
    }

    InputDialog .modal-error {
        margin-top: 0.5;
        color: $error;
        font-size: 90%;
    }

    InputDialog Input {
        width: 100%;
    }

    InputDialog Input:focus {
        border: tall $primary;
    }

    InputDialog Input.-invalid {
        border: tall $error;
    }
    """

    BINDINGS = [
        Binding("enter", "submit", "Submit", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        prompt: str,
        title: str = "Input",
        placeholder: str = "",
        default_value: str = "",
        password: bool = False,
        validator: Validator | None = None,
        help_text: str = "",
        submit_text: str = "OK",
        cancel_text: str = "Cancel",
        **kwargs,
    ):
        """
        Initialize the input dialog.

        Args:
            prompt: The prompt message to display
            title: Title for the dialog
            placeholder: Placeholder text for the input
            default_value: Default value for the input
            password: Whether this is a password input
            validator: Optional validator for the input
            help_text: Optional help text to display
            submit_text: Text for the submit button
            cancel_text: Text for the cancel button
            **kwargs: Additional arguments passed to BaseModal
        """
        super().__init__(title=title, **kwargs)
        self.prompt = prompt
        self.placeholder = placeholder
        self.default_value = default_value
        self.password = password
        self.validator = validator
        self.help_text = help_text
        self.submit_text = submit_text
        self.cancel_text = cancel_text
        self._error_label: Static | None = None

    def compose_content(self) -> ComposeResult:
        """Compose the input dialog content."""
        # Prompt
        yield Static(self.prompt, classes="modal-prompt")

        # Input field
        with Vertical(classes="modal-input"):
            input_widget = Input(
                value=self.default_value,
                placeholder=self.placeholder,
                password=self.password,
                id="input-field",
            )

            if self.validator:
                input_widget.validators = [self.validator]

            yield input_widget

            # Help text
            if self.help_text:
                yield Static(self.help_text, classes="modal-help")

            # Error message placeholder
            self._error_label = Static("", classes="modal-error")
            self._error_label.display = False
            yield self._error_label

    def compose_footer(self) -> ComposeResult:
        """Compose the dialog footer with buttons."""
        with Horizontal(classes="modal-buttons"):
            yield Button(self.submit_text, id="submit", variant="primary")
            yield Button(self.cancel_text, id="cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "submit":
            self.action_submit()
        elif event.button.id == "cancel":
            self.action_cancel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        self.action_submit()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for real-time validation."""
        if self.validator and self._error_label:
            input_widget = self.query_one("#input-field", Input)

            # Validate the current value
            result = self.validator.validate(event.value)

            if result.is_valid:
                self._error_label.display = False
                input_widget.remove_class("-invalid")
            else:
                self._error_label.update(
                    result.failure_descriptions[0]
                    if result.failure_descriptions
                    else "Invalid input"
                )
                self._error_label.display = True
                input_widget.add_class("-invalid")

    def action_submit(self) -> None:
        """Handle the submit action."""
        input_widget = self.query_one("#input-field", Input)
        value = input_widget.value

        # Validate if validator is set
        if self.validator:
            result = self.validator.validate(value)
            if not result.is_valid:
                # Show error and don't dismiss
                if self._error_label:
                    self._error_label.update(
                        result.failure_descriptions[0]
                        if result.failure_descriptions
                        else "Invalid input"
                    )
                    self._error_label.display = True
                input_widget.add_class("-invalid")
                return

        self.dismiss(value)

    def action_cancel(self) -> None:
        """Handle the cancel action."""
        self.dismiss(None)

    def on_mount(self) -> None:
        """Called when the modal is mounted."""
        # Focus the input field
        input_widget = self.query_one("#input-field", Input)
        input_widget.focus()

    @classmethod
    def get_input(
        cls,
        app,
        prompt: str,
        title: str = "Input",
        placeholder: str = "",
        default_value: str = "",
        password: bool = False,
        validator: Validator | None = None,
        help_text: str = "",
        callback: Callable[..., Any] = None,
    ) -> "InputDialog":
        """
        Convenience method to show an input dialog.

        Args:
            app: The Textual app instance
            prompt: The prompt message
            title: Dialog title
            placeholder: Input placeholder
            default_value: Default input value
            password: Whether this is a password input
            validator: Optional validator
            help_text: Optional help text
            callback: Optional callback function to call with result

        Returns:
            The dialog instance
        """
        dialog = cls(
            prompt=prompt,
            title=title,
            placeholder=placeholder,
            default_value=default_value,
            password=password,
            validator=validator,
            help_text=help_text,
        )

        if callback:
            dialog.set_callback(callback)

        app.push_screen(dialog)
        return dialog
