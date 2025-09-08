"""
UI Port interface definition.

This module defines the interface for user interface operations,
including progress display and results presentation.
"""

from typing import Any

from typing_extensions import Protocol


class UIPort(Protocol):
    """
    Interface for user interface operations.

    This protocol defines the contract for UI operations, including
    progress display, results presentation, and user interaction.
    """

    def display_progress(
        self,
        progress_data: dict[str, Any],
        progress_type: str = "general",
        **kwargs: Any,
    ) -> None:
        """
        Display progress information to the user.

        Args:
            progress_data: Dictionary containing progress information
            progress_type: Type of progress to display
            **kwargs: Additional display parameters

        Progress data should contain:
            - 'current': Current progress value
            - 'total': Total progress value
            - 'message': Progress message
            - 'percentage': Optional percentage complete

        Raises:
            UIError: If progress display fails
        """
        ...

    def display_results(
        self, results: dict[str, Any], result_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display results to the user.

        Args:
            results: Dictionary containing results to display
            result_type: Type of results to display
            **kwargs: Additional display parameters

        Results data should contain:
            - 'summary': Summary of results
            - 'details': Detailed results information
            - 'success': Whether the operation was successful
            - 'metadata': Additional result metadata

        Raises:
            UIError: If results display fails
        """
        ...

    def display_error(
        self, error_message: str, error_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display error information to the user.

        Args:
            error_message: Error message to display
            error_type: Type of error
            **kwargs: Additional error display parameters

        Raises:
            UIError: If error display fails
        """
        ...

    def display_warning(
        self, warning_message: str, warning_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display warning information to the user.

        Args:
            warning_message: Warning message to display
            warning_type: Type of warning
            **kwargs: Additional warning display parameters

        Raises:
            UIError: If warning display fails
        """
        ...

    def display_info(
        self, info_message: str, info_type: str = "general", **kwargs: Any
    ) -> None:
        """
        Display informational message to the user.

        Args:
            info_message: Information message to display
            info_type: Type of information
            **kwargs: Additional info display parameters

        Raises:
            UIError: If info display fails
        """
        ...

    def get_user_input(
        self, prompt: str, input_type: str = "string", **kwargs: Any
    ) -> Any:
        """
        Get input from the user.

        Args:
            prompt: Prompt to display to the user
            input_type: Type of input expected (string, number, boolean, etc.)
            **kwargs: Additional input parameters

        Returns:
            User input value of the specified type

        Raises:
            UIError: If input collection fails
        """
        ...

    def confirm_action(
        self, message: str, default: bool = False, **kwargs: Any
    ) -> bool:
        """
        Get confirmation from the user for an action.

        Args:
            message: Message to display for confirmation
            default: Default value if user doesn't respond
            **kwargs: Additional confirmation parameters

        Returns:
            True if user confirms, False otherwise

        Raises:
            UIError: If confirmation collection fails
        """
        ...
