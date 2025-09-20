"""
Theme management for TestCraft Textual UI.

This module provides:
- Multiple built-in themes
- Custom theme support
- Dynamic theme switching
- Color scheme management
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ColorScheme:
    """Represents a color scheme for the UI."""

    # Core colors
    primary: str
    secondary: str
    background: str
    surface: str
    error: str
    warning: str
    success: str
    info: str

    # Text colors
    text: str
    text_muted: str
    text_disabled: str
    text_on_primary: str
    text_on_secondary: str
    text_on_error: str

    # Border colors
    border: str
    border_focused: str
    border_error: str
    border_success: str

    # Special colors
    accent: str
    highlight: str
    shadow: str
    overlay: str


class ThemeManager:
    """
    Manages application themes and color schemes.

    Provides built-in themes and support for custom themes,
    with dynamic switching capabilities.
    """

    # Built-in themes
    THEMES = {
        "default": ColorScheme(
            primary="#007ACC",
            secondary="#5A6370",
            background="#1E1E1E",
            surface="#252526",
            error="#F44747",
            warning="#FFB86C",
            success="#50FA7B",
            info="#8BE9FD",
            text="#CCCCCC",
            text_muted="#808080",
            text_disabled="#4D4D4D",
            text_on_primary="#FFFFFF",
            text_on_secondary="#FFFFFF",
            text_on_error="#FFFFFF",
            border="#3C3C3C",
            border_focused="#007ACC",
            border_error="#F44747",
            border_success="#50FA7B",
            accent="#00D9FF",
            highlight="#264F78",
            shadow="#000000",
            overlay="#00000080",
        ),
        "high_contrast": ColorScheme(
            primary="#00FFFF",
            secondary="#FFFF00",
            background="#000000",
            surface="#0C0C0C",
            error="#FF0000",
            warning="#FFA500",
            success="#00FF00",
            info="#00BFFF",
            text="#FFFFFF",
            text_muted="#C0C0C0",
            text_disabled="#808080",
            text_on_primary="#000000",
            text_on_secondary="#000000",
            text_on_error="#FFFFFF",
            border="#FFFFFF",
            border_focused="#00FFFF",
            border_error="#FF0000",
            border_success="#00FF00",
            accent="#FF00FF",
            highlight="#FFFF00",
            shadow="#000000",
            overlay="#00000080",
        ),
        "solarized_dark": ColorScheme(
            primary="#268BD2",
            secondary="#2AA198",
            background="#002B36",
            surface="#073642",
            error="#DC322F",
            warning="#CB4B16",
            success="#859900",
            info="#B58900",
            text="#839496",
            text_muted="#586E75",
            text_disabled="#073642",
            text_on_primary="#FDF6E3",
            text_on_secondary="#FDF6E3",
            text_on_error="#FDF6E3",
            border="#073642",
            border_focused="#268BD2",
            border_error="#DC322F",
            border_success="#859900",
            accent="#D33682",
            highlight="#073642",
            shadow="#000000",
            overlay="#00000080",
        ),
        "solarized_light": ColorScheme(
            primary="#268BD2",
            secondary="#2AA198",
            background="#FDF6E3",
            surface="#EEE8D5",
            error="#DC322F",
            warning="#CB4B16",
            success="#859900",
            info="#B58900",
            text="#657B83",
            text_muted="#93A1A1",
            text_disabled="#EEE8D5",
            text_on_primary="#FDF6E3",
            text_on_secondary="#FDF6E3",
            text_on_error="#FDF6E3",
            border="#EEE8D5",
            border_focused="#268BD2",
            border_error="#DC322F",
            border_success="#859900",
            accent="#D33682",
            highlight="#EEE8D5",
            shadow="#657B83",
            overlay="#00000040",
        ),
        "monokai": ColorScheme(
            primary="#66D9EF",
            secondary="#A6E22E",
            background="#272822",
            surface="#3E3D32",
            error="#F92672",
            warning="#FD971F",
            success="#A6E22E",
            info="#66D9EF",
            text="#F8F8F2",
            text_muted="#75715E",
            text_disabled="#3E3D32",
            text_on_primary="#272822",
            text_on_secondary="#272822",
            text_on_error="#F8F8F2",
            border="#3E3D32",
            border_focused="#66D9EF",
            border_error="#F92672",
            border_success="#A6E22E",
            accent="#AE81FF",
            highlight="#49483E",
            shadow="#000000",
            overlay="#00000080",
        ),
        "dracula": ColorScheme(
            primary="#BD93F9",
            secondary="#50FA7B",
            background="#282A36",
            surface="#44475A",
            error="#FF5555",
            warning="#FFB86C",
            success="#50FA7B",
            info="#8BE9FD",
            text="#F8F8F2",
            text_muted="#6272A4",
            text_disabled="#44475A",
            text_on_primary="#282A36",
            text_on_secondary="#282A36",
            text_on_error="#F8F8F2",
            border="#44475A",
            border_focused="#BD93F9",
            border_error="#FF5555",
            border_success="#50FA7B",
            accent="#FF79C6",
            highlight="#44475A",
            shadow="#191A21",
            overlay="#00000080",
        ),
    }

    def __init__(self, initial_theme: str = "default") -> None:
        """
        Initialize the theme manager.

        Args:
            initial_theme: Name of the initial theme to use
        """
        self.current_theme_name = initial_theme
        self.current_theme = self.THEMES.get(initial_theme, self.THEMES["default"])
        self.custom_themes: dict[str, ColorScheme] = {}
        self._css_cache: str | None = None

    def get_current_theme(self) -> ColorScheme:
        """Get the current theme."""
        return self.current_theme

    def get_current_theme_name(self) -> str:
        """Get the name of the current theme."""
        return self.current_theme_name

    def set_theme(self, theme_name: str) -> bool:
        """
        Apply a theme to the application.

        Args:
            theme_name: Name of the theme to apply

        Returns:
            True if theme was applied successfully
        """
        # Check built-in themes
        if theme_name in self.THEMES:
            self.current_theme = self.THEMES[theme_name]
            self.current_theme_name = theme_name
            self._css_cache = None
            return True

        # Check custom themes
        if theme_name in self.custom_themes:
            self.current_theme = self.custom_themes[theme_name]
            self.current_theme_name = theme_name
            self._css_cache = None
            return True

        return False

    def register_custom_theme(self, name: str, theme: ColorScheme) -> None:
        """
        Register a custom theme.

        Args:
            name: Name for the custom theme
            theme: ColorScheme object
        """
        self.custom_themes[name] = theme

    def load_theme_from_file(self, path: Path) -> bool:
        """
        Load a theme from a JSON file.

        Args:
            path: Path to the theme file

        Returns:
            True if theme was loaded successfully
        """
        try:
            with open(path) as f:
                data = json.load(f)

            theme = ColorScheme(**data["colors"])
            self.register_custom_theme(data["name"], theme)
            return True
        except Exception as e:
            print(f"Error loading theme from {path}: {e}")
            return False

    def save_theme_to_file(self, name: str, path: Path) -> bool:
        """
        Save a theme to a JSON file.

        Args:
            name: Name of the theme to save
            path: Path to save the theme file

        Returns:
            True if theme was saved successfully
        """
        theme = None

        if name in self.THEMES:
            theme = self.THEMES[name]
        elif name in self.custom_themes:
            theme = self.custom_themes[name]

        if not theme:
            return False

        try:
            data = {
                "name": name,
                "colors": {
                    field: getattr(theme, field) for field in theme.__dataclass_fields__
                },
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving theme to {path}: {e}")
            return False

    def get_available_themes(self) -> list[str]:
        """Get list of all available theme names."""
        return list(self.THEMES.keys()) + list(self.custom_themes.keys())

    def generate_css(self) -> str:
        """
        Generate CSS variables for the current theme.

        Returns:
            CSS string with theme variables
        """
        if self._css_cache:
            return self._css_cache

        theme = self.current_theme

        css = """
        /* TestCraft Theme Variables */
        :root {
            /* Core colors */
            --primary: %(primary)s;
            --secondary: %(secondary)s;
            --background: %(background)s;
            --surface: %(surface)s;
            --error: %(error)s;
            --warning: %(warning)s;
            --success: %(success)s;
            --info: %(info)s;

            /* Text colors */
            --text: %(text)s;
            --text-muted: %(text_muted)s;
            --text-disabled: %(text_disabled)s;
            --text-on-primary: %(text_on_primary)s;
            --text-on-secondary: %(text_on_secondary)s;
            --text-on-error: %(text_on_error)s;

            /* Border colors */
            --border: %(border)s;
            --border-focused: %(border_focused)s;
            --border-error: %(border_error)s;
            --border-success: %(border_success)s;

            /* Special colors */
            --accent: %(accent)s;
            --highlight: %(highlight)s;
            --shadow: %(shadow)s;
            --overlay: %(overlay)s;
        }

        /* Textual-specific theme mappings */
        App {
            background: var(--background);
            color: var(--text);
        }

        Screen {
            background: var(--background);
        }

        /* Widget defaults */
        Button {
            background: var(--surface);
            color: var(--text);
            border: tall var(--border);
        }

        Button:hover {
            background: var(--primary);
            color: var(--text-on-primary);
            border: tall var(--border-focused);
        }

        Button:focus {
            border: tall var(--border-focused);
        }

        Button.-primary {
            background: var(--primary);
            color: var(--text-on-primary);
        }

        Button.-error {
            background: var(--error);
            color: var(--text-on-error);
        }

        Button.-success {
            background: var(--success);
            color: var(--text-on-primary);
        }

        Input {
            background: var(--surface);
            color: var(--text);
            border: tall var(--border);
        }

        Input:focus {
            border: tall var(--border-focused);
        }

        Input.-invalid {
            border: tall var(--border-error);
        }

        /* Modal styling */
        ModalScreen {
            background: var(--overlay);
        }

        .modal-container {
            background: var(--surface);
            border: thick var(--border-focused);
        }
        """ % {field: getattr(theme, field) for field in theme.__dataclass_fields__}

        self._css_cache = css
        return css

    def get_color(self, color_name: str) -> str:
        """
        Get a specific color from the current theme.

        Args:
            color_name: Name of the color to get

        Returns:
            Color value or empty string if not found
        """
        return getattr(self.current_theme, color_name, "")

    def create_variant_css(
        self, widget_name: str, variant: str, colors: dict[str, str]
    ) -> str:
        """
        Create CSS for a widget variant.

        Args:
            widget_name: Name of the widget
            variant: Variant name
            colors: Dictionary of color overrides

        Returns:
            CSS string for the variant
        """
        css_rules = []

        for prop, color in colors.items():
            # Map common properties
            css_prop = {
                "background": "background",
                "text": "color",
                "border": "border",
            }.get(prop, prop)

            css_rules.append(f"    {css_prop}: {color};")

        return f"{widget_name}.-{variant} {{\n" + "\n".join(css_rules) + "\n}"
