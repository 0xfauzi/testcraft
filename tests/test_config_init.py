"""Tests for config initialization output content."""

from pathlib import Path
import tempfile

from testcraft.cli.config_init import ConfigInitializer


class DummyUI:
    def get_user_confirmation(self, message: str, default: bool = False):
        return True

    def display_info(self, msg: str, title: str | None = None):
        pass

    def display_success(self, msg: str, title: str | None = None):
        pass

    def display_error(self, msg: str, title: str | None = None):
        pass


def test_generated_toml_includes_section_caps_and_comments():
    ui = DummyUI()
    initializer = ConfigInitializer(ui)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / ".testcraft.toml"
        path = initializer.create_config_file(format_type="toml", minimal=False, output_path=out)

        content = path.read_text(encoding="utf-8")

        # Dedicated section caps table
        assert "[generation.prompt_budgets.section_caps]" in content
        # General guidance comments
        assert "Per-section item limits used when assembling LLM context" in content
        assert "Increasing a cap shifts budget toward that section" in content
        # Presence of key caps
        assert "snippets = " in content
        assert "neighbors = " in content
        assert "path_constraints = " in content

        # Consequence comments for budgets
        assert "Consequences: Higher allows richer items" in content  # per_item_chars
        assert "Consequences: Higher overall budget" in content  # total_chars

        # Context categories consequences note
        assert "Enabling a category includes that context in prompts" in content
        assert "Turning categories off lowers cost/latency" in content


