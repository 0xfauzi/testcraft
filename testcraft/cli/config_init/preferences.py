"""User preferences collection and conversion for configuration initialization."""

from typing import Any

from ...ports.ui_port import UIPort


class PreferencesCollector:
    """Collect user preferences through interactive prompts."""

    def __init__(self, ui: UIPort):
        """Initialize with UI adapter."""
        self.ui = ui

    def collect_user_preferences(self) -> dict[str, Any]:
        """Collect user preferences through interactive prompts."""
        preferences = {}

        self.ui.print_divider("LLM Provider Setup")

        # LLM Provider selection
        providers = ["openai", "anthropic", "azure-openai", "bedrock"]
        provider = self.ui.get_user_input(
            "Select LLM provider", input_type="choice", choices=providers
        )
        preferences["llm_provider"] = provider

        # Test framework preference
        self.ui.print_divider("Test Framework Setup")

        frameworks = ["pytest", "unittest"]
        framework = self.ui.get_user_input(
            "Select test framework", input_type="choice", choices=frameworks
        )
        preferences["test_framework"] = framework

        # Coverage thresholds
        self.ui.print_divider("Coverage Configuration")

        min_coverage = self.ui.get_user_input(
            "Minimum line coverage percentage (0-100)", input_type="number", default=80
        )
        preferences["min_coverage"] = min_coverage

        # Enable advanced features
        self.ui.print_divider("Advanced Features")

        enable_refinement = self.ui.get_user_confirmation(
            "Enable AI-powered test refinement?", default=False
        )
        preferences["enable_refinement"] = enable_refinement

        enable_streaming = self.ui.get_user_confirmation(
            "Enable streaming LLM responses?", default=False
        )
        preferences["enable_streaming"] = enable_streaming

        # Failed refinement annotation preferences
        if enable_refinement:
            self.ui.print_divider("Refinement Annotation Setup")
            
            annotate_failed_tests = self.ui.get_user_confirmation(
                "Annotate test files with fix instructions when refinement fails?", default=True
            )
            preferences["annotate_failed_tests"] = annotate_failed_tests
            
            if annotate_failed_tests:
                annotation_styles = ["docstring", "hash"]
                annotation_style = self.ui.get_user_input(
                    "Annotation style", input_type="choice", choices=annotation_styles
                )
                preferences["annotation_style"] = annotation_style
                
                annotation_placements = ["top", "bottom"]
                annotation_placement = self.ui.get_user_input(
                    "Where to place annotations", input_type="choice", choices=annotation_placements
                )
                preferences["annotation_placement"] = annotation_placement

        return preferences

    def preferences_to_overrides(self, preferences: dict[str, Any]) -> dict[str, Any]:
        """Map guided setup preferences to concrete config overrides (dot-keyed)."""
        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))

        overrides: dict[str, Any] = {}

        # LLM provider and streaming
        if preferences.get("llm_provider"):
            overrides.setdefault("llm", {})["default_provider"] = preferences["llm_provider"]
        if "enable_streaming" in preferences:
            overrides.setdefault("llm", {})["enable_streaming"] = bool(preferences["enable_streaming"])

        # Generation: framework and coverage threshold
        if preferences.get("test_framework"):
            overrides.setdefault("generation", {})["test_framework"] = preferences["test_framework"]
        if preferences.get("min_coverage") is not None:
            try:
                pct = float(preferences["min_coverage"])  # 0..100
                overrides.setdefault("generation", {})["coverage_threshold"] = clamp01(pct / 100.0)
            except Exception:
                pass

        # Refinement enable and annotations
        if "enable_refinement" in preferences:
            gen = overrides.setdefault("generation", {})
            gen["enable_refinement"] = bool(preferences["enable_refinement"])
            refine = gen.setdefault("refine", {})
            refine["enable"] = bool(preferences["enable_refinement"])

        if preferences.get("annotate_failed_tests") is not None:
            overrides.setdefault("generation", {}).setdefault("refine", {})[
                "annotate_failed_tests"
            ] = bool(preferences["annotate_failed_tests"])

        if preferences.get("annotation_style"):
            overrides.setdefault("generation", {}).setdefault("refine", {})[
                "annotation_style"
            ] = preferences["annotation_style"]

        if preferences.get("annotation_placement"):
            overrides.setdefault("generation", {}).setdefault("refine", {})[
                "annotation_placement"
            ] = preferences["annotation_placement"]

        return overrides

    def flatten_overrides_dict(self, data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten a nested dict into dot-key mapping for overrides."""
        flat: dict[str, Any] = {}
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self.flatten_overrides_dict(v, key))
            else:
                flat[key] = v
        return flat
