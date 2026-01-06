"""
Interactive plan review presenter using Rich components only.

Provides a terminal-driven workflow for inspecting generated plans and
collecting approval decisions without relying on Textual widgets. This keeps
the experience compatible in constrained environments where full TUI stacks
may not be available.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...domain.models import PlanOption
from ...ports.planning_presenter_port import PlanningPresenterPort, PresentationError


def _build_scenario_panel(index: int, scenario: dict) -> Panel:
    """Render a single scenario as a rich Panel."""
    header = f"{index}. {scenario.get('id', f'scenario_{index}')}"
    intent = scenario.get("intent", "No intent provided.")

    sections: list[str] = [intent]

    def add_section(label: str, entries: Sequence[str] | None) -> None:
        if entries:
            formatted = "\n".join(f"- {entry}" for entry in entries)
            sections.append(f"[bold]{label}[/]\n{formatted}")

    add_section("Inputs", scenario.get("inputs"))
    add_section("Assertions", scenario.get("assertions"))
    add_section("Fixtures", scenario.get("fixtures"))
    add_section("Edge Cases", scenario.get("edge_cases"))

    notes = scenario.get("notes")
    if notes:
        sections.append(f"[bold blue]Notes[/]\n{notes}")

    body = "\n\n".join(sections)
    return Panel(body, title=header, border_style="cyan")


class RichPlanningPresenter(PlanningPresenterPort):
    """Planning presenter that renders plans with Rich layout components."""

    def __init__(self) -> None:
        self._console = Console()
        self._plan_option: PlanOption | None = None

    def present_plan(self, plan_option: PlanOption) -> None:
        self._plan_option = plan_option

    async def prompt_acceptance(self) -> tuple[bool, str | None]:
        if self._plan_option is None:
            raise PresentationError("No plan available to present")

        self._render_plan(self._plan_option)

        while True:
            action = (
                (
                    await self._ask(
                        "[bold green]A[/]ccept, [bold red]R[/]eject, [bold yellow]E[/]dit: "
                    )
                )
                .strip()
                .lower()
            )

            if action in {"a", "accept"}:
                return True, None
            if action in {"r", "reject"}:
                return False, None
            if action in {"e", "edit"}:
                edit_text = await self._collect_edit_text()
                if edit_text is None:
                    continue

                decision = (
                    (await self._ask("Apply edits and [A]ccept or [R]eject? "))
                    .strip()
                    .lower()
                )

                if decision.startswith("a"):
                    return True, edit_text
                if decision.startswith("r"):
                    return False, edit_text

                self._console.print("[red]Invalid decision. Returning to main menu.[/]")
                continue

            self._console.print(
                "[red]Unrecognized command. Please choose A, R, or E.[/]"
            )

    def _render_plan(self, plan_option: PlanOption) -> None:
        """Display plan metadata and scenario details."""
        self._console.clear()

        title = Text("Generated Test Plan", style="bold cyan")
        metadata = Table(show_header=False, box=None)
        metadata.add_row("Plan ID", plan_option.plan_id)
        metadata.add_row("Model", plan_option.model_used)
        metadata.add_row(
            "Import",
            plan_option.plan_content.get("import_line", "N/A"),
        )

        self._console.print(Panel(metadata, title=title, border_style="cyan"))

        missing_symbols = plan_option.plan_content.get("missing_symbols") or []
        if missing_symbols:
            missing = "\n".join(f"- {symbol}" for symbol in missing_symbols)
            self._console.print(
                Panel(missing, title="Missing Symbols", border_style="red")
            )

        scenarios = plan_option.plan_content.get("plan") or []
        if not scenarios:
            self._console.print("[yellow]No scenarios were provided in this plan.[/]")
        else:
            for index, scenario in enumerate(scenarios, start=1):
                self._console.print(_build_scenario_panel(index, scenario))
                self._console.print()  # spacer for readability

        instructions = (
            "[bold]Controls[/]: "
            "[green]A[/] Accept  |  "
            "[red]R[/] Reject  |  "
            "[yellow]E[/] Edit with feedback"
        )
        self._console.print(Panel(instructions, border_style="green"))

    async def _collect_edit_text(self) -> str | None:
        """Capture multi-line edit notes from the user."""
        scenarios = (
            self._plan_option.plan_content.get("plan") if self._plan_option else None
        ) or []

        selected_indices: list[int] = []
        if scenarios:
            self._console.print(
                "\n[bold]Select the plan item(s) you want to give feedback on.[/]",
            )
            for index, scenario in enumerate(scenarios, start=1):
                scenario_id = scenario.get("id", f"scenario_{index}")
                intent = scenario.get("intent") or ""
                intent_suffix = f" — {intent}" if intent else ""
                self._console.print(f"[dim]{index}. {scenario_id}{intent_suffix}[/dim]")

            while True:
                selection = (
                    await self._ask(
                        "\nEnter item numbers (comma-separated), 'all', or press Enter for general feedback: "
                    )
                ).strip()

                if not selection:
                    break

                if selection.lower() in {"all", "*"}:
                    selected_indices = list(range(1, len(scenarios) + 1))
                    break

                try:
                    indices = []
                    for part in selection.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        idx = int(part)
                        if idx < 1 or idx > len(scenarios):
                            raise ValueError
                        indices.append(idx)

                    if not indices:
                        raise ValueError

                    selected_indices = sorted(set(indices))
                    break
                except ValueError:
                    self._console.print(
                        "[red]Invalid selection. Use numbers from the list above.[/]"
                    )
                    continue

        if selected_indices:
            descriptor_lines = []
            for idx in selected_indices:
                scenario = scenarios[idx - 1]
                scenario_id = scenario.get("id", f"scenario_{idx}")
                descriptor_lines.append(f"Scenario {idx} ({scenario_id})")

            self._console.print(
                f"\n[bold green]Editing:[/] {', '.join(descriptor_lines)}"
            )
        else:
            self._console.print("\n[bold green]Editing:[/] General feedback")

        self._console.print(
            "Enter edit notes. Submit an empty line to finish, or press Enter immediately to cancel.",
            style="bold yellow",
        )

        lines: list[str] = []
        while True:
            line = await self._ask("> ")
            if not line.strip() and not lines:
                self._console.print(
                    "[dim]No edits captured. Returning to main menu.[/]"
                )
                return None
            if line == "":
                break
            lines.append(line)

        edit_text = "\n".join(lines).strip()
        if not edit_text:
            self._console.print("[dim]No edits captured. Returning to main menu.[/]")
            return None

        if selected_indices:
            descriptor_lines = []
            for idx in selected_indices:
                scenario = scenarios[idx - 1]
                scenario_id = scenario.get("id", f"scenario_{idx}")
                descriptor_lines.append(f"Scenario {idx} ({scenario_id})")

            header = "\n".join(descriptor_lines)
            edit_text = f"{header}\n\n{edit_text}"

        return edit_text

    async def _ask(self, prompt: str) -> str:
        """Async wrapper around Console.input to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._console.input, prompt)
