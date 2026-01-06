"""
CLI presenter adapter for planning workflow.

This module implements the PlanningPresenterPort for command-line interface,
providing interactive plan review and acceptance prompts using Rich formatting.
"""

import asyncio
import logging

import click
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from ...domain.models import PlanOption
from ...ports.planning_presenter_port import PlanningPresenterPort, PresentationError

logger = logging.getLogger(__name__)


class PlanningCliPresenter(PlanningPresenterPort):
    """
    CLI presenter for planning workflow.

    Implements PlanningPresenterPort to provide interactive plan review
    and acceptance prompts in the command-line interface using Rich
    for formatted output.
    """

    def __init__(self, console: Console | None = None):
        """
        Initialize the CLI presenter.

        Args:
            console: Optional Rich console for output (creates default if None)
        """
        self._console = console or Console()

    def present_plan(self, plan_option: PlanOption) -> None:
        """
        Display plan to user for review using Rich formatting.

        Args:
            plan_option: Plan option to present

        Raises:
            PresentationError: If display fails
        """
        try:
            logger.debug("Presenting plan %s to user", plan_option.plan_id[:8])

            # Display plan header
            self._console.print()

            # Format created_at datetime
            created_display = plan_option.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

            self._console.print(
                Panel(
                    f"[bold cyan]Generated Test Plan[/]\n"
                    f"[dim]Plan ID:[/] {plan_option.plan_id[:12]}...\n"
                    f"[dim]Model:[/] {plan_option.model_used}\n"
                    f"[dim]Created:[/] {created_display}",
                    border_style="cyan",
                )
            )

            plan_content = plan_option.plan_content or {}
            scenarios = plan_content.get("plan", [])

            plan_tree = Tree("[bold yellow]Scenarios[/]")

            missing_symbols = plan_content.get("missing_symbols") or []
            if missing_symbols:
                missing_branch = plan_tree.add("[red]Missing Symbols[/]")
                for symbol in missing_symbols:
                    missing_branch.add(f"[red]- {symbol}")

            for index, scenario in enumerate(scenarios, start=1):
                scenario_id = scenario.get("id", f"scenario_{index}")
                intent = scenario.get("intent", "")
                branch = plan_tree.add(
                    f"[bold]{index}. {scenario_id}[/]\n[dim]{intent}[/]"
                )

                def add_section(
                    label: str, entries: list[str] | None, style: str
                ) -> None:
                    if entries:
                        section = branch.add(f"[{style}]{label}[/]")
                        for entry in entries:
                            section.add(f"[{style}]- {entry}")

                add_section("Inputs", scenario.get("inputs"), "cyan")
                add_section("Assertions", scenario.get("assertions"), "magenta")
                add_section("Fixtures", scenario.get("fixtures"), "green")
                add_section("Edge Cases", scenario.get("edge_cases"), "yellow")

                notes = scenario.get("notes")
                if notes:
                    branch.add(Panel(notes, title="Notes", border_style="blue"))

            self._console.print()
            self._console.print(
                Panel(plan_tree, border_style="cyan", title="Plan Details")
            )
            self._console.print()
            self._console.print(
                "[cyan]Review the plan above. You can accept, reject, or edit it when prompted below.[/]\n"
            )

        except Exception as e:
            logger.exception("Failed to present plan: %s", e)
            raise PresentationError(f"Failed to present plan: {e}") from e

    async def prompt_acceptance(self) -> tuple[bool, str | None]:
        return await asyncio.to_thread(self._prompt_acceptance_sync)

    def _prompt_acceptance_sync(self) -> tuple[bool, str | None]:
        """
        Prompt user to accept/reject/edit plan interactively.

        Returns:
            Tuple of (accepted, edit_text) where:
            - accepted: True if user accepts the plan
            - edit_text: Optional user modifications (included in edit history)

        Raises:
            PresentationError: If prompt fails
        """
        try:
            logger.debug("Prompting user for plan acceptance")

            # Interactive prompt with choices
            response = (
                click.prompt(
                    "[a]ccept / [r]eject / [e]dit",
                    type=click.Choice(["a", "r", "e"], case_sensitive=False),
                    show_choices=False,
                )
                .strip()
                .lower()
            )

            if response == "a":
                self._console.print("[green]✓[/] Plan accepted", style="bold")
                return True, None
            elif response == "r":
                self._console.print("[red]✗[/] Plan rejected", style="bold")
                return False, None
            else:  # edit
                self._console.print(
                    "[yellow]✎[/] Enter plan modifications (press Ctrl+D or Ctrl+Z when done):",
                    style="bold",
                )

                # Collect multi-line edit text
                edit_lines = []
                try:
                    while True:
                        line = input()
                        edit_lines.append(line)
                except EOFError:
                    pass

                edit_text = "\n".join(edit_lines).strip()

                if not edit_text:
                    self._console.print(
                        "[yellow]⚠[/] No modifications provided, rejecting plan",
                        style="dim",
                    )
                    return False, None

                # Re-prompt for acceptance after edit
                self._console.print()
                accept_modified = click.confirm(
                    "Accept plan with modifications?", default=True
                )

                if accept_modified:
                    self._console.print(
                        "[green]✓[/] Modified plan accepted", style="bold"
                    )
                    return True, edit_text
                else:
                    self._console.print(
                        "[red]✗[/] Modified plan rejected", style="bold"
                    )
                    return False, edit_text

        except KeyboardInterrupt:
            self._console.print()
            self._console.print(
                "[yellow]⚠[/] Plan review interrupted, rejecting plan", style="dim"
            )
            return False, None
        except Exception as e:
            logger.exception("Failed to prompt for acceptance: %s", e)
            raise PresentationError(f"Failed to prompt for acceptance: {e}") from e
