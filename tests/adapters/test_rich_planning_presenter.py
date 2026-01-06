from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from io import StringIO

import pytest
from rich.console import Console

from testcraft.adapters.presenters.rich_planning_presenter import RichPlanningPresenter
from testcraft.domain.models import PlanOption


def _make_presenter(plan_scenarios: list[dict[str, str]]) -> RichPlanningPresenter:
    """Create a presenter with deterministic console and plan data."""
    presenter = RichPlanningPresenter()
    presenter._console = Console(
        file=StringIO(),
        force_terminal=False,
        color_system=None,
    )

    plan_option = PlanOption(
        plan_id="plan-id",
        plan_content={"plan": plan_scenarios},
        model_used="gpt-test",
        created_at=datetime.now(UTC),
    )
    presenter.present_plan(plan_option)
    return presenter


@pytest.mark.asyncio
async def test_collect_edit_text_with_scenario_selection() -> None:
    plan_scenarios = [
        {"id": "scenario_one", "intent": "Scenario one intent"},
        {"id": "scenario_two", "intent": "Scenario two intent"},
        {"id": "scenario_three", "intent": "Scenario three intent"},
        {"id": "scenario_four", "intent": "Scenario four intent"},
        {"id": "get_schedule_status_returns_table", "intent": "Status intent"},
    ]
    presenter = _make_presenter(plan_scenarios)

    responses: Iterator[str] = iter(
        [
            "5",  # select fifth scenario
            "get_schedule_status test should not test scenario with a Table with correct job info",
            "",
        ]
    )

    async def fake_ask(_: str) -> str:
        try:
            return next(responses)
        except (
            StopIteration
        ) as exc:  # pragma: no cover - guard against unexpected prompts
            raise AssertionError("Unexpected prompt for _ask") from exc

    presenter._ask = fake_ask  # type: ignore[assignment]

    edit_text = await presenter._collect_edit_text()

    assert (
        edit_text == "Scenario 5 (get_schedule_status_returns_table)\n\n"
        "get_schedule_status test should not test scenario with a Table with correct job info"
    )


@pytest.mark.asyncio
async def test_collect_edit_text_general_feedback() -> None:
    plan_scenarios = [
        {"id": "scenario_one", "intent": "Scenario one intent"},
    ]
    presenter = _make_presenter(plan_scenarios)

    responses: Iterator[str] = iter(
        [
            "",  # general feedback (no selection)
            "Overall plan looks correct",
            "",
        ]
    )

    async def fake_ask(_: str) -> str:
        try:
            return next(responses)
        except (
            StopIteration
        ) as exc:  # pragma: no cover - guard against unexpected prompts
            raise AssertionError("Unexpected prompt for _ask") from exc

    presenter._ask = fake_ask  # type: ignore[assignment]

    edit_text = await presenter._collect_edit_text()

    assert edit_text == "Overall plan looks correct"
