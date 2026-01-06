from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from testcraft.domain.models import (
    AnalysisReport,
    Budget,
    GenerationResult,
    PlanningRequest,
    PlanningResult,
    PlanOption,
)


def test_generation_result_requires_error_message() -> None:
    with pytest.raises(ValueError):
        GenerationResult(
            file_path="tests/sample.py",
            content=None,
            success=False,
            error_message=None,
        )


def test_generation_result_with_error_message_succeeds() -> None:
    result = GenerationResult(
        file_path="tests/sample.py",
        content=None,
        success=False,
        error_message="Something went wrong",
    )
    assert result.error_message == "Something went wrong"


def test_planning_result_requires_acceptance_timestamp() -> None:
    plan_option = PlanOption(
        plan_id="abc123",
        plan_content={"plan": []},
        model_used="gpt-test",
        created_at=datetime.now(UTC),
    )
    planning_request = PlanningRequest(
        target_file=Path("tests/sample.py"),
        target_object="sample",
        project_root=None,
        prompt_customization=None,
    )

    with pytest.raises(ValueError):
        PlanningResult(
            request=planning_request,
            plan_option=plan_option,
            accepted=True,
            rejected=False,
            edit_history=[],
            accepted_at=None,
        )


def test_analysis_report_requires_reason_for_files() -> None:
    with pytest.raises(ValueError):
        AnalysisReport(
            files_to_process=["src/app.py"],
            reasons={},
            existing_test_presence={"src/app.py": True},
        )


def test_analysis_report_requires_test_presence_for_files() -> None:
    with pytest.raises(ValueError):
        AnalysisReport(
            files_to_process=["src/app.py"],
            reasons={"src/app.py": "Needs coverage"},
            existing_test_presence={},
        )


def test_budget_requires_positive_value() -> None:
    with pytest.raises(ValueError):
        Budget(max_input_tokens=0)
