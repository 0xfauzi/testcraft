"""
Plan Use Case for user-facing planning workflow orchestration.

This module implements the high-level planning workflow, coordinating plan
generation, user interaction, acceptance decisions, and persistence with
full edit history tracking.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from ..domain.models import ContextPack, PlanningRequest, PlanningResult
from ..ports.planning_port import PlanningPort
from ..ports.planning_presenter_port import PlanningPresenterPort
from ..ports.telemetry_port import MetricValue, SpanKind, TelemetryPort

logger = logging.getLogger(__name__)


class PlanUseCase:
    """
    Use case for planning workflow orchestration.

    Coordinates the full user-facing planning workflow including plan generation,
    user presentation, acceptance decisions, and persistence with comprehensive
    edit history tracking.
    """

    def __init__(
        self,
        planning_port: PlanningPort,
        presenter_port: PlanningPresenterPort,
        telemetry_port: TelemetryPort,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the plan use case.

        Args:
            planning_port: Port for planning operations
            presenter_port: Port for user presentation
            telemetry_port: Port for telemetry operations
            config: Optional configuration dictionary
        """
        self._planning = planning_port
        self._presenter = presenter_port
        self._telemetry = telemetry_port
        self._config = config or {}

    async def execute_planning_workflow(
        self, request: PlanningRequest, context_pack: ContextPack
    ) -> PlanningResult:
        """
        Execute full planning workflow with user interaction.

        This method coordinates the complete planning workflow:
        1. Generate plan using LLM
        2. Present to user for review
        3. Handle acceptance/rejection/edits
        4. Persist with full edit history
        5. Record telemetry

        Args:
            request: Planning request with target information
            context_pack: Context pack for the target

        Returns:
            Planning result with acceptance status and edit history

        Raises:
            PlanningError: If planning workflow fails
        """
        with self._telemetry.create_span(
            "planning_workflow",
            kind=SpanKind.INTERNAL,
            attributes={
                "target_file": str(request.target_file),
                "target_object": request.target_object,
            },
        ) as span:
            start_time = datetime.now(UTC)

            try:
                logger.info(
                    "Starting planning workflow for %s in %s",
                    request.target_object,
                    request.target_file,
                )

                # Step 1: Generate plan
                plan_option = await self._planning.generate_plan(request, context_pack)
                span.set_attribute("plan_id", plan_option.plan_id)
                span.set_attribute("model_used", plan_option.model_used)

                # Step 2: Present to user
                self._presenter.present_plan(plan_option)

                # Step 3: Handle acceptance (interactive or auto-accept)
                auto_accept = self._config.get("planning", {}).get("auto_accept", False)

                if auto_accept:
                    logger.info("Auto-accepting plan (configured)")
                    accepted = True
                    edit_text = None
                else:
                    # Interactive acceptance flow
                    accepted, edit_text = await self._presenter.prompt_acceptance()
                    logger.info(
                        "User %s plan (edited=%s)",
                        "accepted" if accepted else "rejected",
                        bool(edit_text),
                    )

                # Step 4: Build result with edit history
                edit_history = []
                if edit_text:
                    edit_history.append(
                        {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "edit_type": "user_modification",
                            "content": edit_text,
                        }
                    )

                result = PlanningResult(
                    request=request,
                    plan_option=plan_option,
                    accepted=accepted,
                    rejected=not accepted,
                    edit_history=edit_history,
                    accepted_at=datetime.now(UTC) if accepted else None,
                )

                # Step 5: Persist with edit history
                artifact_id = await self._planning.persist_plan(result)
                span.set_attribute("artifact_id", artifact_id)
                span.set_attribute("accepted", accepted)
                span.set_attribute("edit_count", len(edit_history))

                # Step 6: Record telemetry
                end_time = datetime.now(UTC)
                duration_ms = (end_time - start_time).total_seconds() * 1000

                self._telemetry.record_metrics(
                    [
                        MetricValue(
                            name="planning_duration_ms",
                            value=duration_ms,
                            unit="milliseconds",
                            labels={"model": plan_option.model_used},
                            timestamp=end_time,
                        ),
                        MetricValue(
                            name="planning_acceptance_rate",
                            value=1 if accepted else 0,
                            unit="count",
                            labels={"model": plan_option.model_used},
                            timestamp=end_time,
                        ),
                        MetricValue(
                            name="planning_edit_count",
                            value=len(edit_history),
                            unit="count",
                            labels={"model": plan_option.model_used},
                            timestamp=end_time,
                        ),
                    ]
                )

                logger.info(
                    "Planning workflow completed: plan_id=%s, accepted=%s, duration=%.2fms",
                    plan_option.plan_id[:8],
                    accepted,
                    duration_ms,
                )

                return result

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Planning workflow failed: %s", e)
                raise

    async def check_for_existing_plan(
        self, request: PlanningRequest
    ) -> PlanningResult | None:
        """
        Check if an accepted plan exists for the given request.

        This method searches for previously accepted plans that match the
        target file and object, enabling plan reuse across workflow runs.

        Args:
            request: Planning request to search for

        Returns:
            Existing planning result if found and accepted, None otherwise
        """
        try:
            logger.debug("Checking for existing plan for %s", request.target_file)

            # List accepted plans for the target file
            # request.target_file is already a Path
            plans = await self._planning.list_plans(
                target_file=request.target_file, accepted_only=True
            )

            # Find matching plan for the same target object
            for plan in plans:
                if plan.request.target_object == request.target_object:
                    logger.info(
                        "Found existing accepted plan: %s",
                        plan.plan_option.plan_id[:8],
                    )
                    return plan

            logger.debug("No existing accepted plan found")
            return None

        except Exception as e:
            logger.warning("Failed to check for existing plan: %s", e)
            return None
