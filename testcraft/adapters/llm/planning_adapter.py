"""
Planning adapter implementation wrapping LLMOrchestrator and ArtifactStoreAdapter.

This module implements the PlanningPort interface by leveraging the existing
LLM orchestrator for plan generation and the artifact store for persistence.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ...adapters.io.artifact_store import ArtifactStoreAdapter, ArtifactType
from ...application.generation.services.context_pack import ContextPackBuilder
from ...application.generation.services.llm_orchestrator import LLMOrchestrator
from ...domain.models import ContextPack, PlanningRequest, PlanningResult, PlanOption
from ...ports.planning_port import PlanningError, PlanningPort

logger = logging.getLogger(__name__)


class PlanningAdapter(PlanningPort):
    """
    Planning adapter implementing PlanningPort.

    Wraps the existing LLMOrchestrator for plan generation and uses
    ArtifactStoreAdapter for plan persistence with full edit history.
    """

    def __init__(
        self,
        llm_orchestrator: LLMOrchestrator,
        artifact_store: ArtifactStoreAdapter,
        context_pack_builder: ContextPackBuilder | None = None,
    ):
        """
        Initialize the planning adapter.

        Args:
            llm_orchestrator: LLM orchestrator for plan generation
            artifact_store: Artifact store for plan persistence
            context_pack_builder: Optional context pack builder
        """
        self._orchestrator = llm_orchestrator
        self._artifact_store = artifact_store
        self._context_builder = context_pack_builder

    async def generate_plan(
        self, request: PlanningRequest, context_pack: ContextPack
    ) -> PlanOption:
        """
        Generate a plan using LLMOrchestrator.

        Args:
            request: Planning request with target information
            context_pack: Context pack for the target

        Returns:
            Generated plan option with hash and metadata

        Raises:
            PlanningError: If plan generation fails
        """
        try:
            logger.info(
                "Generating plan for %s in %s",
                request.target_object,
                request.target_file,
            )

            # Call LLMOrchestrator.plan_stage()
            # request.project_root is already Path | None
            plan_dict = self._orchestrator.plan_stage(
                context_pack, request.project_root
            )

            # Create PlanOption with hash
            plan_content_json = json.dumps(plan_dict, sort_keys=True)
            plan_id = hashlib.sha256(plan_content_json.encode()).hexdigest()

            # Get model name from orchestrator's LLM port
            model_used = "unknown"
            llm_adapter = getattr(self._orchestrator, "_llm", None)
            if llm_adapter is not None:
                model_used = getattr(
                    llm_adapter,
                    "model_name",
                    getattr(llm_adapter, "model", "unknown"),
                )
                if (not model_used or model_used == "unknown") and hasattr(
                    llm_adapter, "_get_adapter"
                ):
                    try:
                        provider_adapter = llm_adapter._get_adapter(None)
                        model_used = getattr(
                            provider_adapter,
                            "model_name",
                            getattr(provider_adapter, "model", model_used),
                        )
                    except Exception:
                        pass

            plan_option = PlanOption(
                plan_id=plan_id,
                plan_content=plan_dict,
                model_used=model_used,
                created_at=datetime.now(UTC),
            )

            logger.debug("Generated plan %s using model %s", plan_id[:8], model_used)
            return plan_option

        except Exception as e:
            logger.exception("Failed to generate plan: %s", e)
            raise PlanningError(f"Plan generation failed: {e}") from e

    async def persist_plan(self, result: PlanningResult) -> str:
        """
        Persist plan with acceptance status and edit history.

        Args:
            result: Planning result to persist

        Returns:
            Artifact ID for the persisted plan

        Raises:
            PlanningError: If persistence fails
        """
        try:
            logger.info(
                "Persisting plan %s (accepted=%s)",
                result.plan_option.plan_id[:8],
                result.accepted,
            )

            # Prepare artifact data with all fields
            artifact_data = {
                "request": {
                    "target_file": str(result.request.target_file),
                    "target_object": result.request.target_object,
                    "project_root": str(result.request.project_root)
                    if result.request.project_root
                    else None,
                    "prompt_customization": result.request.prompt_customization,
                },
                "plan_option": {
                    "plan_id": result.plan_option.plan_id,
                    "plan_content": result.plan_option.plan_content,
                    "model_used": result.plan_option.model_used,
                    "created_at": result.plan_option.created_at.isoformat(),
                },
                "accepted": result.accepted,
                "rejected": result.rejected,
                "edit_history": result.edit_history,
                "accepted_at": result.accepted_at.isoformat()
                if result.accepted_at
                else None,
            }

            # Store in artifact store with GENERATION_PLAN type
            artifact_id = self._artifact_store.store_artifact(
                ArtifactType.GENERATION_PLAN,
                content=artifact_data,
                tags=["planning", "accepted" if result.accepted else "pending"],
                description=f"Plan for {result.request.target_object}",
            )

            logger.debug("Persisted plan with artifact ID: %s", artifact_id)
            return artifact_id

        except Exception as e:
            logger.exception("Failed to persist plan: %s", e)
            raise PlanningError(f"Plan persistence failed: {e}") from e

    async def retrieve_plan(self, plan_id: str) -> PlanningResult | None:
        """
        Retrieve a persisted plan by ID.

        Args:
            plan_id: SHA-256 hash of plan content

        Returns:
            Planning result if found, None otherwise

        Raises:
            PlanningError: If retrieval fails
        """
        try:
            logger.debug("Retrieving plan %s", plan_id[:8])

            # List all plans and find matching plan_id
            all_artifacts = self._artifact_store.list_artifacts(
                artifact_type=ArtifactType.GENERATION_PLAN
            )

            for artifact_meta in all_artifacts:
                # Retrieve full artifact
                artifact = self._artifact_store.retrieve_artifact(
                    artifact_meta["artifact_id"]
                )
                if artifact:
                    content = artifact["content"]
                    if content.get("plan_option", {}).get("plan_id") == plan_id:
                        # Reconstruct PlanningResult
                        return self._reconstruct_planning_result(content)

            logger.debug("Plan %s not found", plan_id[:8])
            return None

        except Exception as e:
            logger.exception("Failed to retrieve plan: %s", e)
            raise PlanningError(f"Plan retrieval failed: {e}") from e

    async def list_plans(
        self, target_file: Path | None = None, accepted_only: bool = False
    ) -> list[PlanningResult]:
        """
        List plans with optional filtering.

        Args:
            target_file: Optional filter by target file path
            accepted_only: If True, only return accepted plans

        Returns:
            List of planning results matching filters

        Raises:
            PlanningError: If listing fails
        """
        try:
            logger.debug(
                "Listing plans (target_file=%s, accepted_only=%s)",
                target_file,
                accepted_only,
            )

            # Get all planning artifacts
            tags = ["planning"]
            if accepted_only:
                tags.append("accepted")

            artifacts = self._artifact_store.list_artifacts(
                artifact_type=ArtifactType.GENERATION_PLAN,
                tags=tags if accepted_only else None,
            )

            results = []
            for artifact_meta in artifacts:
                # Retrieve full artifact
                artifact = self._artifact_store.retrieve_artifact(
                    artifact_meta["artifact_id"]
                )
                if artifact:
                    content = artifact["content"]

                    # Apply target_file filter if specified
                    if target_file:
                        request_target = content.get("request", {}).get("target_file")
                        if request_target != str(target_file):
                            continue

                    # Apply accepted_only filter if not using tags
                    if accepted_only and not content.get("accepted", False):
                        continue

                    # Reconstruct PlanningResult
                    result = self._reconstruct_planning_result(content)
                    if result:
                        results.append(result)

            logger.debug("Found %d plans matching filters", len(results))
            return results

        except Exception as e:
            logger.exception("Failed to list plans: %s", e)
            raise PlanningError(f"Plan listing failed: {e}") from e

    def _reconstruct_planning_result(
        self, artifact_data: dict
    ) -> PlanningResult | None:
        """
        Reconstruct PlanningResult from artifact data.

        Args:
            artifact_data: Artifact data dictionary

        Returns:
            PlanningResult if reconstruction succeeds, None otherwise
        """
        try:
            from ...domain.models import PlanningRequest

            # Reconstruct request
            request_data = artifact_data.get("request", {})
            request = PlanningRequest(
                target_file=Path(request_data.get("target_file", "")),
                target_object=request_data.get("target_object", ""),
                project_root=Path(request_data["project_root"])
                if request_data.get("project_root")
                else None,
                prompt_customization=request_data.get("prompt_customization"),
            )

            # Reconstruct plan option
            plan_data = artifact_data.get("plan_option", {})

            # Parse created_at and ensure it's timezone-aware
            created_at_str = plan_data.get("created_at", datetime.now(UTC).isoformat())
            created_at = datetime.fromisoformat(created_at_str)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)

            plan_option = PlanOption(
                plan_id=plan_data.get("plan_id", ""),
                plan_content=plan_data.get("plan_content", {}),
                model_used=plan_data.get("model_used", "unknown"),
                created_at=created_at,
            )

            # Reconstruct planning result with timezone-aware accepted_at
            accepted_at = None
            if artifact_data.get("accepted_at"):
                accepted_at = datetime.fromisoformat(artifact_data["accepted_at"])
                if accepted_at.tzinfo is None:
                    accepted_at = accepted_at.replace(tzinfo=UTC)

            result = PlanningResult(
                request=request,
                plan_option=plan_option,
                accepted=artifact_data.get("accepted", False),
                rejected=artifact_data.get("rejected", False),
                edit_history=artifact_data.get("edit_history", []),
                accepted_at=accepted_at,
            )

            return result

        except Exception as e:
            logger.warning("Failed to reconstruct planning result: %s", e)
            return None
