"""Dependency injection container for CLI commands."""

import logging
import sys
from typing import Any

from ..adapters.context.main_adapter import TestcraftContextAdapter

# Import adapters
from ..adapters.io.file_discovery import FileDiscoveryService
from ..adapters.io.state_json import StateJsonAdapter
from ..adapters.io.writer_ast_merge import WriterASTMergeAdapter
from ..adapters.llm.router import LLMRouter
from ..adapters.parsing.codebase_parser import CodebaseParser
from ..adapters.refine.main_adapter import RefineAdapter
from ..adapters.telemetry.cost_manager import CostManager
from ..adapters.telemetry.noop_adapter import NoOpTelemetryAdapter
from ..adapters.telemetry.router import create_telemetry_adapter
from ..application.analyze_usecase import AnalyzeUseCase
from ..application.coverage_usecase import CoverageUseCase
from ..application.generate_usecase import GenerateUseCase
from ..application.status_usecase import StatusUseCase
from ..application.utility_usecases import UtilityUseCase
from ..config.models import TestCraftConfig

logger = logging.getLogger(__name__)


class DependencyError(Exception):
    """Raised when dependency injection fails."""

    pass


def create_dependency_container(config: TestCraftConfig) -> dict[str, Any]:
    """
    Create a dependency injection container with all required services.

    Args:
        config: TestCraft configuration

    Returns:
        Dictionary containing all service instances

    Raises:
        DependencyError: If dependency creation fails
    """
    try:
        container: dict[str, Any] = {}

        # Core services
        container["config"] = config
        # File discovery honors project config patterns/excludes
        container["file_discovery"] = FileDiscoveryService(config=config.test_patterns)

        # Adapters - using existing implementations or creating placeholders
        try:
            # State adapter
            container["state_adapter"] = StateJsonAdapter()

            # Telemetry adapter (router-based, with graceful fallback)
            try:
                container["telemetry_adapter"] = create_telemetry_adapter(
                    config.telemetry.model_dump()
                )
            except Exception:
                container["telemetry_adapter"] = NoOpTelemetryAdapter()

            # Cost adapter - now using real implementation
            container["cost_adapter"] = CostManager(
                config=config.model_dump().get("cost_management", {}),
                telemetry=container["telemetry_adapter"],
            )

            # LLM adapter with cost tracking
            container["llm_adapter"] = LLMRouter(
                config.llm.model_dump(), cost_port=container["cost_adapter"]
            )

            # Writer adapter
            container["writer_adapter"] = WriterASTMergeAdapter()

            # Coverage adapter - placeholder for now
            container["coverage_adapter"] = _create_coverage_adapter(config)

            # Context adapter
            container["context_adapter"] = TestcraftContextAdapter()

            # Parser adapter
            container["parser_adapter"] = CodebaseParser()

            # Refine adapter
            container["refine_adapter"] = RefineAdapter(
                llm=container["llm_adapter"],
                config=config.generation.refine,
                writer_port=container["writer_adapter"],
                telemetry_port=container["telemetry_adapter"],
            )

            # Planning workflow adapters
            from ..adapters.io.artifact_store import ArtifactStoreAdapter
            from ..adapters.io.planning_cli_presenter import PlanningCliPresenter
            from ..adapters.llm.planning_adapter import PlanningAdapter
            from ..adapters.presenters import RichPlanningPresenter
            from ..application.generation.services.context_assembler import (
                ContextAssembler,
            )
            from ..application.generation.services.context_pack import (
                ContextPackBuilder,
            )
            from ..application.generation.services.llm_orchestrator import (
                LLMOrchestrator,
            )
            from ..application.generation.services.symbol_resolver import SymbolResolver

            # Create artifact store for plan persistence
            container["artifact_store"] = ArtifactStoreAdapter()

            # Create context pack builder for planning
            context_assembler = ContextAssembler(
                context_port=container["context_adapter"],
                parser_port=container["parser_adapter"],
                config=config.model_dump(),
            )
            container["context_pack_builder"] = ContextPackBuilder(
                context_assembler=context_assembler,
                file_discovery_service=container["file_discovery"],
            )

            # Create symbol resolver
            container["symbol_resolver"] = SymbolResolver(
                parser_port=container["parser_adapter"]
            )

            # Create LLM orchestrator for planning
            from ..config.models import OrchestratorConfig

            orchestrator_config = OrchestratorConfig(
                max_plan_retries=config.planning.max_retries,
                max_refine_retries=config.generation.orchestrator.max_refine_retries,
            )
            container["llm_orchestrator"] = LLMOrchestrator(
                llm_port=container["llm_adapter"],
                parser_port=container["parser_adapter"],
                context_assembler=context_assembler,
                context_pack_builder=container["context_pack_builder"],
                symbol_resolver=container["symbol_resolver"],
                config=orchestrator_config,
            )

            # Create planning adapter
            container["planning_adapter"] = PlanningAdapter(
                llm_orchestrator=container["llm_orchestrator"],
                artifact_store=container["artifact_store"],
                context_pack_builder=container["context_pack_builder"],
            )

            if sys.stdout.isatty() and sys.stdin.isatty():
                container["planning_presenter"] = RichPlanningPresenter()
            else:
                container["planning_presenter"] = PlanningCliPresenter()

        except Exception as e:
            raise DependencyError(f"Failed to create adapters: {e}") from e

        # Use cases
        try:
            container["generate_usecase"] = GenerateUseCase(
                llm_port=container["llm_adapter"],
                writer_port=container["writer_adapter"],
                coverage_port=container["coverage_adapter"],
                refine_port=container["refine_adapter"],
                context_port=container["context_adapter"],
                parser_port=container["parser_adapter"],
                state_port=container["state_adapter"],
                telemetry_port=container["telemetry_adapter"],
                file_discovery_service=container["file_discovery"],
                config=config.model_dump(),
            )

            container["analyze_usecase"] = AnalyzeUseCase(
                coverage_port=container["coverage_adapter"],
                state_port=container["state_adapter"],
                telemetry_port=container["telemetry_adapter"],
                file_discovery_service=container["file_discovery"],
                config=config.model_dump(),
            )

            container["coverage_usecase"] = CoverageUseCase(
                coverage_port=container["coverage_adapter"],
                state_port=container["state_adapter"],
                telemetry_port=container["telemetry_adapter"],
                file_discovery_service=container["file_discovery"],
                config=config.model_dump(),
            )

            container["status_usecase"] = StatusUseCase(
                state_port=container["state_adapter"],
                telemetry_port=container["telemetry_adapter"],
                file_discovery_service=container["file_discovery"],
                config=config.model_dump(),
            )

            container["utility_usecase"] = UtilityUseCase(
                state_port=container["state_adapter"],
                telemetry_port=container["telemetry_adapter"],
                cost_port=container["cost_adapter"],
                file_discovery_service=container["file_discovery"],
                config=config.model_dump(),
            )

            # Planning use case
            from ..application.plan_usecase import PlanUseCase

            container["plan_usecase"] = PlanUseCase(
                planning_port=container["planning_adapter"],
                presenter_port=container["planning_presenter"],
                telemetry_port=container["telemetry_adapter"],
                config=config.model_dump(),
            )

            # Manual-fix use case wiring (CLI command builds a presenter; we wire services here)
            try:
                from ..application.generation.services.context_assembler import (
                    ContextAssembler as _MFContextAssembler,
                )
                from ..application.generation.services.context_pack import (
                    ContextPackBuilder as _MFContextPackBuilder,
                )
                from ..application.manual_fix_usecase import ManualFixGuidanceUseCase

                _mf_context_assembler = _MFContextAssembler(
                    context_port=container["context_adapter"],
                    parser_port=container["parser_adapter"],
                    config=config.model_dump(),
                )
                _mf_context_pack_builder = _MFContextPackBuilder(
                    context_assembler=_mf_context_assembler,
                    file_discovery_service=container["file_discovery"],
                )

                # Provide a factory to CLI for creating the use case with a presenter
                def _manual_fix_usecase_factory(presenter) -> ManualFixGuidanceUseCase:
                    return ManualFixGuidanceUseCase(
                        llm_orchestrator=container["llm_orchestrator"],
                        parser_port=container["parser_adapter"],
                        context_assembler=_mf_context_assembler,
                        context_pack_builder=_mf_context_pack_builder,
                        telemetry_port=container["telemetry_adapter"],
                        presenter=presenter,
                        config=config.model_dump(),
                    )

                container["manual_fix_usecase_factory"] = _manual_fix_usecase_factory
            except Exception:
                # Optional; CLI builds its own when factory unavailable
                pass

        except Exception as e:
            raise DependencyError(f"Failed to create use cases: {e}") from e

        return container

    except DependencyError:
        raise
    except Exception as e:
        raise DependencyError(
            f"Unexpected error during dependency injection: {e}"
        ) from e


def _create_coverage_adapter(config: TestCraftConfig):
    """
    Create coverage adapter - uses real implementation or graceful fallback.

    Follows established pattern from LLM adapter creation.
    """
    try:
        from ..adapters.coverage.coverage_py_adapter import CoveragePyAdapter

        return CoveragePyAdapter()
    except ImportError:
        logger.warning(
            "coverage.py not installed, using no-op adapter. "
            "Install with: pip install coverage"
        )
        from ..adapters.coverage.coverage_py_adapter import NoOpCoverageAdapter

        return NoOpCoverageAdapter()
