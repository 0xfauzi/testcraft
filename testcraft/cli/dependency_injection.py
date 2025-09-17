"""Dependency injection container for CLI commands."""

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
from ..adapters.coverage.quick_probe import CoverageQuickProbeAdapter
from ..application.analyze_usecase import AnalyzeUseCase
from ..application.coverage_usecase import CoverageUseCase
from ..application.generate_usecase import GenerateUseCase
from ..application.planning_usecase import PlanningUseCase
from ..application.status_usecase import StatusUseCase
from ..application.utility_usecases import UtilityUseCase
from ..config.models import TestCraftConfig


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
        container = {}

        # Core services
        container["config"] = config
        container["file_discovery"] = FileDiscoveryService(config=config.test_patterns)

        # Adapters - using existing implementations or creating placeholders
        try:
            # State adapter
            container["state_adapter"] = StateJsonAdapter()

            # Telemetry adapter - select based on configuration
            container["telemetry_adapter"] = _create_telemetry_adapter(config)

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
            
            # Coverage probe adapter (for test discovery)
            container["coverage_probe_adapter"] = CoverageQuickProbeAdapter(config.test_patterns.test_discovery)

            # Context adapter
            container["context_adapter"] = TestcraftContextAdapter()

            # Parser adapter
            container["parser_adapter"] = CodebaseParser()

            # Refine adapter - build RefineConfig from TOML
            from ..application.generation.config import GenerationConfig
            refine_config = GenerationConfig.build_refine_config_from_toml(config)
            container["refine_adapter"] = RefineAdapter(
                llm=container["llm_adapter"],
                config=refine_config,
                writer_port=container["writer_adapter"],
                telemetry_port=container["telemetry_adapter"]
            )

        except Exception as e:
            raise DependencyError(f"Failed to create adapters: {e}")

        # Use cases
        try:
            # Add coverage probe to config so it can be passed through
            config_dict = config.model_dump()
            config_dict["coverage_probe"] = container["coverage_probe_adapter"]
            
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
                config=config_dict,
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

            container["planning_use_case"] = PlanningUseCase(
                llm_port=container["llm_adapter"],
                parser_port=container["parser_adapter"],
                context_port=container["context_adapter"],
                state_port=container["state_adapter"],
                telemetry_port=container["telemetry_adapter"],
                file_discovery_service=container["file_discovery"],
                config=config.model_dump(),
            )

        except Exception as e:
            raise DependencyError(f"Failed to create use cases: {e}")

        return container

    except DependencyError:
        raise
    except Exception as e:
        raise DependencyError(f"Unexpected error during dependency injection: {e}")


def _create_coverage_adapter(config: TestCraftConfig):
    """Create coverage adapter - placeholder implementation."""

    # This is a placeholder - actual implementation would depend on available coverage adapters
    class PlaceholderCoverageAdapter:
        def __init__(self):
            pass

        def measure_coverage(self, source_files, test_files=None):
            # Placeholder implementation
            return {}

        def get_coverage_summary(self, coverage_data):
            return {
                "overall_line_coverage": 0.0,
                "overall_branch_coverage": 0.0,
                "files_covered": 0,
                "total_lines": 0,
                "missing_coverage": {},
            }

        def identify_gaps(self, coverage_data, threshold=0.8):
            return {}

        def report_coverage(self, coverage_data, output_format="detailed"):
            return {"success": True, "format": output_format}

    return PlaceholderCoverageAdapter()


def _create_telemetry_adapter(config: TestCraftConfig):
    """Create telemetry adapter based on configuration."""
    telemetry_config = config.telemetry
    
    # Check if telemetry is disabled or opt-out is enabled
    if not telemetry_config.enabled or telemetry_config.opt_out_data_collection:
        return NoOpTelemetryAdapter()
    
    backend = telemetry_config.backend
    
    if backend == "opentelemetry":
        # Try to import and create OpenTelemetry adapter
        try:
            from ..adapters.telemetry.opentelemetry_adapter import OpenTelemetryAdapter
            return OpenTelemetryAdapter(telemetry_config.model_dump())
        except ImportError:
            # Fallback to noop if OpenTelemetry dependencies are not available
            import logging
            logging.getLogger(__name__).warning(
                "OpenTelemetry backend requested but dependencies not available, using NoOp adapter"
            )
            return NoOpTelemetryAdapter()
    elif backend == "noop":
        return NoOpTelemetryAdapter()
    else:
        # For other backends (datadog, jaeger), use noop for now
        import logging
        logging.getLogger(__name__).warning(
            "Telemetry backend '%s' not yet implemented, using NoOp adapter", backend
        )
        return NoOpTelemetryAdapter()
