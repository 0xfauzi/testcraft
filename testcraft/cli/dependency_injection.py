"""Dependency injection container for CLI commands."""

from typing import Any, Dict

from ..config.models import TestCraftConfig
from ..application.generate_usecase import GenerateUseCase
from ..application.analyze_usecase import AnalyzeUseCase
from ..application.coverage_usecase import CoverageUseCase
from ..application.status_usecase import StatusUseCase
from ..application.utility_usecases import UtilityUseCase

# Import adapters
from ..adapters.io.file_discovery import FileDiscoveryService
from ..adapters.io.state_json import JSONStateAdapter
from ..adapters.telemetry.noop_adapter import NoopTelemetryAdapter
from ..adapters.llm.router import LLMRouter
from ..adapters.io.writer_ast_merge import ASTMergeWriterAdapter
from ..adapters.coverage import CoverageAdapter  # This would need to be implemented
from ..adapters.context.main_adapter import ContextAdapter
from ..adapters.parsing.codebase_parser import CodebaseParserAdapter
from ..adapters.refine.main_adapter import RefineAdapter


class DependencyError(Exception):
    """Raised when dependency injection fails."""
    pass


def create_dependency_container(config: TestCraftConfig) -> Dict[str, Any]:
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
        container['config'] = config
        container['file_discovery'] = FileDiscoveryService()
        
        # Adapters - using existing implementations or creating placeholders
        try:
            # State adapter
            container['state_adapter'] = JSONStateAdapter()
            
            # Telemetry adapter (using noop for now)
            container['telemetry_adapter'] = NoopTelemetryAdapter()
            
            # LLM adapter
            container['llm_adapter'] = LLMRouter(config.llm)
            
            # Writer adapter
            container['writer_adapter'] = ASTMergeWriterAdapter()
            
            # Coverage adapter - placeholder for now
            container['coverage_adapter'] = _create_coverage_adapter(config)
            
            # Context adapter
            container['context_adapter'] = ContextAdapter(config.context)
            
            # Parser adapter
            container['parser_adapter'] = CodebaseParserAdapter()
            
            # Refine adapter
            container['refine_adapter'] = RefineAdapter(config.generation.refine)
            
            # Cost adapter - optional
            container['cost_adapter'] = None  # Could be implemented later
            
        except Exception as e:
            raise DependencyError(f"Failed to create adapters: {e}")
        
        # Use cases
        try:
            container['generate_usecase'] = GenerateUseCase(
                llm_port=container['llm_adapter'],
                writer_port=container['writer_adapter'],
                coverage_port=container['coverage_adapter'],
                refine_port=container['refine_adapter'],
                context_port=container['context_adapter'],
                parser_port=container['parser_adapter'],
                state_port=container['state_adapter'],
                telemetry_port=container['telemetry_adapter'],
                file_discovery_service=container['file_discovery'],
                config=config.model_dump()
            )
            
            container['analyze_usecase'] = AnalyzeUseCase(
                coverage_port=container['coverage_adapter'],
                state_port=container['state_adapter'],
                telemetry_port=container['telemetry_adapter'],
                file_discovery_service=container['file_discovery'],
                config=config.model_dump()
            )
            
            container['coverage_usecase'] = CoverageUseCase(
                coverage_port=container['coverage_adapter'],
                state_port=container['state_adapter'],
                telemetry_port=container['telemetry_adapter'],
                file_discovery_service=container['file_discovery'],
                config=config.model_dump()
            )
            
            container['status_usecase'] = StatusUseCase(
                state_port=container['state_adapter'],
                telemetry_port=container['telemetry_adapter'],
                file_discovery_service=container['file_discovery'],
                config=config.model_dump()
            )
            
            container['utility_usecase'] = UtilityUseCase(
                state_port=container['state_adapter'],
                telemetry_port=container['telemetry_adapter'],
                cost_port=container['cost_adapter'],
                file_discovery_service=container['file_discovery'],
                config=config.model_dump()
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
                'overall_line_coverage': 0.0,
                'overall_branch_coverage': 0.0,
                'files_covered': 0,
                'total_lines': 0,
                'missing_coverage': {}
            }
        
        def identify_gaps(self, coverage_data, threshold=0.8):
            return {}
        
        def report_coverage(self, coverage_data, output_format='detailed'):
            return {'success': True, 'format': output_format}
    
    return PlaceholderCoverageAdapter()
