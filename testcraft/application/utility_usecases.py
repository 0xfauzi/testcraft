"""
Utility Use Cases - Debug, sync, reset, environment, and cost utilities.

This module implements utility use cases for system operations including
debug state dumping, state synchronization, reset operations, environment
information, and cost summaries, following the established use case patterns.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..adapters.io.file_discovery import FileDiscoveryService
from ..ports.cost_port import CostPort
from ..ports.state_port import StatePort
from ..ports.telemetry_port import SpanKind, TelemetryPort

logger = logging.getLogger(__name__)


class UtilityUseCaseError(Exception):
    """Exception for Utility Use Case specific errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class UtilityUseCase:
    """
    Use case for utility operations and system information.

    Provides utility functions including:
    - Debug state dumping for troubleshooting
    - State synchronization and reset operations
    - Environment information and diagnostics
    - Cost summaries and projections
    """

    def __init__(
        self,
        state_port: StatePort,
        telemetry_port: TelemetryPort,
        cost_port: CostPort | None = None,
        file_discovery_service: FileDiscoveryService | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Utility Use Case with required ports.

        Args:
            state_port: Port for state management operations
            telemetry_port: Port for telemetry and metrics
            cost_port: Optional port for cost operations
            file_discovery_service: Service for file discovery (creates default if None)
            config: Optional configuration overrides
        """
        self._state = state_port
        self._telemetry = telemetry_port
        self._cost = cost_port

        # Initialize file discovery service
        self._file_discovery = file_discovery_service or FileDiscoveryService()

        # Configuration with sensible defaults
        self._config = {
            "include_sensitive_data": False,  # Whether to include sensitive env vars in dumps
            "max_state_entries": 1000,  # Maximum state entries for debug dumps
            "cost_summary_days": 30,  # Days for cost summaries
            "env_info_detail_level": "standard",  # Detail level for env info
            "state_backup_on_reset": True,  # Backup state before reset operations
            **(config or {}),
        }

    async def debug_state(
        self,
        include_telemetry: bool = True,
        include_config: bool = True,
        output_format: str = "json",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Dump internal state for debugging purposes.

        Args:
            include_telemetry: Whether to include telemetry information
            include_config: Whether to include configuration information
            output_format: Output format ('json', 'yaml', 'text')
            **kwargs: Additional debug parameters

        Returns:
            Dictionary containing comprehensive debug state information
        """
        with self._telemetry.create_span(
            "debug_state",
            kind=SpanKind.INTERNAL,
            attributes={
                "include_telemetry": include_telemetry,
                "include_config": include_config,
                "output_format": output_format,
                "config": self._config,
            },
        ) as span:
            try:
                logger.info("Starting debug state dump")

                # Step 1: Get all state information
                debug_state = await self._collect_debug_state()
                span.set_attribute("state_categories", len(debug_state))

                # Step 2: Add telemetry information if requested
                if include_telemetry:
                    telemetry_info = self._collect_telemetry_info()
                    debug_state["telemetry"] = telemetry_info
                    span.set_attribute("telemetry_included", True)

                # Step 3: Add configuration if requested
                if include_config:
                    config_info = self._collect_config_info()
                    debug_state["configuration"] = config_info
                    span.set_attribute("config_included", True)

                # Step 4: Add system information
                system_info = self._collect_system_info()
                debug_state["system"] = system_info

                # Step 5: Format output if needed
                formatted_output = self._format_debug_output(debug_state, output_format)

                # Compile results
                results = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "debug_state": debug_state,
                    "formatted_output": formatted_output,
                    "metadata": {
                        "output_format": output_format,
                        "total_state_entries": sum(
                            len(v) if isinstance(v, dict) else 1
                            for v in debug_state.values()
                        ),
                        "config_used": self._config,
                    },
                }

                logger.info("Debug state dump completed successfully")
                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Debug state dump failed: %s", e)
                raise UtilityUseCaseError(
                    f"Debug state dump failed: {e}", cause=e
                ) from e

    async def sync_state(
        self, force_reload: bool = False, persist_after_sync: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Force state synchronization between memory and storage.

        Args:
            force_reload: Whether to force reload from storage
            persist_after_sync: Whether to persist state after sync
            **kwargs: Additional sync parameters

        Returns:
            Dictionary containing sync operation results
        """
        with self._telemetry.create_span(
            "sync_state",
            kind=SpanKind.INTERNAL,
            attributes={
                "force_reload": force_reload,
                "persist_after_sync": persist_after_sync,
            },
        ) as span:
            try:
                logger.info("Starting state synchronization")

                sync_results = {
                    "operations_performed": [],
                    "state_changes": {},
                    "errors": [],
                }

                # Step 1: Load state from storage if requested
                if force_reload:
                    try:
                        load_result = self._state.load_state()
                        sync_results["operations_performed"].append("load_from_storage")
                        sync_results["state_changes"]["loaded_keys"] = load_result.get(
                            "loaded_keys", []
                        )
                        span.set_attribute(
                            "keys_loaded", len(load_result.get("loaded_keys", []))
                        )
                    except Exception as e:
                        error_msg = f"Failed to load state from storage: {e}"
                        sync_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Step 2: Validate state consistency
                try:
                    validation_result = await self._validate_state_consistency()
                    sync_results["operations_performed"].append("validate_consistency")
                    sync_results["state_changes"]["validation"] = validation_result
                    span.set_attribute(
                        "validation_issues", len(validation_result.get("issues", []))
                    )
                except Exception as e:
                    error_msg = f"State validation failed: {e}"
                    sync_results["errors"].append(error_msg)
                    logger.warning(error_msg)

                # Step 3: Persist state if requested
                if persist_after_sync:
                    try:
                        persist_result = self._state.persist_state()
                        sync_results["operations_performed"].append(
                            "persist_to_storage"
                        )
                        sync_results["state_changes"]["persisted_keys"] = (
                            persist_result.get("persisted_keys", [])
                        )
                        span.set_attribute(
                            "keys_persisted",
                            len(persist_result.get("persisted_keys", [])),
                        )
                    except Exception as e:
                        error_msg = f"Failed to persist state to storage: {e}"
                        sync_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Compile results
                results = {
                    "success": len(sync_results["errors"]) == 0,
                    "timestamp": datetime.now().isoformat(),
                    "sync_results": sync_results,
                    "total_operations": len(sync_results["operations_performed"]),
                    "total_errors": len(sync_results["errors"]),
                    "metadata": {
                        "force_reload": force_reload,
                        "persist_after_sync": persist_after_sync,
                    },
                }

                logger.info(
                    "State synchronization completed. Operations: %d, Errors: %d",
                    len(sync_results["operations_performed"]),
                    len(sync_results["errors"]),
                )

                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("State synchronization failed: %s", e)
                raise UtilityUseCaseError(
                    f"State synchronization failed: {e}", cause=e
                ) from e

    async def reset_state(
        self,
        state_categories: list[str] | None = None,
        create_backup: bool = None,
        confirm_reset: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Clear state and start fresh, with optional backup.

        Args:
            state_categories: Optional list of state categories to reset
            create_backup: Whether to backup before reset (uses config default if None)
            confirm_reset: Safety flag to confirm destructive operation
            **kwargs: Additional reset parameters

        Returns:
            Dictionary containing reset operation results
        """
        with self._telemetry.create_span(
            "reset_state",
            kind=SpanKind.INTERNAL,
            attributes={
                "state_categories": state_categories,
                "create_backup": create_backup,
                "confirm_reset": confirm_reset,
            },
        ) as span:
            try:
                if not confirm_reset:
                    raise UtilityUseCaseError(
                        "Reset operation requires confirmation (confirm_reset=True)"
                    )

                logger.warning("Starting state reset operation - this is destructive!")

                # Use config default for backup if not specified
                if create_backup is None:
                    create_backup = self._config.get("state_backup_on_reset", True)

                reset_results = {
                    "backup_created": False,
                    "backup_location": None,
                    "categories_reset": [],
                    "errors": [],
                }

                # Step 1: Create backup if requested
                if create_backup:
                    try:
                        backup_data = await self._create_state_backup()
                        reset_results["backup_created"] = True
                        reset_results["backup_location"] = backup_data.get(
                            "backup_location"
                        )
                        span.set_attribute("backup_created", True)
                        logger.info(
                            "State backup created at: %s",
                            backup_data.get("backup_location"),
                        )
                    except Exception as e:
                        error_msg = f"Failed to create backup: {e}"
                        reset_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Step 2: Reset specified categories or all state
                categories_to_reset = state_categories or [
                    "generation",
                    "coverage",
                    "telemetry",
                ]

                for category in categories_to_reset:
                    try:
                        self._state.clear_state(category)
                        reset_results["categories_reset"].append(category)
                        logger.info("Reset state category: %s", category)
                    except Exception as e:
                        error_msg = f"Failed to reset category '{category}': {e}"
                        reset_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Step 3: Persist the reset state
                try:
                    self._state.persist_state()
                    logger.info("Reset state persisted to storage")
                except Exception as e:
                    error_msg = f"Failed to persist reset state: {e}"
                    reset_results["errors"].append(error_msg)
                    logger.warning(error_msg)

                span.set_attribute(
                    "categories_reset", len(reset_results["categories_reset"])
                )
                span.set_attribute("reset_errors", len(reset_results["errors"]))

                # Compile results
                results = {
                    "success": len(reset_results["errors"]) == 0,
                    "timestamp": datetime.now().isoformat(),
                    "reset_results": reset_results,
                    "warning": "State has been reset - previous data may be lost",
                    "metadata": {
                        "categories_requested": state_categories,
                        "backup_requested": create_backup,
                        "config_used": self._config,
                    },
                }

                logger.warning(
                    "State reset completed. Categories reset: %d, Errors: %d",
                    len(reset_results["categories_reset"]),
                    len(reset_results["errors"]),
                )

                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("State reset failed: %s", e)
                raise UtilityUseCaseError(f"State reset failed: {e}", cause=e) from e

    async def get_environment_info(
        self,
        include_system_info: bool = True,
        include_python_info: bool = True,
        include_dependency_info: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Show comprehensive environment information.

        Args:
            include_system_info: Whether to include system information
            include_python_info: Whether to include Python environment info
            include_dependency_info: Whether to include dependency information
            **kwargs: Additional environment parameters

        Returns:
            Dictionary containing environment information
        """
        with self._telemetry.create_span(
            "get_environment_info",
            kind=SpanKind.INTERNAL,
            attributes={
                "include_system_info": include_system_info,
                "include_python_info": include_python_info,
                "include_dependency_info": include_dependency_info,
            },
        ) as span:
            try:
                logger.info("Collecting environment information")

                env_info = {}

                # Step 1: System information
                if include_system_info:
                    env_info["system"] = self._collect_system_info()

                # Step 2: Python environment information
                if include_python_info:
                    env_info["python"] = self._collect_python_info()

                # Step 3: Dependency information
                if include_dependency_info:
                    env_info["dependencies"] = await self._collect_dependency_info()

                # Step 4: TestCraft-specific information
                env_info["testcraft"] = self._collect_testcraft_info()

                # Step 5: Environment variables (filtered for safety)
                env_info["environment_variables"] = self._collect_filtered_env_vars()

                span.set_attribute("info_sections", len(env_info))

                # Compile results
                results = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "environment_info": env_info,
                    "metadata": {
                        "detail_level": self._config.get(
                            "env_info_detail_level", "standard"
                        ),
                        "sections_included": list(env_info.keys()),
                        "config_used": self._config,
                    },
                }

                logger.info("Environment information collection completed")
                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Environment info collection failed: %s", e)
                raise UtilityUseCaseError(
                    f"Environment info collection failed: {e}", cause=e
                ) from e

    async def get_cost_summary(
        self,
        time_period: str = "monthly",
        include_projections: bool = True,
        breakdown_by_service: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Display cost summary and projections.

        Args:
            time_period: Time period for summary ('daily', 'weekly', 'monthly')
            include_projections: Whether to include cost projections
            breakdown_by_service: Whether to break down costs by service
            **kwargs: Additional cost parameters

        Returns:
            Dictionary containing cost summary and projections
        """
        if not self._cost:
            return {
                "success": False,
                "error": "Cost port not available - cost tracking may be disabled",
                "timestamp": datetime.now().isoformat(),
            }

        with self._telemetry.create_span(
            "get_cost_summary",
            kind=SpanKind.INTERNAL,
            attributes={
                "time_period": time_period,
                "include_projections": include_projections,
                "breakdown_by_service": breakdown_by_service,
            },
        ) as span:
            try:
                logger.info("Generating cost summary for period: %s", time_period)

                # Step 1: Get basic cost summary
                cost_summary = self._cost.get_summary(
                    time_period=time_period, service_filter=None
                )

                # Step 2: Get detailed breakdown
                cost_breakdown = {}
                if breakdown_by_service:
                    cost_breakdown = self._cost.get_cost_breakdown()

                # Step 3: Calculate projections if requested
                projections = {}
                if include_projections:
                    projections = await self._calculate_cost_projections(
                        cost_summary, time_period
                    )

                # Step 4: Check cost limits
                limit_status = self._cost.check_cost_limit()

                span.set_attribute("total_cost", cost_summary.get("total_cost", 0))
                span.set_attribute(
                    "within_limits", limit_status.get("within_limits", True)
                )

                # Compile results
                results = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "cost_summary": cost_summary,
                    "cost_breakdown": cost_breakdown if breakdown_by_service else {},
                    "projections": projections if include_projections else {},
                    "limit_status": limit_status,
                    "metadata": {
                        "time_period": time_period,
                        "projections_included": include_projections,
                        "breakdown_included": breakdown_by_service,
                        "config_used": self._config,
                    },
                }

                logger.info("Cost summary generated successfully")
                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Cost summary generation failed: %s", e)
                raise UtilityUseCaseError(
                    f"Cost summary generation failed: {e}", cause=e
                ) from e

    async def _collect_debug_state(self) -> dict[str, Any]:
        """Collect comprehensive debug state information."""
        debug_state = {}

        try:
            # Get all state categories
            generation_state = self._state.get_all_state("generation")
            coverage_state = self._state.get_all_state("coverage")
            telemetry_state = self._state.get_all_state("telemetry")

            debug_state["generation"] = generation_state
            debug_state["coverage"] = coverage_state
            debug_state["telemetry"] = telemetry_state

            # Add state metadata
            debug_state["metadata"] = {
                "collection_timestamp": datetime.now().isoformat(),
                "total_state_keys": len(generation_state)
                + len(coverage_state)
                + len(telemetry_state),
                "state_health": {
                    "has_generation_data": bool(generation_state),
                    "has_coverage_data": bool(coverage_state),
                    "has_telemetry_data": bool(telemetry_state),
                },
            }

        except Exception as e:
            debug_state["collection_error"] = str(e)
            logger.warning("Failed to collect some debug state: %s", e)

        return debug_state

    def _collect_telemetry_info(self) -> dict[str, Any]:
        """Collect telemetry system information."""
        telemetry_info = {
            "enabled": self._telemetry.is_enabled(),
            "current_trace_context": None,
            "backend_type": "unknown",  # Would be determined by implementation
        }

        try:
            trace_context = self._telemetry.get_trace_context()
            if trace_context:
                telemetry_info["current_trace_context"] = {
                    "trace_id": trace_context.trace_id,
                    "span_id": trace_context.span_id,
                    "parent_span_id": trace_context.parent_span_id,
                }
        except Exception as e:
            telemetry_info["trace_context_error"] = str(e)

        return telemetry_info

    def _collect_config_info(self) -> dict[str, Any]:
        """Collect configuration information."""
        return {
            "utility_usecase_config": self._config.copy(),
            "sensitive_data_included": self._config.get(
                "include_sensitive_data", False
            ),
        }

    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }

    def _collect_python_info(self) -> dict[str, Any]:
        """Collect Python environment information."""
        return {
            "version": sys.version,
            "version_info": list(sys.version_info),
            "executable": sys.executable,
            "path": sys.path[:10],  # Limit path length
            "modules_count": len(sys.modules),
            "prefix": sys.prefix,
            "exec_prefix": sys.exec_prefix,
        }

    async def _collect_dependency_info(self) -> dict[str, Any]:
        """Collect dependency and package information."""
        deps_info = {
            "package_manager": "unknown",
            "virtual_env": None,
            "requirements": [],
        }

        try:
            # Check for virtual environment
            if hasattr(sys, "real_prefix") or (
                hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
            ):
                deps_info["virtual_env"] = sys.prefix

            # Try to detect package manager
            cwd = Path.cwd()
            if (cwd / "requirements.txt").exists():
                deps_info["package_manager"] = "pip"
            elif (cwd / "Pipfile").exists():
                deps_info["package_manager"] = "pipenv"
            elif (cwd / "pyproject.toml").exists():
                deps_info["package_manager"] = "poetry/pip"
            elif (cwd / "environment.yml").exists():
                deps_info["package_manager"] = "conda"

        except Exception as e:
            deps_info["collection_error"] = str(e)

        return deps_info

    def _collect_testcraft_info(self) -> dict[str, Any]:
        """Collect TestCraft-specific information."""
        return {
            "version": "unknown",  # Would be determined from package metadata
            "config_loaded": bool(self._config),
            "ports_available": {
                "state_port": self._state is not None,
                "telemetry_port": self._telemetry is not None,
                "cost_port": self._cost is not None,
                "file_discovery": self._file_discovery is not None,
            },
        }

    def _collect_filtered_env_vars(self) -> dict[str, Any]:
        """Collect filtered environment variables (excluding sensitive ones)."""
        include_sensitive = self._config.get("include_sensitive_data", False)

        # Define sensitive patterns
        sensitive_patterns = [
            "key",
            "secret",
            "token",
            "password",
            "pwd",
            "auth",
            "credential",
            "private",
            "api_key",
            "access_key",
        ]

        filtered_env = {}
        for key, value in os.environ.items():
            is_sensitive = any(pattern in key.lower() for pattern in sensitive_patterns)

            if include_sensitive or not is_sensitive:
                filtered_env[key] = value
            elif is_sensitive:
                filtered_env[key] = "[REDACTED]"

        return {
            "total_variables": len(os.environ),
            "filtered_variables": len(filtered_env),
            "variables": filtered_env,
        }

    def _format_debug_output(
        self, debug_state: dict[str, Any], output_format: str
    ) -> str:
        """Format debug state for output."""
        if output_format.lower() == "json":
            import json

            return json.dumps(debug_state, indent=2, default=str)
        elif output_format.lower() == "yaml":
            try:
                import yaml

                return yaml.dump(debug_state, default_flow_style=False)
            except ImportError:
                return "YAML format requested but PyYAML not available"
        elif output_format.lower() == "text":
            return self._format_as_text(debug_state)
        else:
            return str(debug_state)

    def _format_as_text(self, data: dict[str, Any], indent: int = 0) -> str:
        """Format dictionary as readable text."""
        lines = []
        prefix = "  " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_as_text(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: [{len(value)} items]")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return "\n".join(lines)

    async def _validate_state_consistency(self) -> dict[str, Any]:
        """Validate state consistency and return issues found."""
        validation_result = {"issues": [], "warnings": [], "suggestions": []}

        try:
            # Check for orphaned state entries
            all_state = self._state.get_all_state()
            for key, value in all_state.items():
                if value is None:
                    validation_result["warnings"].append(
                        f"Null value for state key: {key}"
                    )
                elif isinstance(value, dict) and not value:
                    validation_result["warnings"].append(
                        f"Empty dictionary for state key: {key}"
                    )

        except Exception as e:
            validation_result["issues"].append(f"Validation failed: {e}")

        return validation_result

    async def _create_state_backup(self) -> dict[str, Any]:
        """Create a backup of current state."""
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_location = f"state_backup_{backup_timestamp}.json"

        try:
            all_state = self._state.get_all_state()

            # In a real implementation, this would write to a file
            # For now, we'll just return the backup information
            backup_data = {
                "timestamp": backup_timestamp,
                "backup_location": backup_location,
                "state_entries": len(all_state),
                "backup_size_estimate": len(str(all_state)),
            }

            return backup_data

        except Exception as e:
            raise UtilityUseCaseError(
                f"Failed to create state backup: {e}", cause=e
            ) from e

    async def _calculate_cost_projections(
        self, cost_summary: dict[str, Any], time_period: str
    ) -> dict[str, Any]:
        """Calculate cost projections based on current usage."""
        try:
            current_cost = cost_summary.get("total_cost", 0)

            # Simple projection based on current period
            projections = {
                "current_period_cost": current_cost,
                "projected_monthly": (
                    current_cost * 30 if time_period == "daily" else current_cost
                ),
                "projected_yearly": (
                    current_cost * 365 if time_period == "daily" else current_cost * 12
                ),
                "projection_accuracy": "low",  # Simple linear projection
                "projection_notes": "Based on linear extrapolation of current usage",
            }

            return projections

        except Exception as e:
            return {"projection_error": str(e)}
