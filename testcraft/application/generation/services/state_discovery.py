"""
State synchronization and file discovery service.

Handles state management and source file discovery for test generation,
extracting the logic from GenerateUseCase._sync_state_and_discover_files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ....adapters.io.file_discovery import FileDiscoveryService
from ....ports.state_port import StatePort
from ....ports.telemetry_port import TelemetryPort

logger = logging.getLogger(__name__)


class GenerateUseCaseError(Exception):
    """Exception for Generate Use Case specific errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class StateSyncDiscovery:
    """
    Service for state synchronization and source file discovery.

    Handles loading existing state and discovering source files to process
    based on target files or discovery patterns.
    """

    def __init__(
        self,
        state_port: StatePort,
        file_discovery_service: FileDiscoveryService,
        telemetry_port: TelemetryPort,
    ):
        """
        Initialize the state sync and discovery service.

        Args:
            state_port: Port for state management operations
            file_discovery_service: Service for file discovery
            telemetry_port: Port for telemetry operations
        """
        self._state = state_port
        self._file_discovery = file_discovery_service
        self._telemetry = telemetry_port

    def sync_and_discover(
        self, project_path: Path, target_files: list[str | Path] | None = None
    ) -> dict[str, Any]:
        """
        Synchronize state and discover source files to process.

        Args:
            project_path: Root path of the project
            target_files: Optional list of specific files to target

        Returns:
            Dictionary with discovered files and metadata

        Raises:
            GenerateUseCaseError: If file discovery fails
        """
        with self._telemetry.create_child_span("sync_state_and_discover") as span:
            try:
                # Load current state
                current_state = self._state.get_all_state("generation")
                span.set_attribute("previous_state_keys", len(current_state))

                # Discover source files using FileDiscoveryService
                if target_files:
                    # Use provided target files, filtered for validity
                    file_paths = [str(f) for f in target_files]
                    files = [
                        Path(f)
                        for f in self._file_discovery.filter_existing_files(file_paths)
                    ]
                    span.set_attribute("discovery_method", "target_files")
                else:
                    # Discover source files using file discovery service
                    discovered_files = self._file_discovery.discover_source_files(
                        project_path, include_test_files=False
                    )
                    files = [Path(f) for f in discovered_files]
                    span.set_attribute("discovery_method", "pattern_discovery")

                span.set_attribute("files_found", len(files))

                return {
                    "files": files,
                    "previous_state": current_state,
                    "timestamp": (
                        span.get_trace_context().trace_id
                        if span.get_trace_context()
                        else None
                    ),
                    "project_path": project_path,
                }

            except Exception as e:
                logger.exception("Failed to sync state and discover files: %s", e)
                raise GenerateUseCaseError(
                    f"File discovery failed: {e}", cause=e
                ) from e
