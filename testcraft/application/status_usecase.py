"""
Status Use Case - View generation state/history and statistics.

This module implements the status use case for viewing generation state,
history, and providing summary statistics about test generation activities,
following the established pattern from existing use cases.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..adapters.io.file_discovery import FileDiscoveryService
from ..ports.state_port import StatePort
from ..ports.telemetry_port import SpanKind, TelemetryPort

logger = logging.getLogger(__name__)


class StatusUseCaseError(Exception):
    """Exception for Status Use Case specific errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class StatusUseCase:
    """
    Use case for viewing generation state/history and statistics.

    Provides comprehensive status reporting including:
    - Current state and recent activity
    - Generation history with filtering and sorting
    - Summary statistics and trends
    - File-level status tracking
    """

    def __init__(
        self,
        state_port: StatePort,
        telemetry_port: TelemetryPort,
        file_discovery_service: FileDiscoveryService | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Status Use Case with required ports.

        Args:
            state_port: Port for state management
            telemetry_port: Port for telemetry and metrics
            file_discovery_service: Service for file discovery (creates default if None)
            config: Optional configuration overrides
        """
        self._state = state_port
        self._telemetry = telemetry_port

        # Initialize file discovery service
        self._file_discovery = file_discovery_service or FileDiscoveryService()

        # Configuration with sensible defaults
        self._config = {
            "max_history_entries": 50,  # Maximum history entries to return
            "include_file_details": True,  # Whether to include per-file status
            "summary_time_window_days": 7,  # Days for summary statistics
            "status_cache_timeout": 300,  # Status cache timeout in seconds
            **(config or {}),
        }

    async def get_generation_status(
        self,
        project_path: str | Path | None = None,
        include_history: bool = True,
        include_statistics: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get comprehensive generation status including state, history, and statistics.

        Args:
            project_path: Optional project path for file-specific status
            include_history: Whether to include generation history
            include_statistics: Whether to include summary statistics
            **kwargs: Additional status parameters

        Returns:
            Dictionary containing status information, history, and statistics
        """
        with self._telemetry.create_span(
            "get_generation_status",
            kind=SpanKind.INTERNAL,
            attributes={
                "project_path": str(project_path) if project_path else None,
                "include_history": include_history,
                "include_statistics": include_statistics,
                "config": self._config,
            },
        ) as span:
            try:
                logger.info("Getting generation status for project: %s", project_path)

                # Step 1: Get current state
                current_state = await self._get_current_state()
                span.set_attribute("current_state_keys", len(current_state))

                # Step 2: Get generation history if requested
                history = []
                if include_history:
                    history = await self._get_generation_history()
                    span.set_attribute("history_entries", len(history))

                # Step 3: Get summary statistics if requested
                statistics = {}
                if include_statistics:
                    statistics = await self._get_summary_statistics()
                    span.set_attribute("statistics_calculated", len(statistics))

                # Step 4: Get file-level status if project path provided
                file_status = {}
                if project_path and self._config["include_file_details"]:
                    file_status = await self._get_file_level_status(Path(project_path))
                    span.set_attribute("files_analyzed", len(file_status))

                # Compile results
                results = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "current_state": current_state,
                    "generation_history": history if include_history else [],
                    "summary_statistics": statistics if include_statistics else {},
                    "file_level_status": file_status,
                    "metadata": {
                        "config_used": self._config,
                        "query_parameters": {
                            "include_history": include_history,
                            "include_statistics": include_statistics,
                            "project_path": str(project_path) if project_path else None,
                        },
                    },
                }

                logger.info("Status retrieval completed successfully")
                return results

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Status retrieval failed: %s", e)
                raise StatusUseCaseError(
                    f"Status retrieval failed: {e}", cause=e
                ) from e

    async def get_filtered_history(
        self,
        limit: int = 10,
        status_filter: str | None = None,
        date_range: dict[str, datetime] | None = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Get filtered and sorted generation history.

        Args:
            limit: Maximum number of entries to return
            status_filter: Optional status filter ('success', 'failed', etc.)
            date_range: Optional date range with 'start' and 'end' keys
            sort_by: Field to sort by (timestamp, status, files_processed)
            sort_order: Sort order ('asc' or 'desc')
            **kwargs: Additional filtering parameters

        Returns:
            List of filtered and sorted history records
        """
        with self._telemetry.create_span(
            "get_filtered_history",
            kind=SpanKind.INTERNAL,
            attributes={
                "limit": limit,
                "status_filter": status_filter,
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
        ) as span:
            try:
                # Get full history
                full_history = await self._get_generation_history()

                # Apply filters
                filtered_history = self._apply_history_filters(
                    full_history, status_filter, date_range
                )
                span.set_attribute("filtered_entries", len(filtered_history))

                # Apply sorting
                sorted_history = self._sort_history(
                    filtered_history, sort_by, sort_order
                )

                # Apply limit
                limited_history = sorted_history[:limit]
                span.set_attribute("returned_entries", len(limited_history))

                return limited_history

            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                logger.exception("Filtered history retrieval failed: %s", e)
                raise StatusUseCaseError(
                    f"Filtered history retrieval failed: {e}", cause=e
                ) from e

    async def _get_current_state(self) -> dict[str, Any]:
        """
        Get current generation state from state port.

        Returns:
            Dictionary containing current state information
        """
        with self._telemetry.create_child_span("get_current_state") as span:
            try:
                # Get all generation-related state
                generation_state = self._state.get_all_state("generation")
                coverage_state = self._state.get_all_state("coverage")

                # Get last run information
                last_generation_run = generation_state.get("last_generation_run", {})
                last_coverage_run = coverage_state.get("last_coverage_run", {})

                current_state = {
                    "last_generation_timestamp": last_generation_run.get(
                        "last_run_timestamp"
                    ),
                    "last_coverage_timestamp": last_coverage_run.get(
                        "last_coverage_run_timestamp"
                    ),
                    "generation_summary": last_generation_run.get(
                        "generation_summary", {}
                    ),
                    "refinement_summary": last_generation_run.get(
                        "refinement_summary", {}
                    ),
                    "coverage_improvement": last_generation_run.get(
                        "coverage_improvement", {}
                    ),
                    "active_config": last_generation_run.get("config_used", {}),
                    "state_health": {
                        "has_generation_data": bool(last_generation_run),
                        "has_coverage_data": bool(last_coverage_run),
                        "data_freshness_hours": self._calculate_data_freshness(
                            last_generation_run
                        ),
                    },
                }

                span.set_attribute("state_entries", len(current_state))
                return current_state

            except Exception as e:
                logger.warning("Failed to get current state: %s", e)
                return {
                    "error": str(e),
                    "state_health": {
                        "has_generation_data": False,
                        "has_coverage_data": False,
                    },
                }

    async def _get_generation_history(self) -> list[dict[str, Any]]:
        """
        Get generation history from state records.

        Returns:
            List of historical generation records
        """
        with self._telemetry.create_child_span("get_generation_history") as span:
            try:
                # Get generation state which may contain historical data
                generation_state = self._state.get_all_state("generation")

                # Extract history - in a full implementation, this would pull from
                # a dedicated history storage, but for now we'll work with available state
                history = []

                # Add current run as most recent history entry
                if "last_generation_run" in generation_state:
                    last_run = generation_state["last_generation_run"]
                    history_entry = {
                        "timestamp": last_run.get("last_run_timestamp"),
                        "status": (
                            "success"
                            if last_run.get("generation_summary", {}).get(
                                "successful_generations", 0
                            )
                            > 0
                            else "partial"
                        ),
                        "files_processed": last_run.get("generation_summary", {}).get(
                            "total_files_processed", 0
                        ),
                        "tests_generated": last_run.get("generation_summary", {}).get(
                            "successful_generations", 0
                        ),
                        "tests_refined": last_run.get("refinement_summary", {}).get(
                            "successful_refinements", 0
                        ),
                        "coverage_delta": last_run.get("coverage_improvement", {}).get(
                            "line_coverage_delta", 0
                        ),
                        "config_snapshot": last_run.get("config_used", {}),
                        "entry_type": "generation_run",
                    }
                    history.append(history_entry)

                # Add coverage runs as additional history entries
                coverage_state = self._state.get_all_state("coverage")
                if "last_coverage_run" in coverage_state:
                    coverage_run = coverage_state["last_coverage_run"]
                    history_entry = {
                        "timestamp": coverage_run.get("last_coverage_run_timestamp"),
                        "status": "success",
                        "files_processed": coverage_run.get("files_measured", 0),
                        "coverage_percentage": coverage_run.get(
                            "coverage_summary", {}
                        ).get("overall_line_coverage", 0),
                        "config_snapshot": coverage_run.get("config_used", {}),
                        "entry_type": "coverage_run",
                    }
                    history.append(history_entry)

                # Filter out entries without valid timestamps
                history = [entry for entry in history if entry.get("timestamp")]

                # Sort by timestamp descending
                history.sort(key=lambda x: x["timestamp"], reverse=True)

                # Limit history size
                max_entries = self._config.get("max_history_entries", 50)
                history = history[:max_entries]

                span.set_attribute("history_entries_found", len(history))
                return history

            except Exception as e:
                logger.warning("Failed to get generation history: %s", e)
                return []

    async def _get_summary_statistics(self) -> dict[str, Any]:
        """
        Generate summary statistics for recent activity.

        Returns:
            Dictionary containing summary statistics
        """
        with self._telemetry.create_child_span("get_summary_statistics") as span:
            try:
                time_window_days = self._config.get("summary_time_window_days", 7)
                cutoff_time = datetime.now() - timedelta(days=time_window_days)

                # Get recent history
                history = await self._get_generation_history()

                # Filter to recent entries
                recent_entries = [
                    entry
                    for entry in history
                    if entry.get("timestamp")
                    and datetime.fromtimestamp(entry["timestamp"]) > cutoff_time
                ]

                # Calculate statistics
                total_runs = len(recent_entries)
                successful_runs = len(
                    [e for e in recent_entries if e.get("status") == "success"]
                )
                total_files = sum(e.get("files_processed", 0) for e in recent_entries)
                total_tests = sum(e.get("tests_generated", 0) for e in recent_entries)

                # Coverage statistics
                coverage_entries = [
                    e for e in recent_entries if "coverage_percentage" in e
                ]
                avg_coverage = (
                    sum(e["coverage_percentage"] for e in coverage_entries)
                    / len(coverage_entries)
                    if coverage_entries
                    else 0
                )

                statistics = {
                    "time_window_days": time_window_days,
                    "total_runs": total_runs,
                    "successful_runs": successful_runs,
                    "success_rate": (
                        successful_runs / total_runs if total_runs > 0 else 0
                    ),
                    "total_files_processed": total_files,
                    "total_tests_generated": total_tests,
                    "average_coverage_percentage": avg_coverage,
                    "activity_trend": self._calculate_activity_trend(recent_entries),
                    "recent_activity": {
                        "last_24h": len(
                            [
                                e
                                for e in recent_entries
                                if e.get("timestamp")
                                and datetime.fromtimestamp(e["timestamp"])
                                > datetime.now() - timedelta(hours=24)
                            ]
                        ),
                        "last_week": len(recent_entries),
                    },
                }

                span.set_attribute("statistics_calculated", len(statistics))
                return statistics

            except Exception as e:
                logger.warning("Failed to calculate summary statistics: %s", e)
                return {"error": str(e)}

    async def _get_file_level_status(self, project_path: Path) -> dict[str, Any]:
        """
        Get file-level status information for the project.

        Args:
            project_path: Project root path

        Returns:
            Dictionary containing file-level status
        """
        with self._telemetry.create_child_span("get_file_level_status") as span:
            try:
                # Discover source files
                discovered_files = self._file_discovery.discover_source_files(
                    project_path, include_test_files=False
                )

                file_status = {}
                for file_path in discovered_files:
                    path_obj = Path(file_path)

                    # Check for existing tests
                    has_tests = self._has_existing_tests(path_obj)

                    # Get file modification time
                    mod_time = None
                    try:
                        mod_time = path_obj.stat().st_mtime
                        file_age_days = (datetime.now().timestamp() - mod_time) / (
                            24 * 3600
                        )
                    except Exception:
                        file_age_days = None

                    file_status[str(path_obj)] = {
                        "has_tests": has_tests,
                        "file_age_days": file_age_days,
                        "last_modified": (
                            datetime.fromtimestamp(mod_time).isoformat()
                            if mod_time is not None
                            else None
                        ),
                        "needs_attention": not has_tests,  # Simple heuristic
                    }

                span.set_attribute("files_analyzed", len(file_status))
                return file_status

            except Exception as e:
                logger.warning("Failed to get file-level status: %s", e)
                return {"error": str(e)}

    def _has_existing_tests(self, file_path: Path) -> bool:
        """Check if a file has existing test files."""
        potential_test_files = [
            file_path.parent / f"test_{file_path.name}",
            file_path.parent / f"{file_path.stem}_test.py",
            file_path.parent.parent / "tests" / f"test_{file_path.name}",
        ]

        return any(test_file.exists() for test_file in potential_test_files)

    def _apply_history_filters(
        self,
        history: list[dict[str, Any]],
        status_filter: str | None = None,
        date_range: dict[str, datetime] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply filters to history records."""
        filtered = history.copy()

        # Status filter
        if status_filter:
            filtered = [
                entry for entry in filtered if entry.get("status") == status_filter
            ]

        # Date range filter
        if date_range and "start" in date_range and "end" in date_range:
            start_ts = date_range["start"].timestamp()
            end_ts = date_range["end"].timestamp()
            filtered = [
                entry
                for entry in filtered
                if entry.get("timestamp") and start_ts <= entry["timestamp"] <= end_ts
            ]

        return filtered

    def _sort_history(
        self, history: list[dict[str, Any]], sort_by: str, sort_order: str
    ) -> list[dict[str, Any]]:
        """Sort history records by specified field and order."""
        reverse = sort_order.lower() == "desc"

        try:
            return sorted(history, key=lambda x: x.get(sort_by, 0), reverse=reverse)
        except Exception as e:
            logger.warning("Failed to sort history by %s: %s", sort_by, e)
            return history

    def _calculate_data_freshness(self, last_run: dict[str, Any]) -> float | None:
        """Calculate how many hours ago the last run occurred."""
        if not last_run or "last_run_timestamp" not in last_run:
            return None

        try:
            last_timestamp = last_run["last_run_timestamp"]
            current_time = datetime.now().timestamp()
            hours_ago = (current_time - last_timestamp) / 3600
            return hours_ago
        except Exception:
            return None

    def _calculate_activity_trend(self, recent_entries: list[dict[str, Any]]) -> str:
        """Calculate activity trend based on recent entries."""
        if len(recent_entries) < 2:
            return "insufficient_data"

        # Simple trend calculation based on recent activity
        # This could be made more sophisticated with actual time series analysis
        recent_count = len(list(recent_entries[: len(recent_entries) // 2]))
        older_count = len(list(recent_entries[len(recent_entries) // 2 :]))

        if recent_count > older_count:
            return "increasing"
        elif recent_count < older_count:
            return "decreasing"
        else:
            return "stable"
