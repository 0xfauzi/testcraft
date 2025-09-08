"""
Cost management adapter for tracking and managing costs.

This adapter implements the CostPort interface to provide cost tracking,
budget enforcement, and cost optimization strategies.
"""

import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ...ports.telemetry_port import TelemetryPort


@dataclass
class CostEntry:
    """Individual cost tracking entry."""

    id: str
    timestamp: datetime
    service: str
    operation: str
    cost: float
    tokens_used: int | None = None
    api_calls: int | None = None
    duration: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class CostLimit:
    """Cost limit configuration."""

    id: str
    amount: float
    period: str  # daily, weekly, monthly
    created_at: datetime
    active: bool = True
    metadata: dict[str, Any] | None = None


class CostManager:
    """
    Cost management adapter with telemetry integration.

    This adapter tracks costs, enforces budget limits, and provides
    cost optimization insights with telemetry reporting.
    """

    def __init__(
        self,
        config: dict[str, Any],
        telemetry: TelemetryPort | None = None,
        storage_path: Path | None = None,
    ):
        """
        Initialize the cost manager.

        Args:
            config: Cost management configuration
            telemetry: Optional telemetry adapter for reporting
            storage_path: Path to store cost data (defaults to .artifacts/costs/)
        """
        self.config = config
        self.telemetry = telemetry

        # Storage configuration
        self.storage_path = storage_path or Path.cwd() / ".artifacts" / "costs"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.cost_entries_file = self.storage_path / "cost_entries.json"
        self.cost_limits_file = self.storage_path / "cost_limits.json"

        # Cost thresholds from config
        self.daily_limit = config.get("cost_thresholds", {}).get("daily_limit", 50.0)
        self.per_request_limit = config.get("cost_thresholds", {}).get(
            "per_request_limit", 2.0
        )
        self.warning_threshold = config.get("cost_thresholds", {}).get(
            "warning_threshold", 1.0
        )

        # Load existing data
        self.cost_entries = self._load_cost_entries()
        self.cost_limits = self._load_cost_limits()

        # Set up default daily limit if none exists
        if not self.cost_limits:
            self.set_cost_limit(self.daily_limit, "daily")

    def _load_cost_entries(self) -> list[CostEntry]:
        """Load cost entries from storage."""
        if not self.cost_entries_file.exists():
            return []

        try:
            with open(self.cost_entries_file) as f:
                entries_data = json.load(f)

            entries = []
            for entry_data in entries_data:
                # Parse datetime
                entry_data["timestamp"] = datetime.fromisoformat(
                    entry_data["timestamp"]
                )
                entries.append(CostEntry(**entry_data))

            return entries
        except Exception as e:
            if self.telemetry:
                with self.telemetry.create_span(
                    "cost_manager.load_entries_error"
                ) as span:
                    span.record_exception(e)
            return []

    def _save_cost_entries(self) -> None:
        """Save cost entries to storage."""
        try:
            entries_data = []
            for entry in self.cost_entries:
                entry_dict = asdict(entry)
                # Convert datetime to ISO format
                entry_dict["timestamp"] = entry.timestamp.isoformat()
                entries_data.append(entry_dict)

            with open(self.cost_entries_file, "w") as f:
                json.dump(entries_data, f, indent=2)
        except Exception as e:
            if self.telemetry:
                with self.telemetry.create_span(
                    "cost_manager.save_entries_error"
                ) as span:
                    span.record_exception(e)

    def _load_cost_limits(self) -> list[CostLimit]:
        """Load cost limits from storage."""
        if not self.cost_limits_file.exists():
            return []

        try:
            with open(self.cost_limits_file) as f:
                limits_data = json.load(f)

            limits = []
            for limit_data in limits_data:
                # Parse datetime
                limit_data["created_at"] = datetime.fromisoformat(
                    limit_data["created_at"]
                )
                limits.append(CostLimit(**limit_data))

            return limits
        except Exception as e:
            if self.telemetry:
                with self.telemetry.create_span(
                    "cost_manager.load_limits_error"
                ) as span:
                    span.record_exception(e)
            return []

    def _save_cost_limits(self) -> None:
        """Save cost limits to storage."""
        try:
            limits_data = []
            for limit in self.cost_limits:
                limit_dict = asdict(limit)
                # Convert datetime to ISO format
                limit_dict["created_at"] = limit.created_at.isoformat()
                limits_data.append(limit_dict)

            with open(self.cost_limits_file, "w") as f:
                json.dump(limits_data, f, indent=2)
        except Exception as e:
            if self.telemetry:
                with self.telemetry.create_span(
                    "cost_manager.save_limits_error"
                ) as span:
                    span.record_exception(e)

    def track_usage(
        self, service: str, operation: str, cost_data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Track usage and associated costs for a service operation."""
        tracking_id = str(uuid.uuid4())

        with (
            self.telemetry.create_span("cost_manager.track_usage")
            if self.telemetry
            else nullcontext()
        ) as span:
            if span:
                span.set_attributes(
                    {
                        "service": service,
                        "operation": operation,
                        "cost": cost_data.get("cost", 0),
                        "tracking_id": tracking_id,
                    }
                )

            # Create cost entry
            cost_entry = CostEntry(
                id=tracking_id,
                timestamp=datetime.now(),
                service=service,
                operation=operation,
                cost=float(cost_data.get("cost", 0)),
                tokens_used=cost_data.get("tokens_used"),
                api_calls=cost_data.get("api_calls", 1),
                duration=cost_data.get("duration"),
                metadata=kwargs,
            )

            # Add to entries
            self.cost_entries.append(cost_entry)
            self._save_cost_entries()

            # Report metrics to telemetry
            if self.telemetry:
                self.telemetry.increment_counter(
                    "cost_tracker.operations",
                    labels={"service": service, "operation": operation},
                )

                self.telemetry.record_histogram(
                    "cost_tracker.operation_cost",
                    cost_entry.cost,
                    labels={"service": service, "operation": operation},
                )

                if cost_entry.tokens_used:
                    self.telemetry.increment_counter(
                        "cost_tracker.tokens_used",
                        cost_entry.tokens_used,
                        labels={"service": service, "operation": operation},
                    )

            # Check limits and warn if necessary
            limit_status = self.check_cost_limit()
            warnings = []

            # Check per-request limit
            if cost_entry.cost > self.per_request_limit:
                warning = f"Request cost ${cost_entry.cost:.4f} exceeds limit ${self.per_request_limit:.2f}"
                warnings.append(warning)
                if self.telemetry:
                    span.add_event("cost_limit_exceeded", {"warning": warning})

            # Check warning threshold
            elif cost_entry.cost > self.warning_threshold:
                warning = f"Request cost ${cost_entry.cost:.4f} above warning threshold ${self.warning_threshold:.2f}"
                warnings.append(warning)
                if self.telemetry:
                    span.add_event("cost_warning", {"warning": warning})

            return {
                "tracking_id": tracking_id,
                "total_cost": cost_entry.cost,
                "usage_metadata": {
                    "tokens_used": cost_entry.tokens_used,
                    "api_calls": cost_entry.api_calls,
                    "duration": cost_entry.duration,
                    "warnings": warnings,
                    "within_limits": limit_status.get("within_limits", True),
                },
            }

    def get_summary(
        self,
        time_period: str | None = None,
        service_filter: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get a summary of usage costs for a specified time period."""
        # Determine time range
        now = datetime.now()
        if time_period == "daily":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_period == "weekly":
            start_time = now - timedelta(days=7)
        elif time_period == "monthly":
            start_time = now - timedelta(days=30)
        else:
            start_time = None

        # Filter entries
        filtered_entries = self.cost_entries
        if start_time:
            filtered_entries = [
                e for e in filtered_entries if e.timestamp >= start_time
            ]
        if service_filter:
            filtered_entries = [
                e for e in filtered_entries if e.service in service_filter
            ]

        # Calculate summary
        total_cost = sum(entry.cost for entry in filtered_entries)
        total_tokens = sum(entry.tokens_used or 0 for entry in filtered_entries)
        total_api_calls = sum(entry.api_calls or 0 for entry in filtered_entries)

        # Service breakdown
        service_breakdown = defaultdict(lambda: {"cost": 0, "calls": 0, "tokens": 0})
        operation_breakdown = defaultdict(lambda: {"cost": 0, "calls": 0, "tokens": 0})

        for entry in filtered_entries:
            service_breakdown[entry.service]["cost"] += entry.cost
            service_breakdown[entry.service]["calls"] += entry.api_calls or 0
            service_breakdown[entry.service]["tokens"] += entry.tokens_used or 0

            op_key = f"{entry.service}.{entry.operation}"
            operation_breakdown[op_key]["cost"] += entry.cost
            operation_breakdown[op_key]["calls"] += entry.api_calls or 0
            operation_breakdown[op_key]["tokens"] += entry.tokens_used or 0

        # Report summary metrics
        if self.telemetry:
            self.telemetry.record_gauge(
                "cost_tracker.total_cost",
                total_cost,
                labels={"period": time_period or "all_time"},
            )

        return {
            "total_cost": total_cost,
            "service_breakdown": dict(service_breakdown),
            "operation_breakdown": dict(operation_breakdown),
            "usage_stats": {
                "total_operations": len(filtered_entries),
                "total_api_calls": total_api_calls,
                "total_tokens": total_tokens,
                "average_cost_per_operation": (
                    total_cost / len(filtered_entries) if filtered_entries else 0
                ),
            },
            "summary_metadata": {
                "period": time_period,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": now.isoformat(),
                "service_filter": service_filter,
            },
        }

    def get_cost_breakdown(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get detailed cost breakdown for a specific date range."""
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=7))

        # Filter entries by date range
        filtered_entries = [
            e for e in self.cost_entries if start_date <= e.timestamp <= end_date
        ]

        # Daily breakdown
        daily_costs = defaultdict(float)
        for entry in filtered_entries:
            date_key = entry.timestamp.strftime("%Y-%m-%d")
            daily_costs[date_key] += entry.cost

        # Service and operation costs
        service_costs = defaultdict(float)
        operation_costs = defaultdict(float)

        for entry in filtered_entries:
            service_costs[entry.service] += entry.cost
            operation_costs[f"{entry.service}.{entry.operation}"] += entry.cost

        # Calculate trends
        if len(daily_costs) > 1:
            costs_list = list(daily_costs.values())
            trend_direction = (
                "increasing" if costs_list[-1] > costs_list[0] else "decreasing"
            )
            avg_daily_cost = sum(costs_list) / len(costs_list)
        else:
            trend_direction = "stable"
            avg_daily_cost = sum(daily_costs.values())

        return {
            "daily_costs": dict(daily_costs),
            "service_costs": dict(service_costs),
            "operation_costs": dict(operation_costs),
            "trends": {
                "direction": trend_direction,
                "avg_daily_cost": avg_daily_cost,
                "total_entries": len(filtered_entries),
            },
            "breakdown_metadata": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_cost": sum(daily_costs.values()),
            },
        }

    def set_cost_limit(
        self, limit_amount: float, limit_period: str = "daily", **kwargs: Any
    ) -> dict[str, Any]:
        """Set a cost limit for usage tracking."""
        limit_id = str(uuid.uuid4())

        cost_limit = CostLimit(
            id=limit_id,
            amount=limit_amount,
            period=limit_period,
            created_at=datetime.now(),
            active=True,
            metadata=kwargs,
        )

        # Deactivate existing limits for the same period
        for existing_limit in self.cost_limits:
            if existing_limit.period == limit_period and existing_limit.active:
                existing_limit.active = False

        self.cost_limits.append(cost_limit)
        self._save_cost_limits()

        if self.telemetry:
            self.telemetry.increment_counter(
                "cost_tracker.limits_set", labels={"period": limit_period}
            )

        return {
            "success": True,
            "limit_id": limit_id,
            "limit_details": {
                "amount": limit_amount,
                "period": limit_period,
                "active": True,
            },
            "limit_metadata": kwargs,
        }

    def check_cost_limit(
        self, limit_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Check current usage against cost limits."""
        now = datetime.now()

        # Get active limits to check
        limits_to_check = self.cost_limits
        if limit_id:
            limits_to_check = [
                limit for limit in self.cost_limits if limit.id == limit_id
            ]
        else:
            limits_to_check = [limit for limit in self.cost_limits if limit.active]

        limit_status = {}
        warnings = []
        overall_within_limits = True
        current_usage = 0

        for limit in limits_to_check:
            # Calculate usage for this limit's period
            if limit.period == "daily":
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif limit.period == "weekly":
                start_time = now - timedelta(days=7)
            elif limit.period == "monthly":
                start_time = now - timedelta(days=30)
            else:
                start_time = None

            if start_time:
                period_entries = [
                    e for e in self.cost_entries if e.timestamp >= start_time
                ]
                period_cost = sum(entry.cost for entry in period_entries)
            else:
                period_cost = sum(entry.cost for entry in self.cost_entries)

            current_usage = max(current_usage, period_cost)
            within_limit = period_cost <= limit.amount
            overall_within_limits &= within_limit

            usage_percentage = (
                (period_cost / limit.amount) * 100 if limit.amount > 0 else 0
            )

            limit_status[limit.id] = {
                "amount": limit.amount,
                "period": limit.period,
                "current_usage": period_cost,
                "within_limit": within_limit,
                "usage_percentage": usage_percentage,
            }

            # Generate warnings
            if usage_percentage >= 90:
                warnings.append(
                    f"{limit.period.title()} limit at {usage_percentage:.1f}% (${period_cost:.2f}/${limit.amount:.2f})"
                )
            elif usage_percentage >= 75:
                warnings.append(
                    f"{limit.period.title()} usage at {usage_percentage:.1f}% of limit"
                )

        return {
            "within_limits": overall_within_limits,
            "current_usage": current_usage,
            "limit_status": limit_status,
            "warnings": warnings,
        }

    def export_cost_data(
        self,
        export_format: str = "csv",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Export cost data in the specified format."""
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))

        # Filter entries by date range
        filtered_entries = [
            e for e in self.cost_entries if start_date <= e.timestamp <= end_date
        ]

        # Generate export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"cost_export_{timestamp}.{export_format}"
        export_path = self.storage_path / export_filename

        try:
            if export_format.lower() == "csv":
                import csv

                with open(export_path, "w", newline="") as csvfile:
                    fieldnames = [
                        "id",
                        "timestamp",
                        "service",
                        "operation",
                        "cost",
                        "tokens_used",
                        "api_calls",
                        "duration",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for entry in filtered_entries:
                        writer.writerow(
                            {
                                "id": entry.id,
                                "timestamp": entry.timestamp.isoformat(),
                                "service": entry.service,
                                "operation": entry.operation,
                                "cost": entry.cost,
                                "tokens_used": entry.tokens_used,
                                "api_calls": entry.api_calls,
                                "duration": entry.duration,
                            }
                        )

            elif export_format.lower() == "json":
                export_data = []
                for entry in filtered_entries:
                    entry_dict = asdict(entry)
                    entry_dict["timestamp"] = entry.timestamp.isoformat()
                    export_data.append(entry_dict)

                with open(export_path, "w") as jsonfile:
                    json.dump(export_data, jsonfile, indent=2)

            else:
                raise ValueError(f"Unsupported export format: {export_format}")

            if self.telemetry:
                self.telemetry.increment_counter(
                    "cost_tracker.exports", labels={"format": export_format}
                )

            return {
                "success": True,
                "export_path": str(export_path),
                "export_format": export_format,
                "export_metadata": {
                    "entries_exported": len(filtered_entries),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_cost": sum(e.cost for e in filtered_entries),
                },
            }

        except Exception as e:
            if self.telemetry:
                with self.telemetry.create_span("cost_manager.export_error") as span:
                    span.record_exception(e)

            return {
                "success": False,
                "error": str(e),
                "export_format": export_format,
                "export_metadata": {},
            }


# Context manager for null operations when telemetry is not available
class nullcontext:
    """Null context manager for when telemetry is not available."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return None
