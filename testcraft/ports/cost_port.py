"""
Cost Port interface definition.

This module defines the interface for cost tracking operations,
including usage tracking and cost summaries.
"""

from datetime import datetime
from typing import Any

from typing_extensions import Protocol


class CostPort(Protocol):
    """
    Interface for cost tracking operations.

    This protocol defines the contract for tracking usage costs,
    including API calls, resource usage, and cost summaries.
    """

    def track_usage(
        self, service: str, operation: str, cost_data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Track usage and associated costs for a service operation.

        Args:
            service: Name of the service being used
            operation: Specific operation being performed
            cost_data: Dictionary containing cost information
            **kwargs: Additional tracking parameters

        Cost data should contain:
            - 'tokens_used': Number of tokens used (if applicable)
            - 'api_calls': Number of API calls made
            - 'duration': Duration of the operation
            - 'cost': Cost of the operation

        Returns:
            Dictionary containing:
                - 'tracking_id': Unique identifier for this usage tracking
                - 'total_cost': Total cost for this operation
                - 'usage_metadata': Additional usage metadata

        Raises:
            CostError: If usage tracking fails
        """
        ...

    def get_summary(
        self,
        time_period: str | None = None,
        service_filter: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get a summary of usage costs for a specified time period.

        Args:
            time_period: Time period for the summary (e.g., 'daily', 'weekly', 'monthly')
            service_filter: Optional list of services to include in summary
            **kwargs: Additional summary parameters

        Returns:
            Dictionary containing:
                - 'total_cost': Total cost for the period
                - 'service_breakdown': Cost breakdown by service
                - 'operation_breakdown': Cost breakdown by operation
                - 'usage_stats': Usage statistics for the period
                - 'summary_metadata': Additional summary metadata

        Raises:
            CostError: If summary generation fails
        """
        ...

    def get_cost_breakdown(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get detailed cost breakdown for a specific date range.

        Args:
            start_date: Start date for the breakdown
            end_date: End date for the breakdown
            **kwargs: Additional breakdown parameters

        Returns:
            Dictionary containing:
                - 'daily_costs': Daily cost breakdown
                - 'service_costs': Cost breakdown by service
                - 'operation_costs': Cost breakdown by operation
                - 'trends': Cost trends and patterns
                - 'breakdown_metadata': Additional breakdown metadata

        Raises:
            CostError: If cost breakdown fails
        """
        ...

    def set_cost_limit(
        self, limit_amount: float, limit_period: str = "daily", **kwargs: Any
    ) -> dict[str, Any]:
        """
        Set a cost limit for usage tracking.

        Args:
            limit_amount: Maximum cost amount allowed
            limit_period: Period for the limit (daily, weekly, monthly)
            **kwargs: Additional limit parameters

        Returns:
            Dictionary containing:
                - 'success': Whether the limit was set successfully
                - 'limit_id': Unique identifier for the limit
                - 'limit_details': Details about the set limit
                - 'limit_metadata': Additional limit metadata

        Raises:
            CostError: If limit setting fails
        """
        ...

    def check_cost_limit(
        self, limit_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Check current usage against cost limits.

        Args:
            limit_id: Optional specific limit to check (checks all if None)
            **kwargs: Additional checking parameters

        Returns:
            Dictionary containing:
                - 'within_limits': Whether usage is within limits
                - 'current_usage': Current usage amount
                - 'limit_status': Status of each limit
                - 'warnings': List of any warnings about approaching limits

        Raises:
            CostError: If limit checking fails
        """
        ...

    def export_cost_data(
        self,
        export_format: str = "csv",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Export cost data in the specified format.

        Args:
            export_format: Format to export data in (csv, json, excel)
            start_date: Optional start date for export
            end_date: Optional end date for export
            **kwargs: Additional export parameters

        Returns:
            Dictionary containing:
                - 'success': Whether the export succeeded
                - 'export_path': Path to the exported file
                - 'export_format': Format of the exported data
                - 'export_metadata': Additional export metadata

        Raises:
            CostError: If data export fails
        """
        ...
