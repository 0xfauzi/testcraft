"""
Telemetry Port interface definition.

This module defines the interface for telemetry operations,
including span tracking and metric recording.
"""

from typing import Dict, Any, Optional, List, Union
from typing_extensions import Protocol
from datetime import datetime


class TelemetryPort(Protocol):
    """
    Interface for telemetry operations.
    
    This protocol defines the contract for telemetry operations, including
    span tracking, metric recording, and observability data collection.
    """
    
    def start_span(
        self,
        span_name: str,
        span_type: str = "operation",
        **kwargs: Any
    ) -> str:
        """
        Start a new telemetry span for tracking operations.
        
        Args:
            span_name: Name of the span
            span_type: Type of span (operation, request, etc.)
            **kwargs: Additional span parameters
            
        Returns:
            Span ID for tracking the span
            
        Raises:
            TelemetryError: If span creation fails
        """
        ...
    
    def end_span(
        self,
        span_id: str,
        status: str = "success",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        End a telemetry span and record its completion.
        
        Args:
            span_id: ID of the span to end
            status: Status of the span completion (success, error, etc.)
            **kwargs: Additional span ending parameters
            
        Returns:
            Dictionary containing:
                - 'span_id': ID of the ended span
                - 'duration': Duration of the span
                - 'status': Final status of the span
                - 'span_metadata': Additional span metadata
                
        Raises:
            TelemetryError: If span ending fails
        """
        ...
    
    def record_metric(
        self,
        metric_name: str,
        metric_value: Union[int, float],
        metric_type: str = "counter",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Record a telemetry metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            metric_type: Type of metric (counter, gauge, histogram, etc.)
            **kwargs: Additional metric parameters
            
        Returns:
            Dictionary containing:
                - 'metric_id': Unique identifier for the recorded metric
                - 'metric_name': Name of the recorded metric
                - 'metric_value': Value that was recorded
                - 'metric_metadata': Additional metric metadata
                
        Raises:
            TelemetryError: If metric recording fails
        """
        ...
    
    def add_span_attribute(
        self,
        span_id: str,
        attribute_name: str,
        attribute_value: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Add an attribute to an existing span.
        
        Args:
            span_id: ID of the span to add the attribute to
            attribute_name: Name of the attribute
            attribute_value: Value of the attribute
            **kwargs: Additional attribute parameters
            
        Returns:
            Dictionary containing:
                - 'success': Whether the attribute was added successfully
                - 'span_id': ID of the span
                - 'attribute_name': Name of the added attribute
                - 'attribute_metadata': Additional attribute metadata
                
        Raises:
            TelemetryError: If attribute addition fails
        """
        ...
    
    def record_event(
        self,
        event_name: str,
        event_data: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Record a telemetry event.
        
        Args:
            event_name: Name of the event
            event_data: Data associated with the event
            **kwargs: Additional event parameters
            
        Returns:
            Dictionary containing:
                - 'event_id': Unique identifier for the recorded event
                - 'event_name': Name of the recorded event
                - 'event_timestamp': Timestamp when the event was recorded
                - 'event_metadata': Additional event metadata
                
        Raises:
            TelemetryError: If event recording fails
        """
        ...
    
    def get_telemetry_summary(
        self,
        time_period: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get a summary of telemetry data for a specified time period.
        
        Args:
            time_period: Time period for the summary (e.g., 'hourly', 'daily')
            **kwargs: Additional summary parameters
            
        Returns:
            Dictionary containing:
                - 'span_summary': Summary of span data
                - 'metric_summary': Summary of metric data
                - 'event_summary': Summary of event data
                - 'performance_metrics': Performance-related metrics
                - 'summary_metadata': Additional summary metadata
                
        Raises:
            TelemetryError: If summary generation fails
        """
        ...
