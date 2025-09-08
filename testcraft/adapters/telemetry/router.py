"""
Telemetry adapter router/factory.

This module provides a factory pattern for creating telemetry adapters
based on configuration, making it easy to switch between different
telemetry backends like OpenTelemetry, Datadog, etc.
"""

from typing import Dict, Any, Optional, Type
from ...ports.telemetry_port import TelemetryPort
from .opentelemetry_adapter import OpenTelemetryAdapter
from .noop_adapter import NoOpTelemetryAdapter


class TelemetryAdapterRegistry:
    """Registry for telemetry adapter implementations."""
    
    def __init__(self):
        self._adapters: Dict[str, Type[TelemetryPort]] = {}
        self._register_builtin_adapters()
    
    def _register_builtin_adapters(self) -> None:
        """Register built-in telemetry adapters."""
        self.register('opentelemetry', OpenTelemetryAdapter)
        self.register('noop', NoOpTelemetryAdapter)
        self.register('disabled', NoOpTelemetryAdapter)
    
    def register(self, name: str, adapter_class: Type[TelemetryPort]) -> None:
        """
        Register a telemetry adapter implementation.
        
        Args:
            name: Name/identifier for the adapter
            adapter_class: Adapter class implementing TelemetryPort
        """
        self._adapters[name.lower()] = adapter_class
    
    def get_adapter_class(self, name: str) -> Type[TelemetryPort]:
        """
        Get adapter class by name.
        
        Args:
            name: Name of the adapter
            
        Returns:
            Adapter class implementing TelemetryPort
            
        Raises:
            ValueError: If adapter name is not registered
        """
        adapter_class = self._adapters.get(name.lower())
        if not adapter_class:
            available = list(self._adapters.keys())
            raise ValueError(f"Unknown telemetry adapter '{name}'. Available: {available}")
        return adapter_class
    
    def list_adapters(self) -> Dict[str, Type[TelemetryPort]]:
        """Get all registered adapters."""
        return self._adapters.copy()


# Global registry instance
_telemetry_registry = TelemetryAdapterRegistry()


def create_telemetry_adapter(config: Dict[str, Any]) -> TelemetryPort:
    """
    Factory function to create telemetry adapter based on configuration.
    
    Args:
        config: Telemetry configuration dictionary
        
    Returns:
        Telemetry adapter instance
        
    Configuration should include:
        - 'enabled': bool - Whether telemetry is enabled
        - 'backend': str - Backend type ('opentelemetry', 'datadog', 'noop')
        - 'opt_out_data_collection': bool - Complete opt-out override
        - Backend-specific configuration options
    """
    # Handle opt-out and disabled states
    if config.get('opt_out_data_collection', False) or not config.get('enabled', False):
        return NoOpTelemetryAdapter(config)
    
    # Get backend type
    backend = config.get('backend', 'opentelemetry').lower()
    
    # Special handling for known backends that might need fallbacks
    if backend == 'opentelemetry':
        try:
            # Try to create OpenTelemetry adapter
            adapter_class = _telemetry_registry.get_adapter_class(backend)
            adapter = adapter_class(config)
            
            # If OpenTelemetry is not available, the adapter will be disabled
            # but still functional (graceful degradation)
            return adapter
            
        except Exception:
            # Fallback to no-op if OpenTelemetry setup fails
            return NoOpTelemetryAdapter(config)
    
    elif backend in ['datadog', 'jaeger']:
        # These backends aren't implemented yet, fallback to no-op
        # In the future, these would have their own adapter implementations
        return NoOpTelemetryAdapter(config)
    
    else:
        # Get adapter from registry
        try:
            adapter_class = _telemetry_registry.get_adapter_class(backend)
            return adapter_class(config)
        except ValueError:
            # Unknown backend, fallback to no-op
            return NoOpTelemetryAdapter(config)


def register_telemetry_adapter(name: str, adapter_class: Type[TelemetryPort]) -> None:
    """
    Register a custom telemetry adapter.
    
    Args:
        name: Name/identifier for the adapter
        adapter_class: Adapter class implementing TelemetryPort
    """
    _telemetry_registry.register(name, adapter_class)


def get_available_backends() -> Dict[str, Type[TelemetryPort]]:
    """Get all available telemetry backends."""
    return _telemetry_registry.list_adapters()


class TelemetryContextManager:
    """
    Context manager for telemetry operations.
    
    Provides convenient access to telemetry functionality with
    automatic span management and error handling.
    """
    
    def __init__(self, telemetry: TelemetryPort, operation_name: str, **span_attributes):
        """
        Initialize telemetry context manager.
        
        Args:
            telemetry: Telemetry adapter instance
            operation_name: Name of the operation being traced
            **span_attributes: Additional attributes for the span
        """
        self.telemetry = telemetry
        self.operation_name = operation_name
        self.span_attributes = span_attributes
        self._span_context = None
    
    def __enter__(self):
        """Enter the telemetry context."""
        self._span_context = self.telemetry.create_span(
            self.operation_name,
            attributes=self.span_attributes
        ).__enter__()
        return self._span_context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the telemetry context."""
        if self._span_context:
            # Record exception if one occurred
            if exc_type and hasattr(self._span_context, 'record_exception'):
                self._span_context.record_exception(exc_val)
            
            # Exit the span context
            self._span_context.__exit__(exc_type, exc_val, exc_tb)
        
        return False  # Don't suppress exceptions


def create_telemetry_context(
    telemetry: TelemetryPort,
    operation_name: str,
    **span_attributes
) -> TelemetryContextManager:
    """
    Create a telemetry context manager for an operation.
    
    Args:
        telemetry: Telemetry adapter instance
        operation_name: Name of the operation being traced
        **span_attributes: Additional attributes for the span
        
    Returns:
        Context manager for telemetry operations
        
    Usage:
        with create_telemetry_context(telemetry, "llm_call", model="o4-mini") as span:
            # Perform LLM operation
            span.set_attribute("tokens_used", 150)
    """
    return TelemetryContextManager(telemetry, operation_name, **span_attributes)
