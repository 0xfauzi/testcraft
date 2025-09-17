"""
Telemetry and metrics recording for GenerateUseCase.

This module handles all telemetry operations including span management,
metrics recording, and cost tracking for the test generation process.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ....domain.models import GenerationResult
from ....ports.telemetry_port import MetricValue, TelemetryPort

logger = logging.getLogger(__name__)


class TelemetryService:
    """
    Service for handling telemetry and metrics recording.
    
    Provides centralized telemetry operations for the test generation pipeline
    including per-file metrics, validation metrics, and cost tracking.
    """
    
    def __init__(
        self,
        telemetry_port: TelemetryPort,
        config: dict[str, Any],
    ):
        """Initialize telemetry service with required ports."""
        self._telemetry = telemetry_port
        self._config = config
    
    async def record_final_telemetry_and_costs(
        self,
        span,
        generation_results: list[GenerationResult],
        refinement_results: list[dict[str, Any]],
    ) -> None:
        """Record final telemetry metrics and cost information."""
        try:
            # Record metrics
            metrics = [
                MetricValue(
                    name="tests_generated_total",
                    value=len(generation_results),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="tests_generated_successful",
                    value=self._count_successful_generations(generation_results),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="tests_refined_total",
                    value=len(refinement_results),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="tests_refined_successful",
                    value=sum(1 for r in refinement_results if r.get("success", False)),
                    unit="count",
                    labels={"framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                ),
            ]
            
            self._telemetry.record_metrics(metrics)
            
            # Record span attributes
            span.set_attribute("total_generation_results", len(generation_results))
            span.set_attribute("successful_generations", self._count_successful_generations(generation_results))
            span.set_attribute("total_refinement_results", len(refinement_results))
            span.set_attribute(
                "successful_refinements",
                sum(1 for r in refinement_results if r.get("success", False)),
            )
            
            # Flush telemetry data
            self._telemetry.flush(timeout_seconds=5.0)
            
        except Exception as e:
            logger.warning("Failed to record telemetry and costs: %s", e)
            # Don't fail the entire operation for telemetry issues
    
    def _count_successful_generations(self, generation_results: list[GenerationResult]) -> int:
        """Count successful generation results."""
        return sum(
            1
            for r in generation_results
            if (
                (getattr(r, "success", False))
                if not isinstance(r, dict)
                else bool(r.get("success", False))
            )
        )
    
    async def record_per_file_telemetry(self, file_result: dict[str, Any], file_path: str) -> None:
        """
        Record per-file telemetry metrics and spans.
        
        Args:
            file_result: Result dictionary from processing a single plan  
            file_path: Path to the file being processed
        """
        try:
            # Create per-file span for detailed tracing
            with self._telemetry.create_child_span("process_file_immediate") as file_span:
                file_span.set_attribute("file_path", file_path)
                file_span.set_attribute("success", file_result.get("success", False))
                
                # Generation stage metrics
                if file_result.get("generation_result"):
                    gen_result = file_result["generation_result"] 
                    file_span.set_attribute("generation_success", gen_result.success)
                    if gen_result.content:
                        file_span.set_attribute("content_length", len(gen_result.content))
                
                # Write stage metrics  
                if file_result.get("write_result"):
                    write_result = file_result["write_result"]
                    file_span.set_attribute("write_success", write_result.get("success", False))
                    file_span.set_attribute("bytes_written", write_result.get("bytes_written", 0))
                
                # Refinement stage metrics
                if file_result.get("refinement_result"):
                    refine_result = file_result["refinement_result"]
                    file_span.set_attribute("refinement_success", refine_result.get("success", False))
                    file_span.set_attribute("refinement_iterations", refine_result.get("iterations", 0))
                
                # Record errors if any
                if file_result.get("errors"):
                    file_span.set_attribute("error_count", len(file_result["errors"]))
                    file_span.set_attribute("errors", "; ".join(file_result["errors"]))
            
            # Record individual stage metrics
            await self._record_per_file_metrics(file_result, file_path)
                
        except Exception as e:
            logger.warning("Failed to record per-file telemetry for %s: %s", file_path, e)
    
    async def _record_per_file_metrics(self, file_result: dict[str, Any], file_path: str) -> None:
        """Record individual stage metrics for a file."""
        metrics = []
        
        # Generation metrics
        if file_result.get("generation_result"):
            gen_result = file_result["generation_result"]
            metrics.append(
                MetricValue(
                    name="file_generation_success",
                    value=1 if gen_result.success else 0,
                    unit="count",
                    labels={"file_path": file_path, "framework": self._config["test_framework"]},
                    timestamp=datetime.now(),
                )
            )
        
        # Write metrics
        if file_result.get("write_result"):
            write_result = file_result["write_result"]
            metrics.append(
                MetricValue(
                    name="file_write_success", 
                    value=1 if write_result.get("success", False) else 0,
                    unit="count",
                    labels={"file_path": file_path},
                    timestamp=datetime.now(),
                )
            )
        
        # Refinement metrics
        if file_result.get("refinement_result"):
            refine_result = file_result["refinement_result"]
            metrics.extend([
                MetricValue(
                    name="file_refinement_success",
                    value=1 if refine_result.get("success", False) else 0,
                    unit="count", 
                    labels={"file_path": file_path},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="file_refinement_iterations",
                    value=refine_result.get("iterations", 0),
                    unit="count",
                    labels={"file_path": file_path},
                    timestamp=datetime.now(),
                )
            ])
        
        # Record all metrics
        if metrics:
            self._telemetry.record_metrics(metrics)
    
    def record_module_path_telemetry(self, module_path_info: dict[str, Any], source_path: Path) -> None:
        """
        Record telemetry metrics for module path derivation.
        
        Args:
            module_path_info: Result from ModulePathDeriver.derive_module_path
            source_path: Path to source file
        """
        try:
            validation_status = module_path_info.get("validation_status", "unknown")
            module_path = module_path_info.get("module_path", "")
            
            # Record success/failure metrics
            metrics = [
                MetricValue(
                    name="module_path_derived_total",
                    value=1,
                    unit="count",
                    labels={"file_path": str(source_path), "status": validation_status},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="module_path_derived_success", 
                    value=1 if validation_status == "validated" else 0,
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
            ]
            
            # Record validation status breakdown
            if validation_status in ["validated", "unvalidated", "failed", "error"]:
                metrics.append(
                    MetricValue(
                        name=f"module_path_status_{validation_status}",
                        value=1,
                        unit="count",
                        labels={"file_path": str(source_path)},
                        timestamp=datetime.now(),
                    )
                )
            
            # Record fallback usage
            fallback_paths = module_path_info.get("fallback_paths", [])
            if fallback_paths:
                metrics.append(
                    MetricValue(
                        name="module_path_fallback_used",
                        value=1,
                        unit="count",
                        labels={"file_path": str(source_path), "fallback_count": str(len(fallback_paths))},
                        timestamp=datetime.now(),
                    )
                )
            
            # Record module path pattern (for analytics)
            if module_path:
                has_src = "src" in module_path
                path_depth = len(module_path.split("."))
                metrics.extend([
                    MetricValue(
                        name="module_path_has_src",
                        value=1 if has_src else 0,
                        unit="count",
                        labels={"file_path": str(source_path)},
                        timestamp=datetime.now(),
                    ),
                    MetricValue(
                        name="module_path_depth",
                        value=path_depth,
                        unit="count",
                        labels={"file_path": str(source_path)},
                        timestamp=datetime.now(),
                    )
                ])
            
            # Record all metrics
            if metrics:
                self._telemetry.record_metrics(metrics)
                
        except Exception as e:
            logger.warning("Failed to record module path telemetry for %s: %s", source_path, e)
    
    def record_validation_telemetry(self, issues: list, source_path: Path) -> None:
        """
        Record telemetry metrics for test validation.
        
        Args:
            issues: List of validation issues
            source_path: Path to source file
        """
        try:
            # Count issues by severity and category
            error_count = sum(1 for issue in issues if issue.severity == "error")
            warning_count = sum(1 for issue in issues if issue.severity == "warning")
            
            # Count by category
            category_counts = {}
            for issue in issues:
                category = issue.category
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Record metrics
            metrics = [
                MetricValue(
                    name="test_validation_total_issues",
                    value=len(issues),
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="test_validation_errors",
                    value=error_count,
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
                MetricValue(
                    name="test_validation_warnings",
                    value=warning_count,
                    unit="count",
                    labels={"file_path": str(source_path)},
                    timestamp=datetime.now(),
                ),
            ]
            
            # Add category-specific metrics
            for category, count in category_counts.items():
                metrics.append(
                    MetricValue(
                        name=f"test_validation_{category}_issues",
                        value=count,
                        unit="count",
                        labels={"file_path": str(source_path)},
                        timestamp=datetime.now(),
                    )
                )
            
            # Record all metrics
            if metrics:
                self._telemetry.record_metrics(metrics)
                
        except Exception as e:
            logger.warning("Failed to record validation telemetry for %s: %s", source_path, e)
