"""
Artifact storage functions for evaluation results.

This module provides utilities for storing evaluation results, batch summaries,
and other evaluation artifacts for later analysis and reporting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ...ports.evaluation_port import EvaluationResult
from ..io.artifact_store import ArtifactStoreAdapter, ArtifactType

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Handles storage and management of evaluation artifacts."""

    def __init__(self, artifact_store: ArtifactStoreAdapter):
        """
        Initialize artifact manager.

        Args:
            artifact_store: Artifact storage adapter
        """
        self.artifact_store = artifact_store

    def store_evaluation_artifact(self, result: EvaluationResult) -> None:
        """Store evaluation result as artifact."""
        try:
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                result.to_dict(),
                artifact_id=f"eval_{result.test_id}",
                tags=["evaluation", "single", Path(result.source_file).stem],
                description=f"Evaluation result for {result.source_file}",
            )
        except Exception as e:
            logger.warning(f"Failed to store evaluation artifact: {e}")

    def create_batch_summary(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Create summary of batch evaluation results."""
        successful_count = sum(1 for r in results if r.acceptance.all_checks_pass)

        llm_scores = [
            r.llm_judge.overall_score
            for r in results
            if r.llm_judge and r.llm_judge.overall_score > 0
        ]

        return {
            "total_tests": len(results),
            "successful_tests": successful_count,
            "success_rate": successful_count / len(results) if results else 0.0,
            "average_llm_score": (
                sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def store_batch_summary(
        self, results: list[EvaluationResult], tags: list[str] | None = None
    ) -> None:
        """Store batch evaluation summary as artifact."""
        try:
            batch_summary = self.create_batch_summary(results)
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                batch_summary,
                tags=tags or ["batch", "evaluation", "summary"],
                description=f"Batch evaluation of {len(results)} tests",
            )
        except Exception as e:
            logger.warning(f"Failed to store batch summary: {e}")

    def store_statistical_analysis(
        self,
        analysis_result: dict[str, Any],
        analysis_type: str,
        tags: list[str] | None = None,
    ) -> None:
        """Store statistical analysis results as artifact."""
        try:
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                analysis_result,
                tags=tags or ["statistical", "significance", "analysis"],
                description=f"Statistical significance analysis: {analysis_type}",
            )
        except Exception as e:
            logger.warning(f"Failed to store statistical analysis: {e}")

    def store_bias_analysis(
        self,
        bias_result: dict[str, Any],
        evaluation_count: int,
        tags: list[str] | None = None,
    ) -> None:
        """Store bias analysis results as artifact."""
        try:
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                bias_result,
                tags=tags or ["bias", "mitigation", "fairness"],
                description=f"Bias detection analysis on {evaluation_count} evaluations",
            )
        except Exception as e:
            logger.warning(f"Failed to store bias analysis: {e}")

    def store_golden_repo_results(
        self,
        golden_repo_result: dict[str, Any],
        golden_repo_path: Path,
        tags: list[str] | None = None,
    ) -> None:
        """Store golden repository evaluation results as artifact."""
        try:
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                golden_repo_result,
                tags=tags or ["golden_repo", "regression", "evaluation"],
                description=f"Golden repo evaluation: {golden_repo_path.name}",
            )
        except Exception as e:
            logger.warning(f"Failed to store golden repo results: {e}")

    def store_ab_testing_results(
        self,
        pipeline_results: dict[str, Any],
        variant_count: int,
        tags: list[str] | None = None,
    ) -> None:
        """Store comprehensive A/B testing pipeline results as artifact."""
        try:
            self.artifact_store.store_artifact(
                ArtifactType.ANALYSIS_REPORT,
                pipeline_results,
                tags=tags or ["ab_testing", "pipeline", "comprehensive"],
                description=f"Advanced A/B testing pipeline: {variant_count} variants",
            )
        except Exception as e:
            logger.warning(f"Failed to store A/B testing results: {e}")
