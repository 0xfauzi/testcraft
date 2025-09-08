"""
Artifact storage adapter for managing test generation artifacts.

This module provides an adapter for storing, retrieving, and managing
artifacts from test generation runs including coverage reports, generated
tests, LLM responses, and run history with configurable cleanup policies.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactType(str, Enum):
    """Enumeration of supported artifact types."""

    COVERAGE_REPORT = "coverage_report"
    GENERATED_TEST = "generated_test"
    LLM_RESPONSE = "llm_response"
    RUN_HISTORY = "run_history"
    ANALYSIS_REPORT = "analysis_report"
    GENERATION_PLAN = "generation_plan"


@dataclass
class ArtifactMetadata:
    """Metadata for stored artifacts."""

    artifact_id: str
    artifact_type: ArtifactType
    timestamp: datetime
    file_path: str
    size_bytes: int
    tags: list[str]
    description: str | None = None
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
            "description": self.description,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactMetadata":
        """Create from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=ArtifactType(data["artifact_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            file_path=data["file_path"],
            size_bytes=data["size_bytes"],
            tags=data["tags"],
            description=data.get("description"),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
        )


@dataclass
class CleanupPolicy:
    """Configuration for artifact cleanup policies."""

    max_age_days: int | None = None
    max_artifacts: int | None = None
    max_size_mb: int | None = None
    preserve_tags: list[str] = None

    def __post_init__(self) -> None:
        if self.preserve_tags is None:
            self.preserve_tags = []


class ArtifactStoreError(Exception):
    """Exception raised when artifact storage operations fail."""

    pass


class ArtifactStoreAdapter:
    """
    Artifact storage adapter for managing test generation artifacts.

    This adapter provides methods for storing, retrieving, and cleaning up
    various types of artifacts generated during test creation processes.
    """

    def __init__(
        self,
        base_path: str | Path = ".testcraft/artifacts",
        cleanup_policy: CleanupPolicy | None = None,
    ) -> None:
        """
        Initialize the artifact store adapter.

        Args:
            base_path: Base directory for storing artifacts
            cleanup_policy: Policy for automatic cleanup of old artifacts
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.base_path / "metadata.json"
        self.cleanup_policy = cleanup_policy or CleanupPolicy(
            max_age_days=30, max_artifacts=100
        )

        # Create subdirectories for different artifact types
        for artifact_type in ArtifactType:
            (self.base_path / artifact_type.value).mkdir(exist_ok=True)

        self._load_metadata()

    def store_artifact(
        self,
        artifact_type: ArtifactType,
        content: str | bytes | dict[str, Any],
        artifact_id: str | None = None,
        tags: list[str] | None = None,
        description: str | None = None,
        expiry_days: int | None = None,
    ) -> str:
        """
        Store an artifact in the store.

        Args:
            artifact_type: Type of artifact to store
            content: Content to store (string, bytes, or dict for JSON)
            artifact_id: Optional custom ID (will generate if not provided)
            tags: Optional tags for categorization
            description: Optional description
            expiry_days: Optional expiry in days from now

        Returns:
            The artifact ID

        Raises:
            ArtifactStoreError: If storage fails
        """
        try:
            # Generate artifact ID if not provided
            if artifact_id is None:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                artifact_id = f"{artifact_type.value}_{timestamp}"

            # Determine file extension based on content type
            if isinstance(content, dict):
                file_ext = ".json"
                file_content = json.dumps(content, indent=2, ensure_ascii=False)
            elif isinstance(content, bytes):
                file_ext = ".bin"
                file_content = content
            else:
                file_ext = ".txt"
                file_content = str(content)

            # Create file path
            file_name = f"{artifact_id}{file_ext}"
            file_path = self.base_path / artifact_type.value / file_name

            # Write content to file
            if isinstance(file_content, bytes):
                file_path.write_bytes(file_content)
            else:
                file_path.write_text(file_content, encoding="utf-8")

            # Create metadata
            file_size = file_path.stat().st_size
            expires_at = None
            if expiry_days:
                expires_at = datetime.utcnow() + timedelta(days=expiry_days)

            metadata = ArtifactMetadata(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                timestamp=datetime.utcnow(),
                file_path=str(file_path.relative_to(self.base_path)),
                size_bytes=file_size,
                tags=tags or [],
                description=description,
                expires_at=expires_at,
            )

            # Store metadata
            self._metadata[artifact_id] = metadata
            self._save_metadata()

            return artifact_id

        except Exception as e:
            raise ArtifactStoreError(f"Failed to store artifact: {str(e)}")

    def retrieve_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: ID of the artifact to retrieve

        Returns:
            Dictionary containing artifact data and metadata, or None if not found
        """
        try:
            metadata = self._metadata.get(artifact_id)
            if not metadata:
                return None

            # Check if artifact has expired
            if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                self.remove_artifact(artifact_id)
                return None

            file_path = self.base_path / metadata.file_path
            if not file_path.exists():
                # Cleanup stale metadata
                del self._metadata[artifact_id]
                self._save_metadata()
                return None

            # Read content based on file type
            if file_path.suffix == ".json":
                content = json.loads(file_path.read_text(encoding="utf-8"))
            elif file_path.suffix == ".bin":
                content = file_path.read_bytes()
            else:
                content = file_path.read_text(encoding="utf-8")

            return {
                "artifact_id": artifact_id,
                "content": content,
                "metadata": metadata.to_dict(),
            }

        except Exception as e:
            raise ArtifactStoreError(
                f"Failed to retrieve artifact {artifact_id}: {str(e)}"
            )

    def list_artifacts(
        self,
        artifact_type: ArtifactType | None = None,
        tags: list[str] | None = None,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List artifacts matching the given criteria.

        Args:
            artifact_type: Filter by artifact type
            tags: Filter by tags (artifacts must have all specified tags)
            include_expired: Whether to include expired artifacts

        Returns:
            List of artifact metadata dictionaries
        """
        try:
            results = []
            current_time = datetime.utcnow()

            for _artifact_id, metadata in self._metadata.items():
                # Skip expired artifacts unless requested
                if (
                    not include_expired
                    and metadata.expires_at
                    and current_time > metadata.expires_at
                ):
                    continue

                # Filter by artifact type
                if artifact_type and metadata.artifact_type != artifact_type:
                    continue

                # Filter by tags (must have all specified tags)
                if tags and not all(tag in metadata.tags for tag in tags):
                    continue

                results.append(metadata.to_dict())

            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            return results

        except Exception as e:
            raise ArtifactStoreError(f"Failed to list artifacts: {str(e)}")

    def remove_artifact(self, artifact_id: str) -> bool:
        """
        Remove an artifact from the store.

        Args:
            artifact_id: ID of the artifact to remove

        Returns:
            True if artifact was removed, False if it didn't exist
        """
        try:
            metadata = self._metadata.get(artifact_id)
            if not metadata:
                return False

            # Remove file
            file_path = self.base_path / metadata.file_path
            if file_path.exists():
                file_path.unlink()

            # Remove metadata
            del self._metadata[artifact_id]
            self._save_metadata()

            return True

        except Exception as e:
            raise ArtifactStoreError(
                f"Failed to remove artifact {artifact_id}: {str(e)}"
            )

    def cleanup_expired(self) -> dict[str, int]:
        """
        Remove expired artifacts based on cleanup policy.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            current_time = datetime.utcnow()
            removed_count = 0
            total_size_freed = 0

            # Find expired artifacts
            expired_ids = []
            for artifact_id, metadata in self._metadata.items():
                if metadata.expires_at and current_time > metadata.expires_at:
                    expired_ids.append(artifact_id)

            # Remove expired artifacts
            for artifact_id in expired_ids:
                metadata = self._metadata[artifact_id]
                if self.remove_artifact(artifact_id):
                    removed_count += 1
                    total_size_freed += metadata.size_bytes

            return {
                "expired_artifacts_removed": removed_count,
                "size_freed_bytes": total_size_freed,
            }

        except Exception as e:
            raise ArtifactStoreError(f"Failed to cleanup expired artifacts: {str(e)}")

    def apply_cleanup_policy(self) -> dict[str, int]:
        """
        Apply the configured cleanup policy to remove old artifacts.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            stats = self.cleanup_expired()
            current_time = datetime.utcnow()
            policy = self.cleanup_policy

            # Apply age-based cleanup
            if policy.max_age_days:
                cutoff_date = current_time - timedelta(days=policy.max_age_days)
                old_artifacts = []

                for artifact_id, metadata in self._metadata.items():
                    if metadata.timestamp < cutoff_date and not any(
                        tag in policy.preserve_tags for tag in metadata.tags
                    ):
                        old_artifacts.append((artifact_id, metadata))

                for artifact_id, metadata in old_artifacts:
                    if self.remove_artifact(artifact_id):
                        stats["expired_artifacts_removed"] += 1
                        stats["size_freed_bytes"] += metadata.size_bytes

            # Apply count-based cleanup
            if policy.max_artifacts and len(self._metadata) > policy.max_artifacts:
                # Sort by timestamp (oldest first)
                artifacts_by_age = sorted(
                    self._metadata.items(), key=lambda x: x[1].timestamp
                )

                excess_count = len(self._metadata) - policy.max_artifacts
                for artifact_id, metadata in artifacts_by_age[:excess_count]:
                    if not any(tag in policy.preserve_tags for tag in metadata.tags):
                        if self.remove_artifact(artifact_id):
                            stats["expired_artifacts_removed"] += 1
                            stats["size_freed_bytes"] += metadata.size_bytes

            # Apply size-based cleanup
            if policy.max_size_mb:
                max_size_bytes = policy.max_size_mb * 1024 * 1024
                total_size = sum(
                    metadata.size_bytes for metadata in self._metadata.values()
                )

                if total_size > max_size_bytes:
                    # Remove oldest artifacts until under size limit
                    artifacts_by_age = sorted(
                        self._metadata.items(), key=lambda x: x[1].timestamp
                    )

                    for artifact_id, metadata in artifacts_by_age:
                        if total_size <= max_size_bytes:
                            break

                        if not any(
                            tag in policy.preserve_tags for tag in metadata.tags
                        ):
                            if self.remove_artifact(artifact_id):
                                stats["expired_artifacts_removed"] += 1
                                stats["size_freed_bytes"] += metadata.size_bytes
                                total_size -= metadata.size_bytes

            return stats

        except Exception as e:
            raise ArtifactStoreError(f"Failed to apply cleanup policy: {str(e)}")

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get statistics about the artifact store.

        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                "total_artifacts": len(self._metadata),
                "total_size_bytes": sum(
                    metadata.size_bytes for metadata in self._metadata.values()
                ),
                "artifacts_by_type": {},
                "expired_artifacts": 0,
            }

            current_time = datetime.utcnow()

            for metadata in self._metadata.values():
                # Count by type
                type_name = metadata.artifact_type.value
                if type_name not in stats["artifacts_by_type"]:
                    stats["artifacts_by_type"][type_name] = {
                        "count": 0,
                        "size_bytes": 0,
                    }

                stats["artifacts_by_type"][type_name]["count"] += 1
                stats["artifacts_by_type"][type_name]["size_bytes"] += (
                    metadata.size_bytes
                )

                # Count expired
                if metadata.expires_at and current_time > metadata.expires_at:
                    stats["expired_artifacts"] += 1

            return stats

        except Exception as e:
            raise ArtifactStoreError(f"Failed to get storage statistics: {str(e)}")

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        self._metadata: dict[str, ArtifactMetadata] = {}

        if self.metadata_file.exists():
            try:
                data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
                for artifact_id, metadata_dict in data.items():
                    self._metadata[artifact_id] = ArtifactMetadata.from_dict(
                        metadata_dict
                    )
            except Exception:
                # If metadata is corrupted, start fresh (but keep the files)
                self._metadata = {}

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            data = {
                artifact_id: metadata.to_dict()
                for artifact_id, metadata in self._metadata.items()
            }
            self.metadata_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            raise ArtifactStoreError(f"Failed to save metadata: {str(e)}")


# Convenience functions for common use cases


def store_coverage_report(
    store: ArtifactStoreAdapter,
    coverage_data: dict[str, Any],
    test_run_id: str | None = None,
) -> str:
    """Store a coverage report artifact."""
    return store.store_artifact(
        ArtifactType.COVERAGE_REPORT,
        coverage_data,
        artifact_id=f"coverage_{test_run_id}" if test_run_id else None,
        tags=["coverage", "report"],
        description=(
            f"Coverage report for test run {test_run_id}"
            if test_run_id
            else "Coverage report"
        ),
    )


def store_generated_test(
    store: ArtifactStoreAdapter,
    test_content: str,
    source_file: str,
    test_run_id: str | None = None,
) -> str:
    """Store a generated test artifact."""
    artifact_id = (
        f"test_{Path(source_file).stem}_{test_run_id}" if test_run_id else None
    )
    return store.store_artifact(
        ArtifactType.GENERATED_TEST,
        test_content,
        artifact_id=artifact_id,
        tags=["test", "generated", Path(source_file).stem],
        description=f"Generated test for {source_file}",
    )


def store_llm_response(
    store: ArtifactStoreAdapter,
    response_data: dict[str, Any],
    operation: str,
    test_run_id: str | None = None,
) -> str:
    """Store an LLM response artifact."""
    return store.store_artifact(
        ArtifactType.LLM_RESPONSE,
        response_data,
        tags=["llm", "response", operation],
        description=f"LLM response for {operation} operation",
        expiry_days=7,  # LLM responses expire after a week by default
    )
