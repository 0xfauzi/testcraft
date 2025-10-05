"""
Artifact storage adapter for managing test generation artifacts.

This module provides an adapter for storing, retrieving, and managing
artifacts from test generation runs including coverage reports, generated
tests, LLM responses, and run history with configurable cleanup policies.
"""

import hashlib
import json
import logging
import mimetypes
import os
import platform
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Platform-specific file locking
if platform.system() == "Windows":
    import msvcrt
else:
    import fcntl

# Constants
MAX_ARTIFACT_ID_LENGTH = 255
MAX_PATH_LENGTH = 4096
ARTIFACT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

# Logger setup
logger = logging.getLogger(__name__)


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
        """Create from dictionary with timezone-aware datetime handling."""
        try:
            # Validate required fields
            required_fields = [
                "artifact_id",
                "artifact_type",
                "timestamp",
                "file_path",
                "size_bytes",
                "tags",
            ]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Handle timezone-aware datetime parsing
            timestamp_str = data["timestamp"]
            if timestamp_str.endswith("Z"):
                # Handle ISO format with Z suffix (UTC)
                timestamp_str = timestamp_str.replace("Z", "+00:00")
            elif "+" not in timestamp_str and len(timestamp_str) > 19:
                # Handle ISO format without timezone info (assume UTC)
                timestamp_str = timestamp_str + "+00:00"

            timestamp = datetime.fromisoformat(timestamp_str)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)

            expires_at = None
            if data.get("expires_at"):
                expires_at_str = data["expires_at"]
                if expires_at_str.endswith("Z"):
                    expires_at_str = expires_at_str.replace("Z", "+00:00")
                elif "+" not in expires_at_str and len(expires_at_str) > 19:
                    expires_at_str = expires_at_str + "+00:00"

                expires_at = datetime.fromisoformat(expires_at_str)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)

            return cls(
                artifact_id=data["artifact_id"],
                artifact_type=ArtifactType(data["artifact_type"]),
                timestamp=timestamp,
                file_path=data["file_path"],
                size_bytes=data["size_bytes"],
                tags=data["tags"],
                description=data.get("description"),
                expires_at=expires_at,
            )
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid metadata format: {e}") from e


@dataclass
class CleanupPolicy:
    """Configuration for artifact cleanup policies."""

    max_age_days: int | None = None
    max_artifacts: int | None = None
    max_size_mb: int | None = None
    preserve_tags: list[str] | None = None

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
        # Validate configuration
        self._validate_configuration(base_path, cleanup_policy)

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.base_path / "metadata.json"
        self.metadata_backup_file = self.base_path / "metadata.json.bak"
        self.cleanup_policy = cleanup_policy or CleanupPolicy(
            max_age_days=30, max_artifacts=100
        )

        # Thread safety
        self._metadata_lock = threading.RLock()
        self._file_lock_fd: int | None = None

        # Create subdirectories for different artifact types
        for artifact_type in ArtifactType:
            (self.base_path / artifact_type.value).mkdir(exist_ok=True)

        # Lazy loading - don't load metadata immediately
        self._metadata: dict[str, ArtifactMetadata] | None = None
        self._metadata_loaded = False

    def _ensure_metadata_loaded(self) -> dict[str, ArtifactMetadata]:
        """Ensure metadata is loaded and return it."""
        if not self._metadata_loaded:
            self._load_metadata()
        assert self._metadata is not None, "Metadata should be loaded"
        return self._metadata

    def _validate_configuration(
        self, base_path: str | Path, cleanup_policy: CleanupPolicy | None
    ) -> None:
        """Validate initialization configuration."""
        if not base_path:
            raise ValueError("Base path cannot be empty")

        base_path_str = str(base_path)
        if len(base_path_str) > MAX_PATH_LENGTH:
            raise ValueError(
                f"Base path too long: {len(base_path_str)} > {MAX_PATH_LENGTH}"
            )

        if cleanup_policy:
            if (
                cleanup_policy.max_age_days is not None
                and cleanup_policy.max_age_days < 0
            ):
                raise ValueError("max_age_days must be non-negative")
            if (
                cleanup_policy.max_artifacts is not None
                and cleanup_policy.max_artifacts < 0
            ):
                raise ValueError("max_artifacts must be non-negative")
            if (
                cleanup_policy.max_size_mb is not None
                and cleanup_policy.max_size_mb < 0
            ):
                raise ValueError("max_size_mb must be non-negative")

    def _sanitize_artifact_id(self, artifact_id: str) -> str:
        """Sanitize artifact ID to prevent path traversal and ensure validity."""
        if not artifact_id:
            raise ValueError("Artifact ID cannot be empty")

        if len(artifact_id) > MAX_ARTIFACT_ID_LENGTH:
            raise ValueError(
                f"Artifact ID too long: {len(artifact_id)} > {MAX_ARTIFACT_ID_LENGTH}"
            )

        if not ARTIFACT_ID_PATTERN.match(artifact_id):
            raise ValueError(f"Invalid artifact ID format: {artifact_id}")

        return artifact_id

    def _get_file_extension(self, content: str | bytes | dict[str, Any]) -> str:
        """Determine appropriate file extension based on content type."""
        if isinstance(content, dict):
            return ".json"
        elif isinstance(content, bytes):
            # Try to detect MIME type for bytes
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                mime_type, _ = mimetypes.guess_type(temp_file_path)
                if mime_type:
                    # Map common MIME types to extensions
                    mime_to_ext = {
                        "text/plain": ".txt",
                        "application/json": ".json",
                        "text/xml": ".xml",
                        "text/csv": ".csv",
                    }
                    return mime_to_ext.get(mime_type, ".bin")
                return ".bin"
            finally:
                if temp_file and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            return ".txt"

    def _calculate_checksum(self, content: str | bytes | dict[str, Any]) -> str:
        """Calculate SHA-256 checksum for content."""
        content_bytes: bytes
        if isinstance(content, dict):
            content_bytes = json.dumps(content, sort_keys=True).encode("utf-8")
        elif isinstance(content, bytes):
            content_bytes = content
        else:
            content_bytes = str(content).encode("utf-8")

        return hashlib.sha256(content_bytes).hexdigest()

    def _acquire_file_lock(self) -> None:
        """Acquire file lock for thread safety."""
        if self._file_lock_fd is None:
            fd: int = os.open(self.metadata_file, os.O_RDWR | os.O_CREAT)
            self._file_lock_fd = fd

        if self._file_lock_fd is not None:
            try:
                if platform.system() == "Windows":
                    msvcrt.locking(self._file_lock_fd, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                else:
                    fcntl.flock(self._file_lock_fd, fcntl.LOCK_EX)
            except OSError as e:
                logger.warning(f"Failed to acquire file lock: {e}")

    def _release_file_lock(self) -> None:
        """Release file lock."""
        if self._file_lock_fd is not None:
            try:
                if platform.system() == "Windows":
                    msvcrt.locking(self._file_lock_fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
                else:
                    fcntl.flock(self._file_lock_fd, fcntl.LOCK_UN)
            except OSError as e:
                logger.warning(f"Failed to release file lock: {e}")

    def _load_metadata(self) -> None:
        """Load metadata from disk with thread safety."""
        with self._metadata_lock:
            if self._metadata_loaded:
                return

            self._acquire_file_lock()
            try:
                self._metadata = {}
                if self.metadata_file.exists():
                    try:
                        data = json.loads(
                            self.metadata_file.read_text(encoding="utf-8")
                        )
                        corrupted_entries = []

                        for artifact_id, metadata_dict in data.items():
                            try:
                                self._metadata[artifact_id] = (
                                    ArtifactMetadata.from_dict(
                                        {"artifact_id": artifact_id, **metadata_dict}
                                    )
                                )
                            except (ValueError, KeyError) as e:
                                logger.warning(
                                    f"Skipping corrupted metadata entry {artifact_id}: {e}"
                                )
                                corrupted_entries.append(artifact_id)

                        if corrupted_entries:
                            logger.info(
                                f"Found {len(corrupted_entries)} corrupted metadata entries"
                            )
                            # Save cleaned metadata
                            self._save_metadata()

                    except json.JSONDecodeError as e:
                        logger.error(f"Metadata file corrupted, creating backup: {e}")
                        self._create_metadata_backup()
                        self._metadata = {}
                    except Exception as e:
                        logger.error(f"Unexpected error loading metadata: {e}")
                        self._metadata = {}
                else:
                    logger.debug("No existing metadata file found")

                self._metadata_loaded = True
            finally:
                self._release_file_lock()

    def _save_metadata(self) -> None:
        """Save metadata to disk with atomic writes and thread safety."""
        if not self._metadata:
            return

        with self._metadata_lock:
            # Use atomic write (write to temp file, then rename)
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                dir=self.base_path,
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                try:
                    data = {
                        artifact_id: metadata.to_dict()
                        for artifact_id, metadata in self._metadata.items()
                    }
                    json.dump(data, temp_file, indent=2, ensure_ascii=False)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                    # Atomic rename
                    temp_path = Path(temp_file.name)
                    temp_path.replace(self.metadata_file)

                except Exception as e:
                    logger.error(f"Failed to save metadata: {e}")
                    raise ArtifactStoreError(
                        f"Failed to save metadata: {str(e)}"
                    ) from e
                finally:
                    # Clean up temp file if it still exists
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass

    def _create_metadata_backup(self) -> None:
        """Create a backup of corrupted metadata file."""
        try:
            if self.metadata_file.exists():
                self.metadata_file.replace(self.metadata_backup_file)
                logger.info(f"Created metadata backup: {self.metadata_backup_file}")
        except Exception as e:
            logger.error(f"Failed to create metadata backup: {e}")

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
            # Load metadata if not already loaded
            self._load_metadata()

            # Generate artifact ID if not provided
            if artifact_id is None:
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
                artifact_id = f"{artifact_type.value}_{timestamp}"
            else:
                # Sanitize provided artifact ID
                artifact_id = self._sanitize_artifact_id(artifact_id)

            # Ensure artifact ID doesn't already exist
            metadata_dict = self._ensure_metadata_loaded()
            if artifact_id in metadata_dict:
                raise ValueError(f"Artifact ID already exists: {artifact_id}")

            # Determine file extension and prepare content
            file_ext = self._get_file_extension(content)
            if isinstance(content, dict):
                file_content: str | bytes = json.dumps(
                    content, indent=2, ensure_ascii=False
                )
            elif isinstance(content, bytes):
                file_content = content
            else:
                file_content = str(content)

            # Create file path with validation
            file_name = f"{artifact_id}{file_ext}"
            file_path = self.base_path / artifact_type.value / file_name

            # Validate file path length
            if len(str(file_path)) > MAX_PATH_LENGTH:
                raise ValueError(
                    f"File path too long: {len(str(file_path))} > {MAX_PATH_LENGTH}"
                )

            # Write content to file with context manager
            try:
                if isinstance(file_content, bytes):
                    with open(file_path, "wb") as f:
                        f.write(file_content)
                else:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(file_content)
            except OSError as e:
                raise ArtifactStoreError(
                    f"Failed to write file {file_path}: {e}"
                ) from e

            # Create metadata with timezone-aware datetime
            file_size = file_path.stat().st_size
            expires_at = None
            if expiry_days:
                expires_at = datetime.now(UTC) + timedelta(days=expiry_days)

            metadata = ArtifactMetadata(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                timestamp=datetime.now(UTC),
                file_path=str(file_path.relative_to(self.base_path)),
                size_bytes=file_size,
                tags=tags or [],
                description=description,
                expires_at=expires_at,
            )

            # Store metadata with thread safety
            with self._metadata_lock:
                metadata_dict = self._ensure_metadata_loaded()
                metadata_dict[artifact_id] = metadata
                self._save_metadata()

            logger.debug(f"Stored artifact {artifact_id} ({file_size} bytes)")
            return artifact_id

        except (ValueError, OSError) as e:
            raise ArtifactStoreError(f"Failed to store artifact: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error storing artifact: {e}")
            raise ArtifactStoreError(f"Failed to store artifact: {str(e)}") from e

    def retrieve_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: ID of the artifact to retrieve

        Returns:
            Dictionary containing artifact data and metadata, or None if not found
        """
        try:
            # Load metadata if not already loaded
            metadata_dict = self._ensure_metadata_loaded()

            metadata = metadata_dict.get(artifact_id)
            if not metadata:
                return None

            # Check if artifact has expired using timezone-aware comparison
            current_time = datetime.now(UTC)
            if metadata.expires_at and current_time > metadata.expires_at:
                logger.debug(f"Artifact {artifact_id} has expired, removing")
                self.remove_artifact(artifact_id)
                return None

            file_path = self.base_path / metadata.file_path
            if not file_path.exists():
                logger.warning(
                    f"Artifact file not found: {file_path}, cleaning up metadata"
                )
                # Cleanup stale metadata with thread safety
                with self._metadata_lock:
                    metadata_dict = self._ensure_metadata_loaded()
                    if artifact_id in metadata_dict:
                        del metadata_dict[artifact_id]
                        self._save_metadata()
                return None

            # Read content based on file type with context managers
            try:
                if file_path.suffix == ".json":
                    with open(file_path, encoding="utf-8") as f:
                        content = json.load(f)
                elif file_path.suffix == ".bin":
                    with open(file_path, "rb") as f:
                        content = f.read()
                else:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read artifact file {file_path}: {e}")
                raise ArtifactStoreError(f"Failed to read artifact file: {e}") from e

            return {
                "artifact_id": artifact_id,
                "content": content,
                "metadata": metadata.to_dict(),
            }

        except ArtifactStoreError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving artifact {artifact_id}: {e}")
            raise ArtifactStoreError(
                f"Failed to retrieve artifact {artifact_id}: {str(e)}"
            ) from e

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
            # Load metadata if not already loaded
            metadata_dict = self._ensure_metadata_loaded()

            results = []
            current_time = datetime.now(UTC)

            for _artifact_id, metadata in metadata_dict.items():
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
            logger.error(f"Unexpected error listing artifacts: {e}")
            raise ArtifactStoreError(f"Failed to list artifacts: {str(e)}") from e

    def remove_artifact(self, artifact_id: str) -> bool:
        """
        Remove an artifact from the store.

        Args:
            artifact_id: ID of the artifact to remove

        Returns:
            True if artifact was removed, False if it didn't exist
        """
        try:
            # Load metadata if not already loaded
            metadata_dict = self._ensure_metadata_loaded()

            metadata = metadata_dict.get(artifact_id)
            if not metadata:
                return False

            # Remove file with context manager
            file_path = self.base_path / metadata.file_path
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed artifact file: {file_path}")
            except OSError as e:
                logger.error(f"Failed to remove artifact file {file_path}: {e}")
                raise ArtifactStoreError(f"Failed to remove artifact file: {e}") from e

            # Remove metadata with thread safety
            with self._metadata_lock:
                metadata_dict = self._ensure_metadata_loaded()
                if artifact_id in metadata_dict:
                    del metadata_dict[artifact_id]
                    self._save_metadata()

            return True

        except ArtifactStoreError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error removing artifact {artifact_id}: {e}")
            raise ArtifactStoreError(
                f"Failed to remove artifact {artifact_id}: {str(e)}"
            ) from e

    def cleanup_expired(self) -> dict[str, int]:
        """
        Remove expired artifacts based on cleanup policy.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            # Load metadata if not already loaded
            metadata_dict = self._ensure_metadata_loaded()

            current_time = datetime.now(UTC)
            removed_count = 0
            total_size_freed = 0

            # Find expired artifacts
            expired_ids = []
            with self._metadata_lock:
                metadata_dict = self._ensure_metadata_loaded()
                for artifact_id, metadata in metadata_dict.items():
                    if metadata.expires_at and current_time > metadata.expires_at:
                        expired_ids.append(artifact_id)

            # Remove expired artifacts
            for artifact_id in expired_ids:
                expired_metadata = metadata_dict.get(artifact_id)
                if expired_metadata and self.remove_artifact(artifact_id):
                    removed_count += 1
                    total_size_freed += expired_metadata.size_bytes

            logger.info(
                f"Cleaned up {removed_count} expired artifacts ({total_size_freed} bytes)"
            )
            return {
                "expired_artifacts_removed": removed_count,
                "size_freed_bytes": total_size_freed,
            }

        except Exception as e:
            logger.error(f"Unexpected error during expired artifact cleanup: {e}")
            raise ArtifactStoreError(
                f"Failed to cleanup expired artifacts: {str(e)}"
            ) from e

    def apply_cleanup_policy(self) -> dict[str, int]:
        """
        Apply the configured cleanup policy to remove old artifacts.

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            # Load metadata if not already loaded
            metadata_dict = self._ensure_metadata_loaded()

            stats = self.cleanup_expired()
            current_time = datetime.now(UTC)
            policy = self.cleanup_policy

            # Apply age-based cleanup
            if policy.max_age_days:
                cutoff_date = current_time - timedelta(days=policy.max_age_days)
                old_artifacts = []

                with self._metadata_lock:
                    metadata_dict = self._ensure_metadata_loaded()
                    for artifact_id, metadata in metadata_dict.items():
                        if metadata.timestamp < cutoff_date and not any(
                            tag in (policy.preserve_tags or []) for tag in metadata.tags
                        ):
                            old_artifacts.append((artifact_id, metadata))

                for artifact_id, metadata in old_artifacts:
                    if self.remove_artifact(artifact_id):
                        stats["expired_artifacts_removed"] += 1
                        stats["size_freed_bytes"] += metadata.size_bytes

            # Apply count-based cleanup
            metadata_dict = self._ensure_metadata_loaded()
            if policy.max_artifacts and len(metadata_dict) > policy.max_artifacts:
                # Sort by timestamp (oldest first)
                with self._metadata_lock:
                    metadata_dict = self._ensure_metadata_loaded()
                    artifacts_by_age = sorted(
                        metadata_dict.items(), key=lambda x: x[1].timestamp
                    )

                metadata_dict = self._ensure_metadata_loaded()
                excess_count = len(metadata_dict) - policy.max_artifacts
                for artifact_id, metadata in artifacts_by_age[:excess_count]:
                    if not any(
                        tag in (policy.preserve_tags or []) for tag in metadata.tags
                    ):
                        if self.remove_artifact(artifact_id):
                            stats["expired_artifacts_removed"] += 1
                            stats["size_freed_bytes"] += metadata.size_bytes

            # Apply size-based cleanup
            if policy.max_size_mb:
                max_size_bytes = policy.max_size_mb * 1024 * 1024
                with self._metadata_lock:
                    metadata_dict = self._ensure_metadata_loaded()
                    total_size = sum(
                        metadata.size_bytes for metadata in metadata_dict.values()
                    )

                if total_size > max_size_bytes:
                    # Remove oldest artifacts until under size limit
                    with self._metadata_lock:
                        metadata_dict = self._ensure_metadata_loaded()
                        artifacts_by_age = sorted(
                            metadata_dict.items(), key=lambda x: x[1].timestamp
                        )

                    current_total_size = total_size
                    for artifact_id, metadata in artifacts_by_age:
                        if current_total_size <= max_size_bytes:
                            break

                        if not any(
                            tag in (policy.preserve_tags or []) for tag in metadata.tags
                        ):
                            if self.remove_artifact(artifact_id):
                                stats["expired_artifacts_removed"] += 1
                                stats["size_freed_bytes"] += metadata.size_bytes
                                current_total_size -= metadata.size_bytes

            logger.info(
                f"Applied cleanup policy: removed {stats['expired_artifacts_removed']} artifacts, freed {stats['size_freed_bytes']} bytes"
            )
            return stats

        except Exception as e:
            logger.error(f"Unexpected error applying cleanup policy: {e}")
            raise ArtifactStoreError(f"Failed to apply cleanup policy: {str(e)}") from e

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get statistics about the artifact store.

        Returns:
            Dictionary with storage statistics
        """
        try:
            # Load metadata if not already loaded
            metadata_dict = self._ensure_metadata_loaded()

            stats: dict[str, Any] = {
                "total_artifacts": len(metadata_dict),
                "total_size_bytes": sum(
                    metadata.size_bytes for metadata in metadata_dict.values()
                ),
                "artifacts_by_type": {},
                "expired_artifacts": 0,
            }

            current_time = datetime.now(UTC)

            for metadata in metadata_dict.values():
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
            logger.error(f"Unexpected error getting storage statistics: {e}")
            raise ArtifactStoreError(
                f"Failed to get storage statistics: {str(e)}"
            ) from e


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
