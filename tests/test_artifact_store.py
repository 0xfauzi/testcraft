"""
Tests for artifact storage adapter.

This module tests the ArtifactStoreAdapter implementation,
verifying artifact storage, retrieval, and cleanup functionality.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from testcraft.adapters.io.artifact_store import (ArtifactStoreAdapter,
                                                  ArtifactType, CleanupPolicy,
                                                  store_coverage_report,
                                                  store_generated_test,
                                                  store_llm_response)


class TestArtifactStoreAdapter:
    """Test cases for ArtifactStoreAdapter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.store = ArtifactStoreAdapter(
            base_path=self.temp_dir,
            cleanup_policy=CleanupPolicy(max_age_days=7, max_artifacts=10),
        )

        # Sample test data
        self.sample_coverage_data = {
            "line_coverage": 0.85,
            "branch_coverage": 0.78,
            "files_analyzed": ["src/module1.py", "src/module2.py"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.sample_test_content = '''
import pytest

def test_example_function():
    """Test example function."""
    assert True

def test_another_function():
    """Test another function."""
    result = 2 + 2
    assert result == 4
'''

        self.sample_llm_response = {
            "model": "o4-mini",
            "prompt": "Generate tests for module1.py",
            "response": "Generated test content...",
            "tokens_used": 150,
            "cost": 0.0045,
        }

    def test_store_text_artifact(self) -> None:
        """Test storing text artifact."""
        content = "This is a test artifact"
        artifact_id = self.store.store_artifact(
            ArtifactType.GENERATED_TEST,
            content,
            tags=["test", "sample"],
            description="Test artifact",
        )

        assert artifact_id is not None
        assert artifact_id in self.store._metadata

        # Verify metadata
        metadata = self.store._metadata[artifact_id]
        assert metadata.artifact_type == ArtifactType.GENERATED_TEST
        assert metadata.tags == ["test", "sample"]
        assert metadata.description == "Test artifact"
        assert metadata.size_bytes > 0

    def test_store_json_artifact(self) -> None:
        """Test storing JSON artifact."""
        artifact_id = self.store.store_artifact(
            ArtifactType.COVERAGE_REPORT, self.sample_coverage_data, tags=["coverage"]
        )

        assert artifact_id is not None

        # Verify file was created with .json extension
        metadata = self.store._metadata[artifact_id]
        file_path = self.store.base_path / metadata.file_path
        assert file_path.exists()
        assert file_path.suffix == ".json"

        # Verify content is valid JSON
        with open(file_path) as f:
            loaded_data = json.load(f)
        assert loaded_data == self.sample_coverage_data

    def test_store_binary_artifact(self) -> None:
        """Test storing binary artifact."""
        binary_content = b"\x00\x01\x02\x03\xff"
        artifact_id = self.store.store_artifact(
            ArtifactType.LLM_RESPONSE, binary_content
        )

        assert artifact_id is not None

        # Verify file was created with .bin extension
        metadata = self.store._metadata[artifact_id]
        file_path = self.store.base_path / metadata.file_path
        assert file_path.exists()
        assert file_path.suffix == ".bin"

        # Verify binary content
        assert file_path.read_bytes() == binary_content

    def test_retrieve_artifact(self) -> None:
        """Test retrieving stored artifact."""
        # Store artifact first
        artifact_id = self.store.store_artifact(
            ArtifactType.COVERAGE_REPORT,
            self.sample_coverage_data,
            description="Test coverage report",
        )

        # Retrieve artifact
        retrieved = self.store.retrieve_artifact(artifact_id)

        assert retrieved is not None
        assert retrieved["artifact_id"] == artifact_id
        assert retrieved["content"] == self.sample_coverage_data
        assert retrieved["metadata"]["description"] == "Test coverage report"

    def test_retrieve_nonexistent_artifact(self) -> None:
        """Test retrieving non-existent artifact."""
        result = self.store.retrieve_artifact("nonexistent_id")
        assert result is None

    def test_list_artifacts_all(self) -> None:
        """Test listing all artifacts."""
        # Store multiple artifacts
        id1 = self.store.store_artifact(ArtifactType.COVERAGE_REPORT, {"test": 1})
        id2 = self.store.store_artifact(ArtifactType.GENERATED_TEST, "test content")
        id3 = self.store.store_artifact(ArtifactType.LLM_RESPONSE, {"llm": "data"})

        artifacts = self.store.list_artifacts()

        assert len(artifacts) == 3
        artifact_ids = [a["artifact_id"] for a in artifacts]
        assert id1 in artifact_ids
        assert id2 in artifact_ids
        assert id3 in artifact_ids

    def test_list_artifacts_by_type(self) -> None:
        """Test listing artifacts by type."""
        # Store artifacts of different types
        coverage_id = self.store.store_artifact(
            ArtifactType.COVERAGE_REPORT, {"test": 1}
        )
        test_id = self.store.store_artifact(ArtifactType.GENERATED_TEST, "test content")

        coverage_artifacts = self.store.list_artifacts(
            artifact_type=ArtifactType.COVERAGE_REPORT
        )
        test_artifacts = self.store.list_artifacts(
            artifact_type=ArtifactType.GENERATED_TEST
        )

        assert len(coverage_artifacts) == 1
        assert coverage_artifacts[0]["artifact_id"] == coverage_id

        assert len(test_artifacts) == 1
        assert test_artifacts[0]["artifact_id"] == test_id

    def test_list_artifacts_by_tags(self) -> None:
        """Test listing artifacts by tags."""
        # Store artifacts with different tags
        id1 = self.store.store_artifact(
            ArtifactType.COVERAGE_REPORT, {"test": 1}, tags=["tag1", "tag2"]
        )
        self.store.store_artifact(
            ArtifactType.GENERATED_TEST, "content", tags=["tag1", "tag3"]
        )
        self.store.store_artifact(
            ArtifactType.LLM_RESPONSE, {"llm": "data"}, tags=["tag2"]
        )

        # Filter by single tag
        tag1_artifacts = self.store.list_artifacts(tags=["tag1"])
        assert len(tag1_artifacts) == 2

        # Filter by multiple tags (must have all)
        tag12_artifacts = self.store.list_artifacts(tags=["tag1", "tag2"])
        assert len(tag12_artifacts) == 1
        assert tag12_artifacts[0]["artifact_id"] == id1

    def test_remove_artifact(self) -> None:
        """Test removing artifact."""
        # Store artifact first
        artifact_id = self.store.store_artifact(
            ArtifactType.GENERATED_TEST, "test content"
        )

        # Verify it exists
        assert artifact_id in self.store._metadata
        metadata = self.store._metadata[artifact_id]
        file_path = self.store.base_path / metadata.file_path
        assert file_path.exists()

        # Remove artifact
        result = self.store.remove_artifact(artifact_id)
        assert result

        # Verify it's gone
        assert artifact_id not in self.store._metadata
        assert not file_path.exists()

    def test_remove_nonexistent_artifact(self) -> None:
        """Test removing non-existent artifact."""
        result = self.store.remove_artifact("nonexistent_id")
        assert not result

    def test_store_with_expiry(self) -> None:
        """Test storing artifact with expiry."""
        artifact_id = self.store.store_artifact(
            ArtifactType.LLM_RESPONSE, self.sample_llm_response, expiry_days=1
        )

        metadata = self.store._metadata[artifact_id]
        assert metadata.expires_at is not None

        # Should expire approximately 1 day from now
        expected_expiry = datetime.utcnow() + timedelta(days=1)
        time_diff = abs((metadata.expires_at - expected_expiry).total_seconds())
        assert time_diff < 60  # Within 1 minute tolerance

    def test_retrieve_expired_artifact(self) -> None:
        """Test retrieving expired artifact."""
        # Store artifact with past expiry
        artifact_id = self.store.store_artifact(
            ArtifactType.LLM_RESPONSE, {"test": "data"}
        )

        # Manually set expiry to past
        metadata = self.store._metadata[artifact_id]
        metadata.expires_at = datetime.utcnow() - timedelta(hours=1)
        self.store._save_metadata()

        # Attempt to retrieve should return None and clean up
        result = self.store.retrieve_artifact(artifact_id)
        assert result is None
        assert artifact_id not in self.store._metadata

    def test_cleanup_expired(self) -> None:
        """Test cleanup of expired artifacts."""
        # Store some artifacts with different expiry times
        expired_id1 = self.store.store_artifact(
            ArtifactType.COVERAGE_REPORT, {"test": 1}
        )
        expired_id2 = self.store.store_artifact(ArtifactType.GENERATED_TEST, "test")
        valid_id = self.store.store_artifact(
            ArtifactType.LLM_RESPONSE, {"valid": "data"}
        )

        # Manually set some to expired
        self.store._metadata[expired_id1].expires_at = datetime.utcnow() - timedelta(
            hours=1
        )
        self.store._metadata[expired_id2].expires_at = datetime.utcnow() - timedelta(
            hours=2
        )
        # valid_id has no expiry
        self.store._save_metadata()

        # Run cleanup
        stats = self.store.cleanup_expired()

        assert stats["expired_artifacts_removed"] == 2
        assert stats["size_freed_bytes"] > 0

        # Verify only valid artifact remains
        assert expired_id1 not in self.store._metadata
        assert expired_id2 not in self.store._metadata
        assert valid_id in self.store._metadata

    def test_apply_cleanup_policy_age(self) -> None:
        """Test cleanup policy based on age."""
        # Create store with short max age
        store = ArtifactStoreAdapter(
            base_path=self.temp_dir / "age_test",
            cleanup_policy=CleanupPolicy(max_age_days=1),
        )

        # Store artifacts
        old_id = store.store_artifact(ArtifactType.COVERAGE_REPORT, {"old": "data"})
        new_id = store.store_artifact(ArtifactType.GENERATED_TEST, "new content")

        # Manually set one to be old
        store._metadata[old_id].timestamp = datetime.utcnow() - timedelta(days=2)
        store._save_metadata()

        # Apply cleanup policy
        stats = store.apply_cleanup_policy()

        assert stats["expired_artifacts_removed"] >= 1
        assert old_id not in store._metadata
        assert new_id in store._metadata

    def test_apply_cleanup_policy_count(self) -> None:
        """Test cleanup policy based on artifact count."""
        # Create store with low max count
        store = ArtifactStoreAdapter(
            base_path=self.temp_dir / "count_test",
            cleanup_policy=CleanupPolicy(max_artifacts=2),
        )

        # Store more artifacts than limit
        ids = []
        for i in range(4):
            artifact_id = store.store_artifact(
                ArtifactType.COVERAGE_REPORT, {f"test_{i}": i}
            )
            ids.append(artifact_id)

        # Apply cleanup policy
        stats = store.apply_cleanup_policy()

        assert len(store._metadata) <= 2
        assert stats["expired_artifacts_removed"] >= 2

    def test_apply_cleanup_policy_preserve_tags(self) -> None:
        """Test cleanup policy with preserved tags."""
        # Create store that preserves "important" tag
        store = ArtifactStoreAdapter(
            base_path=self.temp_dir / "preserve_test",
            cleanup_policy=CleanupPolicy(max_age_days=1, preserve_tags=["important"]),
        )

        # Store old artifacts, one with preserved tag
        old_normal_id = store.store_artifact(
            ArtifactType.COVERAGE_REPORT, {"normal": "data"}
        )
        old_important_id = store.store_artifact(
            ArtifactType.COVERAGE_REPORT, {"important": "data"}, tags=["important"]
        )

        # Make both old
        old_time = datetime.utcnow() - timedelta(days=2)
        store._metadata[old_normal_id].timestamp = old_time
        store._metadata[old_important_id].timestamp = old_time
        store._save_metadata()

        # Apply cleanup
        store.apply_cleanup_policy()

        # Important artifact should be preserved
        assert old_normal_id not in store._metadata
        assert old_important_id in store._metadata

    def test_get_storage_stats(self) -> None:
        """Test getting storage statistics."""
        # Store various artifacts
        self.store.store_artifact(
            ArtifactType.COVERAGE_REPORT, {"test": 1}, tags=["tag1"]
        )
        self.store.store_artifact(
            ArtifactType.GENERATED_TEST, "test content", tags=["tag2"]
        )
        expired_id = self.store.store_artifact(
            ArtifactType.LLM_RESPONSE, {"llm": "data"}
        )

        # Make one expired
        self.store._metadata[expired_id].expires_at = datetime.utcnow() - timedelta(
            hours=1
        )

        stats = self.store.get_storage_stats()

        assert stats["total_artifacts"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["expired_artifacts"] == 1

        # Check type breakdown
        assert "coverage_report" in stats["artifacts_by_type"]
        assert "generated_test" in stats["artifacts_by_type"]
        assert "llm_response" in stats["artifacts_by_type"]

    def test_convenience_functions(self) -> None:
        """Test convenience functions for common use cases."""
        # Test coverage report storage
        coverage_id = store_coverage_report(
            self.store, self.sample_coverage_data, "test_run_123"
        )

        retrieved = self.store.retrieve_artifact(coverage_id)
        assert retrieved["content"] == self.sample_coverage_data
        assert "coverage" in retrieved["metadata"]["tags"]

        # Test generated test storage
        test_id = store_generated_test(
            self.store, self.sample_test_content, "src/module1.py", "test_run_123"
        )

        retrieved = self.store.retrieve_artifact(test_id)
        assert retrieved["content"] == self.sample_test_content
        assert "test" in retrieved["metadata"]["tags"]
        assert "generated" in retrieved["metadata"]["tags"]

        # Test LLM response storage
        llm_id = store_llm_response(
            self.store, self.sample_llm_response, "test_generation", "test_run_123"
        )

        retrieved = self.store.retrieve_artifact(llm_id)
        assert retrieved["content"] == self.sample_llm_response
        assert "llm" in retrieved["metadata"]["tags"]
        assert "response" in retrieved["metadata"]["tags"]

    def test_metadata_persistence(self) -> None:
        """Test that metadata persists across adapter instances."""
        # Store artifact with first adapter
        artifact_id = self.store.store_artifact(
            ArtifactType.COVERAGE_REPORT,
            self.sample_coverage_data,
            description="Persistent test",
        )

        # Create new adapter instance pointing to same directory
        new_store = ArtifactStoreAdapter(base_path=self.temp_dir)

        # Should be able to retrieve artifact
        retrieved = new_store.retrieve_artifact(artifact_id)
        assert retrieved is not None
        assert retrieved["content"] == self.sample_coverage_data
        assert retrieved["metadata"]["description"] == "Persistent test"

    def test_corrupted_metadata_handling(self) -> None:
        """Test handling of corrupted metadata file."""
        # Store valid artifact first
        self.store.store_artifact(ArtifactType.COVERAGE_REPORT, {"test": 1})

        # Corrupt metadata file
        metadata_file = self.store.metadata_file
        metadata_file.write_text("invalid json content")

        # Create new adapter - should handle corruption gracefully
        new_store = ArtifactStoreAdapter(base_path=self.temp_dir)
        assert len(new_store._metadata) == 0  # Should start fresh

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)
