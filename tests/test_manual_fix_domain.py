"""Unit tests for manual-fix domain models."""

from datetime import datetime
from pathlib import Path

import pytest

from testcraft.domain.manual_fix import (
    ManualFixContextAttachment,
    ManualFixRecommendation,
    ManualFixRequest,
)


class TestManualFixRequest:
    """Test suite for ManualFixRequest domain model."""

    def test_create_request_with_minimal_fields(self):
        """Verify request creation with required fields only."""
        req = ManualFixRequest(
            project_root=Path("/repo"),
            target_file=Path("/repo/src/module.py"),
            target_object="func",
        )
        assert req.project_root == Path("/repo")
        assert req.target_file == Path("/repo/src/module.py")
        assert req.target_object == "func"
        assert req.trace_excerpt == ""
        assert req.notes == ""

    def test_create_request_with_all_fields(self):
        """Verify request creation with all fields populated."""
        req = ManualFixRequest(
            project_root=Path("/repo"),
            target_file=Path("/repo/src/module.py"),
            target_object="Class.method",
            trace_excerpt="AssertionError: expected 42",
            notes="Repro steps: call with x=10",
        )
        assert req.trace_excerpt == "AssertionError: expected 42"
        assert req.notes == "Repro steps: call with x=10"

    def test_request_is_immutable(self):
        """Verify request objects are frozen/immutable."""
        req = ManualFixRequest(
            project_root=Path("/repo"),
            target_file=Path("/repo/src/module.py"),
            target_object="func",
        )
        with pytest.raises(Exception):  # Pydantic ValidationError or AttributeError
            req.target_object = "other"


class TestManualFixRecommendation:
    """Test suite for ManualFixRecommendation domain model."""

    def test_hash_stability(self):
        """Verify hash computation is deterministic."""
        test_code = "def test_x(): assert False"
        bug_md = "# Bug\nSummary: fail"
        prompt_hash = "abc123"

        h1 = ManualFixRecommendation.compute_hash(test_code, bug_md, prompt_hash)
        h2 = ManualFixRecommendation.compute_hash(test_code, bug_md, prompt_hash)

        assert h1 == h2
        assert len(h1) == 16

    def test_hash_changes_with_content(self):
        """Verify hash changes when content changes."""
        base_hash = ManualFixRecommendation.compute_hash("code", "note", "prompt")
        changed_code = ManualFixRecommendation.compute_hash(
            "different", "note", "prompt"
        )
        changed_note = ManualFixRecommendation.compute_hash(
            "code", "different", "prompt"
        )
        changed_prompt = ManualFixRecommendation.compute_hash(
            "code", "note", "different"
        )

        assert base_hash != changed_code
        assert base_hash != changed_note
        assert base_hash != changed_prompt

    def test_create_recommendation(self):
        """Verify recommendation creation with all fields."""
        attachment = ManualFixContextAttachment(
            prompt_hash="abc123",
            model_id="gpt-4.1",
            params={"temperature": 0.1},
        )

        rec = ManualFixRecommendation(
            test_code="def test_x(): assert False",
            bug_note_markdown="# Bug\nDetails",
            attachment=attachment,
            recommendation_hash="fedcba9876543210",
        )

        assert rec.test_code == "def test_x(): assert False"
        assert rec.bug_note_markdown == "# Bug\nDetails"
        assert rec.attachment == attachment
        assert rec.recommendation_hash == "fedcba9876543210"

    def test_recommendation_is_immutable(self):
        """Verify recommendation objects are frozen/immutable."""
        rec = ManualFixRecommendation(
            test_code="def test_x(): pass",
            bug_note_markdown="# Note",
            recommendation_hash="abc123",
        )
        with pytest.raises(Exception):
            rec.test_code = "changed"


class TestManualFixContextAttachment:
    """Test suite for ManualFixContextAttachment model."""

    def test_create_attachment(self):
        """Verify attachment creation."""
        att = ManualFixContextAttachment(
            prompt_hash="xyz789",
            model_id="claude-3",
            params={"max_tokens": 8000},
        )
        assert att.prompt_hash == "xyz789"
        assert att.model_id == "claude-3"
        assert att.params == {"max_tokens": 8000}
        assert isinstance(att.created_at, datetime)
