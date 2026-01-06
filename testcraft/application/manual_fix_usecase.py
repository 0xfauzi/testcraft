"""
Manual Fix Guidance Use Case.

Coordinates building context, invoking the orchestrator manual-fix stage,
parsing the response into structured recommendation, presenting for acceptance,
and persisting Markdown artifacts with deduplication.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.manual_fix import (
    ManualFixContextAttachment,
    ManualFixRecommendation,
    ManualFixRequest,
)
from ..domain.models import ContextPack
from ..ports.parser_port import ParserPort
from ..ports.telemetry_port import SpanKind, TelemetryPort
from .generation.services.context_assembler import ContextAssembler
from .generation.services.context_pack import ContextPackBuilder
from .generation.services.llm_orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class ManualFixResult:
    """Result of manual-fix processing including artifact path and acceptance."""

    accepted: bool
    artifact_path: Path | None
    recommendation: ManualFixRecommendation | None


class ManualFixGuidanceUseCase:
    """Application use case to drive the Manual Fix Guidance pipeline."""

    def __init__(
        self,
        *,
        llm_orchestrator: LLMOrchestrator,
        parser_port: ParserPort,
        context_assembler: ContextAssembler,
        context_pack_builder: ContextPackBuilder,
        telemetry_port: TelemetryPort,
        presenter,  # ManualFixPresenterPort (duck-typed)
        config: dict[str, Any],
    ) -> None:
        self._orchestrator = llm_orchestrator
        self._parser = parser_port
        self._context_assembler = context_assembler
        self._context_pack_builder = context_pack_builder
        self._telemetry = telemetry_port
        self._presenter = presenter
        self._config = config or {}

    async def run(self, request: ManualFixRequest) -> ManualFixResult:
        """Execute the manual-fix workflow and optionally persist an artifact."""
        with self._telemetry.create_span(
            "manual_fix_guidance",
            kind=SpanKind.INTERNAL,
            attributes={
                "target_file": str(request.target_file),
                "target_object": request.target_object,
            },
        ) as span:
            # Build context pack for the exact target
            context_pack = self._context_pack_builder.build_context_pack(
                target_file=request.target_file,
                target_object=request.target_object,
                project_root=request.project_root,
            )
            if context_pack is None:
                span.set_attribute("error", "Failed to build context pack")
                return ManualFixResult(
                    accepted=False, artifact_path=None, recommendation=None
                )

            # Orchestrator stage (synchronous method returning dict)
            feedback = {"trace_excerpt": request.trace_excerpt, "notes": request.notes}

            with self._telemetry.create_child_span("manual_fix_llm_call") as llm_span:
                result = self._orchestrator.manual_fix_stage(
                    context_pack=context_pack,
                    feedback=feedback,
                    project_root=request.project_root,
                )
                if not result or not isinstance(result, dict):
                    llm_span.set_attribute("error", "Empty or invalid response")
                    return ManualFixResult(
                        accepted=False, artifact_path=None, recommendation=None
                    )

            # Parse the orchestrator response
            with self._telemetry.create_child_span("manual_fix_parse") as parse_span:
                raw_text = str(result.get("manual_fix_response", ""))
                test_code, bug_note_md = self._extract_fenced_blocks(raw_text)

                parse_span.set_attribute("test_code_extracted", bool(test_code))
                parse_span.set_attribute("bug_note_extracted", bool(bug_note_md))

                if not test_code or not bug_note_md:
                    # Best-effort: treat entire text as bug note to avoid losing guidance
                    test_code = test_code or ""
                    bug_note_md = bug_note_md or raw_text
                    logger.warning(
                        "Manual-fix response missing expected fenced blocks; using fallback"
                    )

            # Attachment metadata
            prompt_hash = self._compute_prompt_hash(context_pack)
            attachment = ManualFixContextAttachment(
                prompt_hash=prompt_hash,
                model_id=None,
                params={},
            )

            rec_hash = ManualFixRecommendation.compute_hash(
                test_code, bug_note_md, prompt_hash
            )
            recommendation = ManualFixRecommendation(
                test_code=test_code,
                bug_note_markdown=bug_note_md,
                attachment=attachment,
                recommendation_hash=rec_hash,
            )

            span.set_attribute("recommendation_hash", rec_hash)

            # Present and get acceptance (CLI-only in this version)
            auto_accept = bool(
                self._config.get("manual_fix", {}).get("auto_accept", False)
            )

            with self._telemetry.create_child_span(
                "manual_fix_present"
            ) as present_span:
                accepted = (
                    True
                    if auto_accept
                    else self._presenter.present(recommendation, dry_run=False)
                )
                present_span.set_attribute("auto_accept", auto_accept)
                present_span.set_attribute("accepted", accepted)

            artifact_path: Path | None = None
            if accepted:
                with self._telemetry.create_child_span(
                    "manual_fix_persist"
                ) as persist_span:
                    artifact_path = await self._persist_markdown_artifact(
                        recommendation, request, context_pack
                    )
                    persist_span.set_attribute(
                        "artifact_path", str(artifact_path) if artifact_path else "none"
                    )

                    # Record metrics
                    from datetime import datetime

                    from ..ports.telemetry_port import MetricValue

                    self._telemetry.record_metrics(
                        [
                            MetricValue(
                                name="manual_fix_accepted",
                                value=1,
                                unit="count",
                                labels={
                                    "target": request.target_object,
                                    "hash": rec_hash,
                                },
                                timestamp=datetime.now(),
                            ),
                            MetricValue(
                                name="manual_fix_files_touched",
                                value=1,
                                unit="count",
                                labels={"target_file": str(request.target_file)},
                                timestamp=datetime.now(),
                            ),
                        ]
                    )

            span.set_attribute("accepted", accepted)
            span.set_attribute("artifact_written", artifact_path is not None)

            return ManualFixResult(
                accepted=accepted,
                artifact_path=artifact_path,
                recommendation=recommendation,
            )

    # --------------------------- helpers ---------------------------
    def _extract_fenced_blocks(self, text: str) -> tuple[str | None, str | None]:
        """
        Extract first python code block and first markdown block.
        Returns (test_code, bug_note_md).
        """
        # Find all fenced code blocks
        all_fences = list(re.finditer(r"```([\w]*)\s*\n(.*?)\n```", text, re.DOTALL))

        test_code = None
        bug_md = None

        # First pass: look for explicit python and markdown blocks
        for fence in all_fences:
            lang = fence.group(1).strip().lower()
            content = fence.group(2).strip()

            if lang in ("python", "") and test_code is None:
                test_code = content
            elif lang in ("markdown", "md") and bug_md is None:
                bug_md = content

        # Second pass: if we didn't find both, use first fence as test and second as bug note
        if (test_code is None or bug_md is None) and len(all_fences) >= 2:
            if test_code is None:
                test_code = all_fences[0].group(2).strip()
            if bug_md is None:
                bug_md = all_fences[1].group(2).strip()

        return (test_code, bug_md)

    def _compute_prompt_hash(self, context_pack: ContextPack) -> str:
        """Compute a simple prompt hash from key context fields to aid dedupe."""
        import hashlib

        parts = [
            context_pack.import_map.target_import if context_pack.import_map else "",
            context_pack.focal.signature,
            context_pack.focal.docstring or "",
        ]
        h = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return h[:16]

    async def _persist_markdown_artifact(
        self,
        recommendation: ManualFixRecommendation,
        request: ManualFixRequest,
        context_pack: ContextPack,
    ) -> Path:
        """Persist recommendation as Markdown artifact with deduplication by hash."""
        out_dir = Path(
            self._config.get("manual_fix", {}).get(
                "output_dir", ".testcraft/manual_fixes"
            )
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build filename: <target_simple>_<hash>.md
        target_simple = (request.target_object or "target").replace(".", "_")
        file_name = f"{target_simple}_{recommendation.recommendation_hash}.md"
        path = out_dir / file_name

        if path.exists():
            return path

        # Compose markdown content with metadata header
        meta_lines = [
            "# Manual Fix Recommendation",
            f"- Target: {request.target_file}::{request.target_object}",
            f"- Hash: {recommendation.recommendation_hash}",
            f"- Created: {recommendation.created_at.isoformat()}Z",
        ]
        if context_pack.import_map:
            meta_lines.append(
                f"- Canonical Import: {context_pack.import_map.target_import}"
            )
        if recommendation.attachment:
            meta_lines.append(f"- Prompt Hash: {recommendation.attachment.prompt_hash}")
            if recommendation.attachment.model_id:
                meta_lines.append(f"- Model: {recommendation.attachment.model_id}")

        content = (
            "\n".join(meta_lines)
            + "\n\n## Failing Test (will pass after bug is fixed)\n\n"
            + "```python\n"
            + recommendation.test_code.strip()
            + "\n```\n\n## BUG NOTE\n\n"
            + recommendation.bug_note_markdown.strip()
            + "\n"
        )

        path.write_text(content, encoding="utf-8")
        return path
