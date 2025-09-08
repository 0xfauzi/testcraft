"""
PromptFoo/PromptLayer integration adapter for TestCraft evaluation system.

This module provides integration patterns following PromptFoo and PromptLayer
conventions for prompt evaluation, versioning, and result tracking.
Implements standardized configuration formats, evaluation workflows,
and result storage patterns used by modern prompt evaluation frameworks.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ...adapters.io.artifact_store import ArtifactStoreAdapter, ArtifactType
from ...ports.evaluation_port import (
    EvaluationConfig,
)


class PromptVersionStatus(Enum):
    """Status of a prompt version (PromptFoo pattern)."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class EvaluationStatus(Enum):
    """Status of an evaluation run (PromptLayer pattern)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PromptVersion:
    """Prompt version metadata (PromptFoo pattern)."""

    id: str
    name: str
    prompt: str
    version: str
    description: str | None = None
    tags: list[str] | None = None
    status: PromptVersionStatus = PromptVersionStatus.DRAFT
    created_at: datetime | None = None
    updated_at: datetime | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EvaluationRun:
    """Evaluation run metadata (PromptLayer pattern)."""

    id: str
    name: str
    description: str | None
    prompt_variants: list[PromptVersion]
    test_dataset: list[dict[str, Any]]
    config: EvaluationConfig
    status: EvaluationStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    results_summary: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PromptFooConfig:
    """PromptFoo-style configuration structure."""

    description: str
    prompts: list[str | dict[str, Any]]
    tests: str | list[dict[str, Any]]
    providers: list[str] | None = None
    evaluators: list[dict[str, Any]] | None = None
    defaultTest: dict[str, Any] | None = None
    env: dict[str, str] | None = None
    sharing: bool | None = None
    outputPath: str | None = None

    def to_testcraft_format(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], EvaluationConfig]:
        """Convert PromptFoo config to TestCraft format."""
        # Convert prompts
        prompt_variants = []
        for i, prompt in enumerate(self.prompts):
            if isinstance(prompt, str):
                variant = {
                    "id": f"prompt_{i}",
                    "name": f"Prompt {i + 1}",
                    "prompt": prompt,
                    "version": "1.0.0",
                }
            else:
                variant = {
                    "id": prompt.get("id", f"prompt_{i}"),
                    "name": prompt.get("name", f"Prompt {i + 1}"),
                    "prompt": prompt.get("prompt", prompt.get("content", "")),
                    "version": prompt.get("version", "1.0.0"),
                    "description": prompt.get("description"),
                    "tags": prompt.get("tags", []),
                }
            prompt_variants.append(variant)

        # Convert test dataset
        if isinstance(self.tests, str):
            # Assume it's a file path - would need to be loaded
            test_dataset = []
        else:
            test_dataset = []
            for test in self.tests:
                dataset_item = {
                    "name": test.get("vars", {}).get("name", "unnamed_test"),
                    "source_file": test.get("vars", {}).get(
                        "source_file", "unknown.py"
                    ),
                    "source_content": test.get("vars", {}).get("source_code", ""),
                    "expected_output": (
                        test.get("assert", {}).get("contains")
                        if test.get("assert")
                        else None
                    ),
                }
                test_dataset.append(dataset_item)

        # Convert evaluation config
        rubric_dimensions = ["correctness", "coverage", "clarity", "safety"]
        if self.evaluators:
            for evaluator in self.evaluators:
                if evaluator.get("type") == "llm-rubric":
                    rubric_dimensions = evaluator.get("rubric", {}).keys()
                    break

        eval_config = EvaluationConfig(
            acceptance_checks=True,
            llm_judge_enabled=True,
            rubric_dimensions=list(rubric_dimensions),
            statistical_testing=True,
            human_review_enabled=False,
        )

        return prompt_variants, test_dataset, eval_config


class PromptFooIntegrationAdapter:
    """
    Integration adapter for PromptFoo/PromptLayer evaluation patterns.

    This adapter provides compatibility with PromptFoo configuration formats,
    PromptLayer result tracking patterns, and standardized evaluation workflows
    used by modern prompt evaluation frameworks.
    """

    def __init__(
        self,
        artifact_store: ArtifactStoreAdapter,
        project_root: Path,
        promptfoo_config_path: Path | None = None,
    ):
        """Initialize integration adapter."""
        self.artifact_store = artifact_store
        self.project_root = project_root
        self.promptfoo_config_path = promptfoo_config_path

        # Initialize storage paths
        self.prompt_registry_path = project_root / ".testcraft" / "prompts"
        self.evaluation_runs_path = project_root / ".testcraft" / "evaluation_runs"
        self.results_path = project_root / ".testcraft" / "results"

        # Ensure directories exist
        for path in [
            self.prompt_registry_path,
            self.evaluation_runs_path,
            self.results_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Load existing prompt registry
        self._prompt_registry: dict[str, PromptVersion] = self._load_prompt_registry()

        # Track active evaluation runs
        self._active_runs: dict[str, EvaluationRun] = {}

    def load_promptfoo_config(self, config_path: Path) -> PromptFooConfig:
        """Load PromptFoo configuration file."""
        try:
            with open(config_path) as f:
                if config_path.suffix == ".json":
                    config_data = json.load(f)
                elif config_path.suffix in [".yml", ".yaml"]:
                    import yaml

                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            return PromptFooConfig(**config_data)

        except Exception as e:
            raise ValueError(f"Failed to load PromptFoo config: {e}")

    def register_prompt_version(
        self,
        prompt: str,
        name: str,
        version: str,
        description: str | None = None,
        tags: list[str] | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new prompt version (PromptFoo pattern)."""
        # Generate deterministic ID based on content
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        prompt_id = f"{name.lower().replace(' ', '_')}_{version}_{prompt_hash}"

        # Create prompt version
        prompt_version = PromptVersion(
            id=prompt_id,
            name=name,
            prompt=prompt,
            version=version,
            description=description,
            tags=tags or [],
            status=PromptVersionStatus.DRAFT,
            parent_id=parent_id,
            metadata=metadata or {},
        )

        # Store in registry
        self._prompt_registry[prompt_id] = prompt_version
        self._save_prompt_registry()

        return prompt_id

    def get_prompt_version(self, prompt_id: str) -> PromptVersion | None:
        """Get prompt version by ID."""
        return self._prompt_registry.get(prompt_id)

    def list_prompt_versions(
        self,
        name_filter: str | None = None,
        status_filter: PromptVersionStatus | None = None,
        tags_filter: list[str] | None = None,
    ) -> list[PromptVersion]:
        """List prompt versions with optional filtering."""
        versions = list(self._prompt_registry.values())

        if name_filter:
            versions = [v for v in versions if name_filter.lower() in v.name.lower()]

        if status_filter:
            versions = [v for v in versions if v.status == status_filter]

        if tags_filter:
            versions = [
                v for v in versions if any(tag in v.tags for tag in tags_filter)
            ]

        return sorted(versions, key=lambda x: x.created_at, reverse=True)

    def activate_prompt_version(self, prompt_id: str) -> None:
        """Activate a prompt version (PromptFoo pattern)."""
        if prompt_id not in self._prompt_registry:
            raise ValueError(f"Prompt version not found: {prompt_id}")

        prompt_version = self._prompt_registry[prompt_id]

        # Deactivate other versions of the same prompt
        for _other_id, other_version in self._prompt_registry.items():
            if (
                other_version.name == prompt_version.name
                and other_version.status == PromptVersionStatus.ACTIVE
            ):
                other_version.status = PromptVersionStatus.DEPRECATED
                other_version.updated_at = datetime.now(UTC)

        # Activate this version
        prompt_version.status = PromptVersionStatus.ACTIVE
        prompt_version.updated_at = datetime.now(UTC)

        self._save_prompt_registry()

    def create_evaluation_run(
        self,
        name: str,
        description: str,
        prompt_variants: list[dict[str, Any]],
        test_dataset: list[dict[str, Any]],
        config: EvaluationConfig,
    ) -> str:
        """Create new evaluation run (PromptLayer pattern)."""
        run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(name.encode()).hexdigest()[:8]}"

        # Convert prompt variants to PromptVersion objects
        prompt_versions = []
        for variant in prompt_variants:
            prompt_version = PromptVersion(
                id=variant.get("id", f"variant_{len(prompt_versions)}"),
                name=variant.get("name", f"Variant {len(prompt_versions) + 1}"),
                prompt=variant["prompt"],
                version=variant.get("version", "1.0.0"),
                description=variant.get("description"),
                tags=variant.get("tags", []),
                status=PromptVersionStatus.ACTIVE,
            )
            prompt_versions.append(prompt_version)

        # Create evaluation run
        eval_run = EvaluationRun(
            id=run_id,
            name=name,
            description=description,
            prompt_variants=prompt_versions,
            test_dataset=test_dataset,
            config=config,
            status=EvaluationStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        # Store run metadata
        self._active_runs[run_id] = eval_run
        self._save_evaluation_run(eval_run)

        return run_id

    def start_evaluation_run(self, run_id: str) -> None:
        """Mark evaluation run as started."""
        if run_id not in self._active_runs:
            raise ValueError(f"Evaluation run not found: {run_id}")

        eval_run = self._active_runs[run_id]
        eval_run.status = EvaluationStatus.RUNNING
        eval_run.started_at = datetime.now(UTC)

        self._save_evaluation_run(eval_run)

    def complete_evaluation_run(
        self,
        run_id: str,
        results_summary: dict[str, Any],
        error_message: str | None = None,
    ) -> None:
        """Mark evaluation run as completed."""
        if run_id not in self._active_runs:
            raise ValueError(f"Evaluation run not found: {run_id}")

        eval_run = self._active_runs[run_id]
        eval_run.status = (
            EvaluationStatus.COMPLETED if not error_message else EvaluationStatus.FAILED
        )
        eval_run.completed_at = datetime.now(UTC)
        eval_run.results_summary = results_summary
        eval_run.error_message = error_message

        self._save_evaluation_run(eval_run)

        # Store detailed results in artifact store
        if results_summary:
            self.artifact_store.store_artifact(
                ArtifactType.EVALUATION_RESULT,
                results_summary,
                tags=["evaluation_run", "completed", run_id],
                description=f"Results for evaluation run: {eval_run.name}",
            )

    def get_evaluation_run(self, run_id: str) -> EvaluationRun | None:
        """Get evaluation run by ID."""
        if run_id in self._active_runs:
            return self._active_runs[run_id]

        # Try loading from disk
        run_file = self.evaluation_runs_path / f"{run_id}.json"
        if run_file.exists():
            with open(run_file) as f:
                run_data = json.load(f)
                return self._dict_to_evaluation_run(run_data)

        return None

    def list_evaluation_runs(
        self, status_filter: EvaluationStatus | None = None, limit: int = 50
    ) -> list[EvaluationRun]:
        """List evaluation runs with optional filtering."""
        runs = []

        # Add active runs
        runs.extend(self._active_runs.values())

        # Load historical runs from disk
        for run_file in self.evaluation_runs_path.glob("*.json"):
            if run_file.stem not in self._active_runs:
                try:
                    with open(run_file) as f:
                        run_data = json.load(f)
                        run = self._dict_to_evaluation_run(run_data)
                        runs.append(run)
                except Exception:
                    continue  # Skip corrupted files

        # Apply filtering
        if status_filter:
            runs = [r for r in runs if r.status == status_filter]

        # Sort by creation time (newest first) and limit
        runs.sort(key=lambda x: x.created_at, reverse=True)
        return runs[:limit]

    def export_promptfoo_results(self, run_id: str, output_path: Path) -> None:
        """Export results in PromptFoo format."""
        eval_run = self.get_evaluation_run(run_id)
        if not eval_run or not eval_run.results_summary:
            raise ValueError(f"No results found for evaluation run: {run_id}")

        # Convert to PromptFoo results format
        promptfoo_results = {
            "version": 2,
            "timestamp": (
                eval_run.completed_at.isoformat() if eval_run.completed_at else None
            ),
            "results": {
                "table": {
                    "head": {
                        "prompts": [
                            {"id": v.id, "label": v.name}
                            for v in eval_run.prompt_variants
                        ],
                        "vars": ["source_code", "expected_output"],
                    },
                    "body": [],
                },
                "stats": eval_run.results_summary.get("statistics", {}),
                "config": {
                    "description": eval_run.description,
                    "prompts": [
                        {"id": v.id, "content": v.prompt}
                        for v in eval_run.prompt_variants
                    ],
                    "tests": [
                        {
                            "vars": test_case,
                            "assert": [
                                {"type": "llm-rubric", "value": "evaluate test quality"}
                            ],
                        }
                        for test_case in eval_run.test_dataset
                    ],
                },
            },
        }

        # Add evaluation results to table body
        variant_results = eval_run.results_summary.get("variant_evaluations", [])
        for _i, test_case in enumerate(eval_run.test_dataset):
            row = {"vars": test_case, "outputs": []}

            for variant_result in variant_results:
                if variant_result["variant_id"] in [
                    v.id for v in eval_run.prompt_variants
                ]:
                    # Find corresponding evaluation for this test case
                    individual_evals = variant_result.get("individual_evaluations", [])
                    eval_for_case = next(
                        (
                            e
                            for e in individual_evals
                            if e.get("test_case") == test_case.get("name")
                        ),
                        None,
                    )

                    output = {
                        "pass": (
                            eval_for_case.get("acceptance_result", {}).get(
                                "passed", False
                            )
                            if eval_for_case
                            else False
                        ),
                        "score": (
                            eval_for_case.get("llm_judge_result", {}).get(
                                "overall_score", 0
                            )
                            if eval_for_case
                            else 0
                        ),
                        "namedScores": (
                            eval_for_case.get("llm_judge_result", {}).get("scores", {})
                            if eval_for_case
                            else {}
                        ),
                        "text": (
                            eval_for_case.get("generated_test", "")
                            if eval_for_case
                            else ""
                        ),
                    }
                    row["outputs"].append(output)

            promptfoo_results["results"]["table"]["body"].append(row)

        # Save to file
        with open(output_path, "w") as f:
            json.dump(promptfoo_results, f, indent=2, default=str)

    def import_promptfoo_config_and_run(
        self, config_path: Path, run_name: str | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], EvaluationConfig]:
        """Import PromptFoo config and prepare for TestCraft evaluation."""
        promptfoo_config = self.load_promptfoo_config(config_path)

        # Convert to TestCraft format
        prompt_variants, test_dataset, eval_config = (
            promptfoo_config.to_testcraft_format()
        )

        # Register prompt variants
        for variant in prompt_variants:
            self.register_prompt_version(
                prompt=variant["prompt"],
                name=variant["name"],
                version=variant["version"],
                description=variant.get("description"),
                tags=variant.get("tags", []) + ["imported", "promptfoo"],
            )

        return prompt_variants, test_dataset, eval_config

    def _load_prompt_registry(self) -> dict[str, PromptVersion]:
        """Load prompt registry from disk."""
        registry_file = self.prompt_registry_path / "registry.json"
        if not registry_file.exists():
            return {}

        try:
            with open(registry_file) as f:
                registry_data = json.load(f)

            registry = {}
            for prompt_id, prompt_data in registry_data.items():
                prompt_version = PromptVersion(
                    **{
                        **prompt_data,
                        "status": PromptVersionStatus(prompt_data["status"]),
                        "created_at": (
                            datetime.fromisoformat(prompt_data["created_at"])
                            if prompt_data.get("created_at")
                            else None
                        ),
                        "updated_at": (
                            datetime.fromisoformat(prompt_data["updated_at"])
                            if prompt_data.get("updated_at")
                            else None
                        ),
                    }
                )
                registry[prompt_id] = prompt_version

            return registry

        except Exception:
            return {}  # Start fresh if corrupted

    def _save_prompt_registry(self) -> None:
        """Save prompt registry to disk."""
        registry_file = self.prompt_registry_path / "registry.json"

        registry_data = {}
        for prompt_id, prompt_version in self._prompt_registry.items():
            data = asdict(prompt_version)
            data["status"] = prompt_version.status.value
            data["created_at"] = (
                prompt_version.created_at.isoformat()
                if prompt_version.created_at
                else None
            )
            data["updated_at"] = (
                prompt_version.updated_at.isoformat()
                if prompt_version.updated_at
                else None
            )
            registry_data[prompt_id] = data

        with open(registry_file, "w") as f:
            json.dump(registry_data, f, indent=2, default=str)

    def _save_evaluation_run(self, eval_run: EvaluationRun) -> None:
        """Save evaluation run to disk."""
        run_file = self.evaluation_runs_path / f"{eval_run.id}.json"

        run_data = asdict(eval_run)
        run_data["status"] = eval_run.status.value
        run_data["created_at"] = eval_run.created_at.isoformat()
        run_data["started_at"] = (
            eval_run.started_at.isoformat() if eval_run.started_at else None
        )
        run_data["completed_at"] = (
            eval_run.completed_at.isoformat() if eval_run.completed_at else None
        )

        # Convert PromptVersion objects to dicts
        run_data["prompt_variants"] = [
            {
                **asdict(pv),
                "status": pv.status.value,
                "created_at": pv.created_at.isoformat() if pv.created_at else None,
                "updated_at": pv.updated_at.isoformat() if pv.updated_at else None,
            }
            for pv in eval_run.prompt_variants
        ]

        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2, default=str)

    def _dict_to_evaluation_run(self, run_data: dict[str, Any]) -> EvaluationRun:
        """Convert dictionary to EvaluationRun object."""
        # Convert prompt variants
        prompt_variants = []
        for pv_data in run_data.get("prompt_variants", []):
            pv = PromptVersion(
                **{
                    **pv_data,
                    "status": PromptVersionStatus(pv_data["status"]),
                    "created_at": (
                        datetime.fromisoformat(pv_data["created_at"])
                        if pv_data.get("created_at")
                        else None
                    ),
                    "updated_at": (
                        datetime.fromisoformat(pv_data["updated_at"])
                        if pv_data.get("updated_at")
                        else None
                    ),
                }
            )
            prompt_variants.append(pv)

        # Create EvaluationRun
        return EvaluationRun(
            id=run_data["id"],
            name=run_data["name"],
            description=run_data["description"],
            prompt_variants=prompt_variants,
            test_dataset=run_data["test_dataset"],
            config=EvaluationConfig(**run_data["config"]),
            status=EvaluationStatus(run_data["status"]),
            created_at=datetime.fromisoformat(run_data["created_at"]),
            started_at=(
                datetime.fromisoformat(run_data["started_at"])
                if run_data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(run_data["completed_at"])
                if run_data.get("completed_at")
                else None
            ),
            error_message=run_data.get("error_message"),
            metadata=run_data.get("metadata", {}),
            results_summary=run_data.get("results_summary"),
        )
