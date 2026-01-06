from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from testcraft.application.generation.services.import_resolver import ImportResolver
from testcraft.application.generation.services.llm_orchestrator import (
    LLMOrchestrator,
)
from testcraft.config.models import OrchestratorConfig
from testcraft.domain.models import (
    ContextPack,
    DependencyMetadata,
    Focal,
    ImportMap,
    Target,
)

from .services.import_resolver.namespace_fixtures import create_namespace_project


class _StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def generate_tests(self, **_: str) -> str:
        return self._response


class _SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    def generate_tests(self, **_: str) -> str:
        if self._index >= len(self._responses):
            return self._responses[-1]

        response = self._responses[self._index]
        self._index += 1
        return response


class _StubSymbolResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path | None]] = []

    def resolve_symbols(
        self, missing_symbols: list[str], project_root: Path | None
    ) -> list[SimpleNamespace]:
        self.calls.append((missing_symbols, project_root))
        return []

    def update_environment_modules(
        self,
        *,
        external_modules: set[str] | None = None,
        stdlib_modules: set[str] | None = None,
    ) -> None:  # pragma: no cover - simple stub
        return None


class _StubPromptRegistry:
    version = "test"

    def get_system_prompt(self, _: str) -> str:  # pragma: no cover - trivial stub
        return ""

    def get_user_prompt(
        self,
        _: str,
        *,
        additional_context: dict | None = None,
        version: str | None = None,
    ) -> str:  # pragma: no cover - trivial stub
        return ""


def _build_namespace_context_pack(project: dict[str, Path]) -> ContextPack:
    resolver = ImportResolver()
    import_map_data = resolver.resolve(project["test_file"])

    project_root = project["project_root"]
    test_file = project["test_file"]
    relative_path = test_file.relative_to(project_root).as_posix()
    module_path = ".".join(Path(relative_path).with_suffix("").parts)

    return ContextPack(
        target=Target(
            module_file=str(test_file),
            object="module",
            module_path=module_path,
            relative_path=relative_path,
            exists_in_source=True,
        ),
        import_map=ImportMap(
            target_import=import_map_data["target_import"],
            sys_path_roots=list(import_map_data["sys_path_roots"]),
            needs_bootstrap=import_map_data["needs_bootstrap"],
            bootstrap_conftest=import_map_data["bootstrap_conftest"],
        ),
        focal=Focal(
            source="def sentinel():\n    pass\n", signature="def sentinel() -> None"
        ),
        dependency_metadata=DependencyMetadata(),
    )


@pytest.mark.parametrize(
    "project_root_relative", ["tests/src/weather_collector/test_config.py"]
)
def test_plan_stage_records_unresolved_symbols(
    tmp_path: Path, project_root_relative: str
) -> None:
    unresolved_symbol = "tests.src.weather_collector.test_config.SomeFixture"
    stub_llm = _StubLLM(
        response=f'{{"plan": [], "missing_symbols": ["{unresolved_symbol}"]}}'
    )
    stub_symbol_resolver = _StubSymbolResolver()
    prompt_registry = _StubPromptRegistry()

    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=stub_symbol_resolver,
        prompt_registry=prompt_registry,
        config=OrchestratorConfig(max_plan_retries=1, max_refine_retries=1),
        max_plan_retries=1,
    )

    project_root = tmp_path / "weather_project"
    project_root.mkdir()

    target_path = project_root / project_root_relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("def test_placeholder():\n    pass\n", encoding="utf-8")

    context_pack = ContextPack(
        target=Target(
            module_file=str(target_path),
            object="module",
            module_path="tests.src.weather_collector.test_config",
            relative_path=project_root_relative,
            exists_in_source=True,
        ),
        import_map=ImportMap(
            target_import="import tests.src.weather_collector.test_config as _under_test",
            sys_path_roots=[str((project_root / "tests" / "src").resolve())],
            needs_bootstrap=False,
            bootstrap_conftest="",
        ),
        focal=Focal(
            source="def sentinel():\n    pass\n", signature="def sentinel() -> None"
        ),
        dependency_metadata=DependencyMetadata(),
    )

    plan = orchestrator._plan_stage(context_pack, project_root=project_root)

    assert plan["missing_symbols"] == []
    metadata = plan.get("metadata", {}).get("symbol_resolution", {})
    assert metadata.get("unresolved_symbols") == [unresolved_symbol]

    assert stub_symbol_resolver.calls  # ensure resolver was invoked


def test_plan_stage_rejects_error_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    project = create_namespace_project(tmp_path)
    context_pack = _build_namespace_context_pack(project)

    stub_llm = _StubLLM('{"error": "model overloaded"}')
    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=_StubSymbolResolver(),
        prompt_registry=_StubPromptRegistry(),
        config=OrchestratorConfig(max_plan_retries=1, max_refine_retries=1),
        max_plan_retries=1,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError) as exc_info:
            orchestrator.plan_stage(context_pack, project_root=project["project_root"])

    assert "PLAN stage refused to proceed" in str(exc_info.value)
    assert "model overloaded" in caplog.text


def test_plan_stage_raises_on_malformed_json(tmp_path: Path) -> None:
    stub_llm = _StubLLM("{")
    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=_StubSymbolResolver(),
        prompt_registry=_StubPromptRegistry(),
        config=OrchestratorConfig(max_plan_retries=1, max_refine_retries=1),
        max_plan_retries=1,
    )

    project_root = tmp_path / "proj"
    project_root.mkdir()
    target_path = project_root / "tests" / "src" / "pkg" / "test_sample.py"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("def test_placeholder():\n    pass\n", encoding="utf-8")

    context_pack = ContextPack(
        target=Target(
            module_file=str(target_path),
            object="module",
            module_path="tests.src.pkg.test_sample",
            relative_path="tests/src/pkg/test_sample.py",
            exists_in_source=True,
        ),
        import_map=ImportMap(
            target_import="import tests.src.pkg.test_sample as _under_test",
            sys_path_roots=[str((project_root / "tests" / "src").resolve())],
            needs_bootstrap=False,
            bootstrap_conftest="",
        ),
        focal=Focal(
            source="def sentinel():\n    pass\n", signature="def sentinel() -> None"
        ),
        dependency_metadata=DependencyMetadata(),
    )

    with pytest.raises(ValueError) as exc_info:
        orchestrator.plan_and_generate(context_pack, project_root=project_root)

    assert "malformed JSON" in str(exc_info.value)


def test_plan_stage_accumulates_unresolved_symbols_across_attempts(
    tmp_path: Path,
) -> None:
    project = create_namespace_project(tmp_path)

    responses = [
        '{"plan": [], "missing_symbols": ["tests.src.weather_collector.fixture_a"]}',
        '{"plan": [], "missing_symbols": ["tests.src.weather_collector.fixture_b"]}',
    ]
    stub_llm = _SequenceLLM(responses)
    stub_symbol_resolver = _StubSymbolResolver()

    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=stub_symbol_resolver,
        prompt_registry=_StubPromptRegistry(),
        config=OrchestratorConfig(max_plan_retries=2, max_refine_retries=1),
        max_plan_retries=2,
    )

    project_root = project["project_root"]
    context_pack = _build_namespace_context_pack(project)

    plan = orchestrator._plan_stage(context_pack, project_root=project_root)

    assert plan["missing_symbols"] == []
    metadata = plan.get("metadata", {}).get("symbol_resolution", {})
    assert metadata.get("unresolved_symbols") == [
        "tests.src.weather_collector.fixture_a",
        "tests.src.weather_collector.fixture_b",
    ]
    assert len(stub_symbol_resolver.calls) == 2


def test_plan_stage_rejects_invalid_schema(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    project = create_namespace_project(tmp_path)
    context_pack = _build_namespace_context_pack(project)

    stub_llm = _StubLLM('{"plan": "not-a-list"}')
    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=_StubSymbolResolver(),
        prompt_registry=_StubPromptRegistry(),
        config=OrchestratorConfig(max_plan_retries=1, max_refine_retries=1),
        max_plan_retries=1,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError) as exc_info:
            orchestrator.plan_stage(context_pack, project_root=project["project_root"])

    assert "PLAN response 'plan' field must be a list" in caplog.text
    assert "PLAN stage failed" in str(exc_info.value)


def test_refine_stage_rejects_invalid_schema(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    project = create_namespace_project(tmp_path)
    context_pack = _build_namespace_context_pack(project)

    stub_llm = _StubLLM('{"refined_code": {"oops": "dict"}}')
    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=_StubSymbolResolver(),
        prompt_registry=_StubPromptRegistry(),
        config=OrchestratorConfig(max_plan_retries=1, max_refine_retries=1),
        max_refine_retries=1,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError) as exc_info:
            orchestrator.refine_stage(
                context_pack,
                existing_code="def test_existing():\n    pass\n",
                feedback={},
                project_root=project["project_root"],
            )

    assert "REFINE response 'refined_code' must be a string" in caplog.text
    assert "REFINE stage failed" in str(exc_info.value)


def test_refine_stage_rejects_error_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    project = create_namespace_project(tmp_path)
    context_pack = _build_namespace_context_pack(project)

    stub_llm = _StubLLM('{"error": "refine disabled"}')
    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=_StubSymbolResolver(),
        prompt_registry=_StubPromptRegistry(),
        config=OrchestratorConfig(max_plan_retries=1, max_refine_retries=1),
        max_refine_retries=1,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError) as exc_info:
            orchestrator.refine_stage(
                context_pack,
                existing_code="def test_original():\n    assert True\n",
                feedback={},
                project_root=project["project_root"],
            )

    assert "REFINE stage refused to proceed" in str(exc_info.value)
    assert "refine disabled" in caplog.text


def test_refine_stage_rejects_missing_code_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    project = create_namespace_project(tmp_path)
    context_pack = _build_namespace_context_pack(project)

    stub_llm = _StubLLM('{"refined_code": ""}')
    orchestrator = LLMOrchestrator(
        llm_port=stub_llm,
        parser_port=SimpleNamespace(),
        context_assembler=SimpleNamespace(),
        context_pack_builder=None,
        symbol_resolver=_StubSymbolResolver(),
        prompt_registry=_StubPromptRegistry(),
        config=OrchestratorConfig(max_plan_retries=1, max_refine_retries=1),
        max_refine_retries=1,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError) as exc_info:
            orchestrator.refine_stage(
                context_pack,
                existing_code="def test_original():\n    assert True\n",
                feedback={},
                project_root=project["project_root"],
            )

    assert "REFINE response missing 'refined_code' or 'tests' content" in str(
        exc_info.value
    )
    assert "usable code payload" in caplog.text
