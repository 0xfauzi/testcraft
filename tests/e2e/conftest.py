from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import tomli_w

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


@dataclass(slots=True)
class CommandSpec:
    """Single CLI invocation with optional expectations."""

    args: Sequence[str]
    expect_exit: int = 0
    logfile: str | None = None
    timeout: int | None = None
    env_updates: dict[str, str] | None = None
    description: str | None = None
    post_run: Callable[[CommandResult, Path, dict[str, str]], None] | None = None


@dataclass(slots=True)
class CommandResult:
    """Result for a command executed via run_cli_chain."""

    spec: CommandSpec
    exit_code: int
    output: str
    log_path: Path
    duration: float


@dataclass(slots=True)
class ScenarioContext:
    """Context object surfaced to scenario assertions."""

    project_root: Path
    env: dict[str, str]
    logs_dir: Path
    results: list[CommandResult]


ScenarioAssertion = Callable[[ScenarioContext], None]


@dataclass(slots=True)
class ScenarioInput:
    """Inputs passed to scenario factories to build command chains."""

    project_root: Path
    source_file: Path
    extra_source_files: list[Path]
    tests_root: Path
    logs_dir: Path
    provider: str
    env: dict[str, str]


@dataclass(slots=True)
class ScenarioSpec:
    """Curated scenario describing a CLI journey."""

    name: str
    build_commands: Callable[[ScenarioInput], list[CommandSpec]]
    assertions: list[ScenarioAssertion] = field(default_factory=list)
    live_services: bool = True
    multi_source_project: bool = False
    notes: str | None = None
    env_overrides: dict[str, str] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _strip_quotes(value: str) -> str:
    """Remove matching quotes around a value."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> dict[str, str]:
    """Load key=value pairs from a .env-style file."""
    if not path.exists():
        return {}

    env_vars: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = _strip_quotes(raw_value.strip())
        if key:
            env_vars[key] = value
    return env_vars


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from CLI output."""
    return ANSI_ESCAPE_RE.sub("", text)


def _write_module(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def dummy_project_factory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[bool], Path]:
    def _factory(multi_source: bool = False) -> Path:
        project_root = tmp_path_factory.mktemp("demo_project")
        package_dir = project_root / "src" / "demo_app"
        package_name = package_dir.name
        tests_root = project_root / "tests"
        unit_tests_dir = tests_root / "unit"
        generated_tests_dir = tests_root / "src"
        mirrored_tests_dir = generated_tests_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        unit_tests_dir.mkdir(parents=True, exist_ok=True)
        generated_tests_dir.mkdir(parents=True, exist_ok=True)
        mirrored_tests_dir.mkdir(parents=True, exist_ok=True)

        (project_root / "pyproject.toml").write_text(
            textwrap.dedent(
                """
                [project]
                name = "demo-project"
                version = "0.0.1"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        (package_dir / "__init__.py").write_text(
            '"""Demo application package for end-to-end testing."""\n',
            encoding="utf-8",
        )
        (tests_root / "__init__.py").write_text("", encoding="utf-8")
        (unit_tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (generated_tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (mirrored_tests_dir / "__init__.py").write_text("", encoding="utf-8")

        sources: dict[str, str] = {
            "calculator.py": textwrap.dedent(
                """
                \"\"\"Simple arithmetic helpers used during test generation.\"\"\"


                def add(a: float, b: float) -> float:
                    return a + b
                """
            ).strip()
        }

        if multi_source:
            sources["geometry.py"] = textwrap.dedent(
                """
                \"\"\"Geometry helpers for batch scenarios.\"\"\"


                def area_of_square(edge: float) -> float:
                    return edge * edge
                """
            ).strip()

            sources["statistics.py"] = textwrap.dedent(
                """
                \"\"\"Statistics helpers to exercise multi-file pipelines.\"\"\"


                def mean(values: list[float]) -> float:
                    return sum(values) / len(values) if values else 0.0
                """
            ).strip()

        for filename, contents in sources.items():
            (package_dir / filename).write_text(contents + "\n", encoding="utf-8")

        return project_root

    return _factory


@pytest.fixture
def dummy_project(dummy_project_factory: Callable[[bool], Path]) -> Path:
    return dummy_project_factory()


def _apply_pythonpath(env: dict[str, str], project_root: Path, repo_root: Path) -> None:
    pythonpath_entries = [str(project_root), str(repo_root)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)


def _create_sitecustomize(mode: str, base_dir: Path) -> Path:
    module_dir = base_dir / "sitecustomize"
    module_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = module_dir / "__init__.py"
    fake_module = (
        textwrap.dedent(
            f"""
        from __future__ import annotations

        import logging
        import types

        from testcraft.cli import dependency_injection as _di

        _LOGGER = logging.getLogger("testcraft.tests.e2e.fake_di")

        _MODE_RAW = {mode!r}
        _MODES = [part.strip() for part in _MODE_RAW.split(",") if part.strip()]


        def _wrap_container(original, config):
            container = original(config)
            if "fake-llm" in _MODES:
                class _FakeLLM:
                    def provider_status(self):
                        return {{
                            "status": "noop",
                            "provider": "fake",
                            "reason": "offline",
                        }}

                    def ensure_adapter(self):
                        return False

                    def __getattr__(self, item):
                        raise RuntimeError(f"LLM adapter method '{{item}}' not supported in fake mode")

                container["llm_adapter"] = _FakeLLM()

                generate_usecase = container.get("generate_usecase")
                if generate_usecase is not None:
                    if "fake-generate-fail" in _MODES:
                        async def _fail_generate_tests(self, *args, **kwargs):
                            raise RuntimeError("Simulated generation failure")

                        generate_usecase.generate_tests = types.MethodType(
                            _fail_generate_tests, generate_usecase
                        )
                    else:
                        async def _fake_generate_tests(self, *args, **kwargs):
                            return {{
                                "success": True,
                                "files_processed": 0,
                                "tests_generated": 0,
                                "files_written": 0,
                                "warnings": ["LLM offline - generation skipped"],
                            }}

                        generate_usecase.generate_tests = types.MethodType(
                            _fake_generate_tests, generate_usecase
                        )

                    orchestrator = getattr(generate_usecase, "_llm_orchestrator", None)
                    if orchestrator is not None:
                        def _fake_plan_stage(*_args, **_kwargs):
                            return {{
                                "plan_id": "offline-plan",
                                "quality_gates": [],
                                "steps": [],
                            }}

                        orchestrator.plan_stage = _fake_plan_stage

            if "fake-cost" in _MODES:
                class _FakeCost:
                    def __getattr__(self, _):
                        raise RuntimeError("Cost adapter unavailable")

                container["cost_adapter"] = _FakeCost()
            if "fake-utility" in _MODES:
                utility = container.get("utility_usecase")
                if utility is not None:
                    def _fail_sync(_self, *args, **kwargs):
                        raise RuntimeError("Sync state disabled in fake mode")

                    def _fail_reset(_self, *args, **kwargs):
                        raise RuntimeError("Reset state disabled in fake mode")

                    utility.sync_state = types.MethodType(_fail_sync, utility)
                    utility.reset_state = types.MethodType(_fail_reset, utility)

            return container


        if any(mode in {"fake-llm", "fake-cost", "fake-utility", "fake-generate-fail"} for mode in _MODES):
            if getattr(_di.create_dependency_container, "__wrapped__", None) is None:
                original = _di.create_dependency_container

                def _patched(config):
                    return _wrap_container(original, config)

                _patched.__wrapped__ = original  # type: ignore[attr-defined]
                _di.create_dependency_container = _patched
                _LOGGER.debug("Injected fake dependency container for modes %%s", _MODES)
            else:
                _LOGGER.debug("Fake dependency container already patched")

        if "auto-confirm" in _MODES:
            try:
                from rich.prompt import Confirm

                def _always_confirm(*_args, **_kwargs):
                    return True

                Confirm.ask = _always_confirm  # type: ignore[assignment]
                _LOGGER.debug("Patched Confirm.ask for auto-confirm mode")
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning("Failed to patch Confirm.ask: %%s", exc)

        if "manual-fix-decline" in _MODES:
            try:
                import click as _click

                def _decline(*_args, **_kwargs):
                    return False

                _click.confirm = _decline  # type: ignore[assignment]
                _LOGGER.debug("Patched click.confirm to decline manual-fix prompts")
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning("Failed to patch click.confirm: %%s", exc)
        """
        ).strip()
        + "\n"
    )
    _write_module(sitecustomize_path, fake_module)
    return module_dir


@pytest.fixture
def cli_env_factory(
    dummy_project_factory: Callable[[bool], Path],
) -> Callable[..., dict[str, str]]:
    from testcraft.config.models import TestCraftConfig

    repo_root = Path(__file__).resolve().parents[2]

    def _factory(
        *,
        live: bool = True,
        project_root: Path | None = None,
        extra_env: dict[str, str] | None = None,
        fake_mode: str | None = None,
    ) -> dict[str, str]:
        root = project_root or dummy_project_factory()
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

        if live:
            env_file = repo_root / ".env"
            if not env_file.exists():
                pytest.fail(
                    "E2E CLI workflow requires real credentials loaded from a .env file at the repository root."
                )

            for key, value in _load_env_file(env_file).items():
                env.setdefault(key, value)

            required_envs: dict[str, list[str]] = {
                "openai": ["OPENAI_API_KEY"],
                "anthropic": ["ANTHROPIC_API_KEY"],
                "azure-openai": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
                "bedrock": [
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_REGION",
                ],
            }

            default_provider = (
                TestCraftConfig().llm.default_provider or "openai"
            ).lower()
            explicit_provider = env.get("TESTCRAFT_E2E_PROVIDER")
            provider = (explicit_provider or default_provider).lower()

            def has_credentials(name: str) -> bool:
                return all(env.get(var) for var in required_envs.get(name, []))

            if not explicit_provider and not has_credentials(provider):
                fallback = next(
                    (name for name in required_envs if has_credentials(name)),
                    None,
                )
                if fallback:
                    provider = fallback

            if provider not in required_envs:
                pytest.fail(f"Unsupported provider configured for E2E test: {provider}")

            missing_vars = [var for var in required_envs[provider] if not env.get(var)]

            if missing_vars:
                pytest.fail(
                    "E2E CLI workflow requires real credentials. "
                    "Missing environment variables: "
                    + ", ".join(sorted(missing_vars))
                    + ". Set TESTCRAFT_E2E_PROVIDER to a provider with credentials or "
                    "export the required variables."
                )

            env["TESTCRAFT_E2E_PROVIDER"] = provider
        else:
            env.setdefault("TESTCRAFT_E2E_PROVIDER", "openai")
            env.setdefault("TESTCRAFT_FAKE_RUN", "1")

        if fake_mode:
            harness_dir = root / ".tmp" / "fake_di"
            sitecustomize_dir = _create_sitecustomize(fake_mode, harness_dir)
            existing_pythonpath = env.get("PYTHONPATH")
            entries = [str(sitecustomize_dir), str(root), str(repo_root)]
            if existing_pythonpath:
                entries.append(existing_pythonpath)
            env["PYTHONPATH"] = os.pathsep.join(entries)
        else:
            _apply_pythonpath(env, root, repo_root)

        if extra_env:
            env.update(extra_env)

        env.setdefault("TESTCRAFT_LOG_LEVEL", "INFO")

        return env

    return _factory


@pytest.fixture
def cli_env(
    cli_env_factory: Callable[..., dict[str, str]], dummy_project: Path
) -> dict[str, str]:
    return cli_env_factory(project_root=dummy_project)


def _resolve_logfile(cwd: Path, logs_dir: Path, log_name: str) -> Path:
    candidate = logs_dir / log_name
    candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        return candidate.relative_to(cwd)
    except ValueError:
        raise ValueError("Log directories must be within the project root")


@pytest.fixture
def run_cli() -> Callable[
    [Sequence[str], Path, dict[str, str], str, int | None], tuple[int, str, Path, float]
]:
    def _run_cli(
        command: Sequence[str],
        cwd: Path,
        env: dict[str, str],
        logfile_name: str,
        timeout: int | None = 420,
    ) -> tuple[int, str, Path, float]:
        log_path = cwd / logfile_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-m", "testcraft.cli.main", *command]
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                args,
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if timeout is None:
                proc.wait()
            else:
                deadline = start + timeout
                interval = 5.0
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        proc.kill()
                        raise RuntimeError(
                            f"Command timed out ({timeout}s): {' '.join(command)}"
                        )
                    try:
                        proc.wait(timeout=min(interval, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        interval = min(interval * 2, 60.0)

        output = log_path.read_text(encoding="utf-8")
        duration = time.monotonic() - start
        return proc.returncode, output, log_path, duration

    return _run_cli


def run_cli_chain(
    commands: Sequence[CommandSpec],
    *,
    run_cli_impl: Callable[
        [Sequence[str], Path, dict[str, str], str, int | None],
        tuple[int, str, Path, float],
    ],
    cwd: Path,
    env: dict[str, str],
    logs_dir: Path,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    logs_dir.mkdir(parents=True, exist_ok=True)
    for idx, spec in enumerate(commands):
        log_name = spec.logfile or f"{idx:02d}_{spec.args[0].replace(' ', '_')}.log"
        log_relative = _resolve_logfile(cwd, logs_dir, log_name)
        command_env = env.copy()
        if spec.env_updates:
            command_env.update(spec.env_updates)
        exit_code, output, log_path, duration = run_cli_impl(
            list(spec.args),
            cwd=cwd,
            env=command_env,
            logfile_name=str(log_relative),
            timeout=spec.timeout,
        )
        result = CommandResult(
            spec=spec,
            exit_code=exit_code,
            output=output,
            log_path=log_path,
            duration=duration,
        )
        if spec.expect_exit is not None:
            assert exit_code == spec.expect_exit, (
                f"Command {' '.join(spec.args)} exited with {exit_code}, expected {spec.expect_exit}. Log: {log_path}"
            )
        results.append(result)
        if spec.post_run:
            spec.post_run(result, cwd, env)
    return results


@pytest.fixture
def tmp_catalog_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    catalog_src = repo_root / "testcraft" / "config" / "model_catalog.toml"
    if not catalog_src.exists():
        pytest.skip("Model catalog not available")
    target_dir = tmp_path_factory.mktemp("catalog_snapshot")
    snapshot_path = target_dir / "model_catalog_snapshot.toml"
    snapshot_path.write_bytes(catalog_src.read_bytes())
    return snapshot_path


def _configure_project_command(
    provider: str, model: str | None, scenario: str
) -> CommandSpec:
    def _post(result: CommandResult, cwd: Path, _env: dict[str, str]) -> None:
        config_path = cwd / ".testcraft.toml"
        config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        config_data.setdefault("llm", {})
        config_data["llm"]["default_provider"] = provider
        generation_section = config_data.setdefault("generation", {})
        generation_section.setdefault("enable_refinement", True)
        generation_section.setdefault("immediate_refinement", True)
        generation_section.setdefault("plan_first", True)
        refine_section = generation_section.setdefault("refine", {})
        refine_section.setdefault("enable", True)
        if model:
            key_map = {
                "openai": "openai_model",
                "anthropic": "anthropic_model",
                "azure-openai": "azure_openai_deployment",
                "bedrock": "bedrock_model_id",
            }
            target_key = key_map.get(provider)
            if target_key:
                config_data["llm"][target_key] = model
        config_path.write_text(tomli_w.dumps(config_data), encoding="utf-8")

    return CommandSpec(
        args=["init-config", "--minimal"],
        logfile=f"{scenario}/01_init.log",
        post_run=_post,
    )


def _baseline_scenarios() -> list[ScenarioSpec]:
    def _common_commands(
        scenario_name: str,
        include_models: bool = False,
        *,
        plan_first: bool = False,
        dry_run: bool = False,
        manual_fix: bool = False,
        coverage: bool = False,
        analyze_prefix: Sequence[str] | None = None,
        status_prefix: Sequence[str] | None = None,
        models_prefix: Sequence[str] | None = None,
        generate_global_prefix: Sequence[str] | None = None,
        generate_extra_args: Sequence[str] | None = None,
    ) -> Callable[[ScenarioInput], list[CommandSpec]]:
        def _builder(params: ScenarioInput) -> list[CommandSpec]:
            commands: list[CommandSpec] = []

            def _rel(path: Path) -> str:
                if path.is_absolute():
                    try:
                        return str(path.relative_to(params.project_root))
                    except ValueError:
                        return str(path)
                return str(path)

            configure = _configure_project_command(
                params.provider, params.env.get("TESTCRAFT_E2E_MODEL"), scenario_name
            )
            commands.append(configure)
            if dry_run:
                dry_args = [
                    "generate",
                    ".",
                    "--dry-run",
                    "--target-files",
                    _rel(params.source_file),
                ]
                for extra_file in params.extra_source_files:
                    dry_args.extend(["--target-files", _rel(extra_file)])
                commands.append(
                    CommandSpec(
                        args=dry_args,
                        logfile=f"{scenario_name}/02_dry_generate.log",
                    )
                )
                analyze_args = [
                    "analyze",
                    ".",
                    "--target-files",
                    _rel(params.source_file),
                ]
                for extra_file in params.extra_source_files:
                    analyze_args.extend(["--target-files", _rel(extra_file)])
                commands.append(
                    CommandSpec(
                        args=analyze_args,
                        logfile=f"{scenario_name}/03_analyze.log",
                    )
                )
                commands.append(
                    CommandSpec(
                        args=["status", ".", "--limit", "1"],
                        logfile=f"{scenario_name}/04_status.log",
                    )
                )
                return commands

            analyze_args = list(analyze_prefix or []) + [
                "analyze",
                ".",
                "--target-files",
                _rel(params.source_file),
            ]
            for extra_file in params.extra_source_files:
                analyze_args.extend(["--target-files", _rel(extra_file)])
            commands.append(
                CommandSpec(
                    args=analyze_args,
                    logfile=f"{scenario_name}/02_analyze.log",
                )
            )

            if plan_first:
                plan_output = (
                    params.project_root
                    / ".testcraft"
                    / "artifacts"
                    / f"{scenario_name}_plan.json"
                )
                commands.append(
                    CommandSpec(
                        args=[
                            "plan",
                            "--disable-gates",
                            "--target-file",
                            str(params.source_file),
                            "--target-object",
                            "add",
                            "--output",
                            str(plan_output),
                        ],
                        logfile=f"{scenario_name}/03_plan.log",
                    )
                )
                generate_logfile = f"{scenario_name}/04_generate.log"
            else:
                generate_logfile = f"{scenario_name}/03_generate.log"

            generate_args = list(generate_global_prefix or []) + [
                "generate",
                ".",
                "--target-files",
                _rel(params.source_file),
                "--auto-accept-fixes",
            ]
            for extra_file in params.extra_source_files:
                generate_args.extend(["--target-files", _rel(extra_file)])
            if plan_first:
                generate_args.extend(["--plan-first", "--auto-accept-plan"])
            if manual_fix:
                generate_args.extend(["--manual-fix-on-fail", "--keep-failed-writes"])
            if generate_extra_args:
                generate_args.extend(generate_extra_args)

            commands.append(
                CommandSpec(
                    args=generate_args,
                    logfile=generate_logfile,
                    timeout=900,
                )
            )

            if manual_fix:

                def _ensure_playbook(
                    _: CommandResult, cwd: Path, _env: dict[str, str]
                ) -> None:
                    artifacts_dir = cwd / ".testcraft" / "artifacts"
                    markdowns = list(artifacts_dir.glob("**/*.md"))
                    if not markdowns:
                        raise AssertionError("Expected manual-fix markdown artifact")

                commands.append(
                    CommandSpec(
                        args=[
                            "manual-fix",
                            "--playbook",
                            _rel(
                                params.tests_root
                                / "src"
                                / "demo_app"
                                / "test_calculator.py"
                            ),
                        ],
                        logfile=f"{scenario_name}/05_manual_fix.log",
                        post_run=_ensure_playbook,
                    )
                )

            if coverage:
                commands.append(
                    CommandSpec(
                        args=[
                            "coverage",
                            ".",
                            "--source-files",
                            _rel(params.source_file),
                            "--test-files",
                            _rel(
                                params.tests_root
                                / "src"
                                / "demo_app"
                                / "test_calculator.py"
                            ),
                            "--output-dir",
                            ".testcraft/artifacts/coverage",
                            "--format",
                            "detailed",
                            "--format",
                            "json",
                        ],
                        logfile=f"{scenario_name}/05_coverage.log",
                    )
                )

            status_args = list(status_prefix or []) + [
                "status",
                ".",
                "--history",
                "--statistics",
            ]
            commands.append(
                CommandSpec(
                    args=status_args,
                    logfile=f"{scenario_name}/99_status.log",
                )
            )

            if include_models:
                models_args = list(models_prefix or []) + [
                    "models",
                    "show",
                    "--provider",
                    params.provider,
                    "--format",
                    "json",
                ]
                commands.append(
                    CommandSpec(
                        args=models_args,
                        logfile=f"{scenario_name}/99_models.log",
                    )
                )

            return commands

        return _builder

    def _batch_refine_commands(params: ScenarioInput) -> list[CommandSpec]:
        commands: list[CommandSpec] = []

        def _rel(path: Path) -> str:
            if path.is_absolute():
                try:
                    return str(path.relative_to(params.project_root))
                except ValueError:
                    return str(path)
            return str(path)

        configure = _configure_project_command(
            params.provider, params.env.get("TESTCRAFT_E2E_MODEL"), "batch_refine"
        )
        commands.append(configure)

        analyze_args = [
            "--verbose",
            "analyze",
            ".",
            "--target-files",
            _rel(params.source_file),
        ]
        for extra in params.extra_source_files:
            analyze_args.extend(["--target-files", _rel(extra)])
        commands.append(
            CommandSpec(
                args=analyze_args,
                logfile="batch_refine/02_analyze.log",
            )
        )

        generate_args = [
            "generate",
            ".",
            "--target-files",
            _rel(params.source_file),
        ]
        for extra in params.extra_source_files:
            generate_args.extend(["--target-files", _rel(extra)])
        generate_args.extend(
            [
                "--auto-accept-fixes",
                "--plan-first",
                "--auto-accept-plan",
                "--batch-size",
                "2",
                "--enable-symbol-resolution",
                "--no-immediate",
                "--max-plan-retries",
                "1",
                "--max-refine-retries",
                "1",
            ]
        )
        commands.append(
            CommandSpec(
                args=generate_args,
                logfile="batch_refine/03_generate.log",
                timeout=900,
            )
        )

        commands.append(
            CommandSpec(
                args=[
                    "status",
                    ".",
                    "--history",
                    "--statistics",
                ],
                logfile="batch_refine/99_status.log",
            )
        )

        return commands

    return [
        ScenarioSpec(
            name="live_baseline",
            build_commands=_common_commands(
                "live_baseline",
                include_models=True,
                analyze_prefix=("--verbose",),
                models_prefix=("--ui", "minimal"),
            ),
            live_services=True,
            notes="Baseline init→analyze→generate→status→models journey",
            metadata={"plan_expected": False},
        ),
        ScenarioSpec(
            name="plan_and_generate",
            build_commands=_common_commands(
                "plan_and_generate",
                plan_first=True,
                analyze_prefix=("--verbose",),
            ),
            live_services=True,
            metadata={
                "plan_output_rel": Path(
                    ".testcraft/artifacts/plan_and_generate_plan.json"
                ),
                "plan_first": True,
            },
        ),
        ScenarioSpec(
            name="dry_run_chain",
            build_commands=_common_commands(
                "dry_run_chain",
                plan_first=False,
                dry_run=True,
                status_prefix=("--quiet",),
            ),
            metadata={"dry_run": True},
        ),
        ScenarioSpec(
            name="manual_fix_followup",
            build_commands=_common_commands(
                "manual_fix_followup",
                manual_fix=True,
                generate_extra_args=("--streaming", "--disable-ruff"),
            ),
            notes="Manual fix after generation",
            metadata={"manual_fix": True, "disable_ruff": True, "streaming": True},
        ),
        ScenarioSpec(
            name="coverage_pass",
            build_commands=_common_commands(
                "coverage_pass",
                coverage=True,
                models_prefix=("--ui", "minimal", "--compact"),
            ),
            notes="Coverage run after generation",
            metadata={
                "coverage_dir_rel": Path(".testcraft/artifacts/coverage"),
                "formats": ["detailed", "json"],
            },
        ),
        ScenarioSpec(
            name="batch_refine",
            build_commands=_batch_refine_commands,
            live_services=True,
            multi_source_project=True,
            metadata={
                "batch_size": 2,
                "extra_sources": [
                    Path("src/demo_app/geometry.py"),
                    Path("src/demo_app/statistics.py"),
                ],
            },
        ),
    ]


@pytest.fixture
def scenario_matrix() -> list[ScenarioSpec]:
    return _baseline_scenarios()


__all__ = [
    "CommandSpec",
    "CommandResult",
    "ScenarioContext",
    "ScenarioInput",
    "ScenarioSpec",
    "dummy_project",
    "dummy_project_factory",
    "cli_env",
    "cli_env_factory",
    "run_cli",
    "run_cli_chain",
    "strip_ansi",
    "scenario_matrix",
    "tmp_catalog_copy",
]
