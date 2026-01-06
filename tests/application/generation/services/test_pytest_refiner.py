from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.config.models import RefineConfig


class _StubTelemetrySpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def __enter__(self) -> _StubTelemetrySpan:  # pragma: no cover - simple stub
        return self

    def __exit__(
        self, exc_type, exc_val, exc_tb
    ) -> None:  # pragma: no cover - simple stub
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class _StubTelemetryPort:
    def create_child_span(
        self, _: str
    ) -> _StubTelemetrySpan:  # pragma: no cover - trivial stub
        return _StubTelemetrySpan()


class _StubRefinePort:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def refine_from_failures(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self._result


class _RecordingTracker:
    def __init__(self) -> None:
        self.rollbacks: list[tuple[str, str]] = []

    def record_rollback(self, file_path: str, reason: str) -> None:
        self.rollbacks.append((file_path, reason))

    def update_file_status(
        self, *args: Any, **kwargs: Any
    ) -> None:  # pragma: no cover - simple stub
        return None

    def update_refinement_result(
        self, *args: Any, **kwargs: Any
    ) -> None:  # pragma: no cover - simple stub
        return None


@pytest.mark.asyncio
async def test_write_aborts_on_external_mtime_change(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    refiner: PytestRefiner | None = None
    try:
        refine_port = _StubRefinePort({"success": True, "final_status": "success"})
        refiner = PytestRefiner(
            refine_port=refine_port,
            telemetry_port=_StubTelemetryPort(),
            executor=executor,
            config=RefineConfig(enable=True),
        )

        test_file = tmp_path / "test_sample.py"
        test_file.write_text(
            "def test_example():\n    assert False\n", encoding="utf-8"
        )

        assert refiner._backup_original_content(test_file)

        # Simulate an external change after backup
        test_file.write_text("# external change\n", encoding="utf-8")
        baseline_mtime = refiner._content_backups[test_file]["mtime_ns"]
        current_mtime = test_file.stat().st_mtime_ns
        assert baseline_mtime != current_mtime

        write_result = await refiner._write_with_validation_and_rollback(
            test_file=test_file,
            refined_content="def test_example():\n    assert True\n",
            context={},
        )

        assert write_result["success"] is False
        assert any("modified externally" in issue for issue in write_result["issues"])
    finally:
        if refiner is not None:
            await refiner._shutdown_file_queues()
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_refine_iteration_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    refiner: PytestRefiner | None = None
    try:
        refine_port = _StubRefinePort(
            {
                "success": False,
                "final_status": "llm_no_change",
                "refined_content": "def test_example():\n    assert True\n",
            }
        )

        config = RefineConfig(
            enable=True,
            max_retries=0,
            refinement_backoff_sec=0.0,
            iteration_timeout_sec=1.0,
            file_lock_timeout_sec=0.5,
        )

        refiner = PytestRefiner(
            refine_port=refine_port,
            telemetry_port=_StubTelemetryPort(),
            executor=executor,
            config=config,
        )

        test_file = tmp_path / "test_iteration_timeout.py"
        test_file.write_text(
            "def test_iteration_timeout():\n    assert False\n", encoding="utf-8"
        )

        async def slow_pytest(_: str) -> dict[str, Any]:
            await asyncio.sleep(1.5)
            return {"stdout": "", "stderr": "", "returncode": 1}

        async def build_context(_: Path, __: str) -> dict[str, Any]:
            return {}

        monkeypatch.setattr(refiner, "run_pytest", slow_pytest)

        result = await refiner.refine_until_pass(
            str(test_file), max_iterations=1, build_source_context_fn=build_context
        )

        assert result["success"] is False
        assert result["final_status"] == "iteration_timeout"
        assert "exceeded" in result["error"].lower()
    finally:
        if refiner is not None:
            await refiner._shutdown_file_queues()
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_sleep_with_iteration_timeout_triggers_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    refiner: PytestRefiner | None = None
    try:
        refiner = PytestRefiner(
            refine_port=_StubRefinePort({"success": False}),
            telemetry_port=_StubTelemetryPort(),
            executor=executor,
            config=RefineConfig(enable=True),
        )

        loop = asyncio.get_running_loop()
        iteration_deadline = loop.time() + 0.01
        captured: dict[str, str] = {}

        async def timeout_abort(stage: str, failure_output: str) -> dict[str, Any]:
            captured["stage"] = stage
            captured["failure_output"] = failure_output
            return {
                "final_status": "iteration_timeout",
                "stage": stage,
                "failure_output": failure_output,
            }

        result = await refiner._sleep_with_iteration_timeout(
            duration=0.05,
            iteration_deadline=iteration_deadline,
            current_stage="backoff_wait",
            failure_output="timeout",
            timeout_abort=timeout_abort,
        )

        assert result is not None
        assert result["final_status"] == "iteration_timeout"
        assert captured["stage"] == "backoff_wait"
        assert captured["failure_output"] == "timeout"
    finally:
        if refiner is not None:
            await refiner._shutdown_file_queues()
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_write_aborts_when_file_lock_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    refiner: PytestRefiner | None = None
    try:
        config = RefineConfig(enable=True)
        refiner = PytestRefiner(
            refine_port=_StubRefinePort({"success": False}),
            telemetry_port=_StubTelemetryPort(),
            executor=executor,
            config=config,
        )

        # Use a very small lock timeout to keep the test fast
        refiner._lock_timeout = 0.05

        test_file = tmp_path / "test_lock.py"
        test_file.write_text("def test_lock():\n    assert False\n", encoding="utf-8")

        import fcntl

        def blocking_flock(
            fd: int, flags: int
        ) -> None:  # pragma: no cover - simple test helper
            raise BlockingIOError

        monkeypatch.setattr(fcntl, "flock", blocking_flock)

        result = await refiner._write_with_validation_and_rollback(
            test_file=test_file,
            refined_content="def test_lock():\n    assert True\n",
            context={},
        )

        assert result["success"] is False
        assert any("lock" in issue for issue in result["issues"])
    finally:
        if refiner is not None:
            await refiner._shutdown_file_queues()
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_validation_rollback_updates_status_tracker(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    refiner: PytestRefiner | None = None
    try:
        refiner = PytestRefiner(
            refine_port=_StubRefinePort({"success": False}),
            telemetry_port=_StubTelemetryPort(),
            executor=executor,
            config=RefineConfig(enable=True),
        )

        tracker = _RecordingTracker()
        refiner._status_tracker = tracker  # type: ignore[attr-defined]

        test_file = tmp_path / "test_needs_rollback.py"
        test_file.write_text(
            "def test_needs_rollback():\n    assert False\n", encoding="utf-8"
        )

        result = await refiner._write_with_validation_and_rollback(
            test_file=test_file,
            refined_content="",  # Invalid content forces rollback
            context={},
        )

        assert result["rollback_performed"] is True
        assert tracker.rollbacks
        file_path, reason = tracker.rollbacks[0]
        assert file_path.endswith("test_needs_rollback.py")
        assert len(reason) <= 160
    finally:
        if refiner is not None:
            await refiner._shutdown_file_queues()
        executor.shutdown(wait=True)
