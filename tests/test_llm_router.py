import pytest

pytest.importorskip("pydantic")

from testcraft.adapters.llm.router import LLMRouter


def test_provider_status_reports_noop_for_unknown_provider():
    router = LLMRouter(config={"default_provider": "mystery"})

    # Before initialization status is unknown
    status_before = router.provider_status()
    assert status_before["status"] == "unknown"

    router.ensure_adapter()

    status_after = router.provider_status()
    assert status_after["status"] == "noop"
    assert status_after["provider"] == "mystery"
    assert status_after["reason"] == "unknown_provider"
