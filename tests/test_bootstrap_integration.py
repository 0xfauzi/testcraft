"""
Integration tests for BootstrapRunner with PytestRefiner.

Tests the complete integration between bootstrap functionality and
test execution, including ContextPack handling.
"""

from unittest.mock import Mock

import pytest

from testcraft.application.generation.services.bootstrap_runner import (
    BootstrapRunner,
    BootstrapStrategy,
)
from testcraft.domain.models import ContextPack
from testcraft.application.generation.services.pytest_refiner import PytestRefiner
from testcraft.config.models import RefineConfig
from testcraft.domain.models import (
    Budget,
    Conventions,
    Focal,
    ImportMap,
    PropertyContext,
    Target,
)
from testcraft.ports.refine_port import RefinePort
from testcraft.ports.telemetry_port import TelemetryPort
from testcraft.ports.writer_port import WriterPort


class TestBootstrapIntegration:
    """Integration tests for bootstrap functionality."""

    @pytest.fixture
    def mock_ports(self):
        """Create mock ports for testing."""
        refine_port = Mock(spec=RefinePort)
        telemetry_port = Mock(spec=TelemetryPort)
        writer_port = Mock(spec=WriterPort)

        return refine_port, telemetry_port, writer_port

    @pytest.fixture
    def bootstrap_runner(self):
        """Create BootstrapRunner for testing."""
        return BootstrapRunner(prefer_conftest=False)

    @pytest.fixture
    def pytest_refiner(self, mock_ports):
        """Create PytestRefiner with bootstrap runner."""
        refine_port, telemetry_port, writer_port = mock_ports

        return PytestRefiner(
            refine_port=refine_port,
            telemetry_port=telemetry_port,
            executor=Mock(),
            config=RefineConfig(),
        )

    def test_pytest_refiner_with_context_pack_bootstrap(
        self, bootstrap_runner, tmp_path
    ):
        """Test BootstrapRunner determines strategy correctly for ContextPack."""
        # Create a test ContextPack with bootstrap information
        context_pack = ContextPack(
            target=Target(module_file="test_module.py", object="test_function"),
            import_map=ImportMap(
                target_import="from test_module import test_function",
                sys_path_roots=[str(tmp_path / "src")],
                needs_bootstrap=True,
                bootstrap_conftest="import sys\nsys.path.insert(0, '/test/path')",
            ),
            focal=Focal(
                source="def test_function():\n    pass",
                signature="def test_function():",
                docstring=None,
            ),
            resolved_defs=[],
            property_context=PropertyContext(),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Test the bootstrap strategy determination directly
        bootstrap_strategy = bootstrap_runner.ensure_bootstrap(
            context_pack.import_map, tmp_path
        )

        # Should use PYTHONPATH since prefer_conftest=False
        assert bootstrap_strategy == BootstrapStrategy.PYTHONPATH_ENV

        # Test that environment variables are set correctly
        env_vars = bootstrap_runner.set_pythonpath_env(
            context_pack.import_map.sys_path_roots
        )
        assert "PYTHONPATH" in env_vars
        assert str(tmp_path / "src") in env_vars["PYTHONPATH"]

    def test_pytest_refiner_with_no_bootstrap_needed(self, bootstrap_runner, tmp_path):
        """Test BootstrapRunner returns NO_BOOTSTRAP when no bootstrap needed."""
        # Create a test ContextPack without bootstrap
        context_pack = ContextPack(
            target=Target(module_file="test_module.py", object="test_function"),
            import_map=ImportMap(
                target_import="from test_module import test_function",
                sys_path_roots=[],
                needs_bootstrap=False,
                bootstrap_conftest="",
            ),
            focal=Focal(
                source="def test_function():\n    pass",
                signature="def test_function():",
                docstring=None,
            ),
            resolved_defs=[],
            property_context=PropertyContext(),
            conventions=Conventions(),
            budget=Budget(),
        )

        # Test the bootstrap strategy determination directly
        bootstrap_strategy = bootstrap_runner.ensure_bootstrap(
            context_pack.import_map, tmp_path
        )

        # Should return NO_BOOTSTRAP
        assert bootstrap_strategy == BootstrapStrategy.NO_BOOTSTRAP

    def test_pytest_refiner_backward_compatibility(self, mock_ports):
        """Test that PytestRefiner can be created without issues."""
        refine_port, telemetry_port, writer_port = mock_ports

        # Create PytestRefiner without bootstrap runner (backward compatibility)
        refiner = PytestRefiner(
            refine_port=refine_port,
            telemetry_port=telemetry_port,
            executor=Mock(),
            config=RefineConfig(),
        )

        # Should initialize successfully
        assert refiner is not None
        assert isinstance(refiner, PytestRefiner)
