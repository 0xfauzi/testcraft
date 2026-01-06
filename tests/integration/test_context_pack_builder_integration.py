import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from testcraft.adapters.context.main_adapter import TestcraftContextAdapter
from testcraft.adapters.parsing.codebase_parser import CodebaseParser
from testcraft.application.generation.config import GenerationConfig
from testcraft.application.generation.services.context_assembler import (
    ContextAssembler,
)
from testcraft.application.generation.services.context_pack import ContextPackBuilder
from testcraft.application.generation.services.import_resolver import ImportResolver


@pytest.mark.integration
def test_context_pack_builder_generates_rich_components(tmp_path):
    # Create a realistic package structure that the import resolver can inspect
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "sample-project"
            version = "0.0.1"
            """
        ).strip(),
        encoding="utf-8",
    )

    package_dir = tmp_path / "sample_pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    module_path = package_dir / "services.py"
    module_path.write_text(
        textwrap.dedent(
            """
            class SampleService:
                \"\"\"Service for performing demo calculations.\"\"\"

                def calculate(self, a: int, b: int) -> int:
                    \"\"\"Return the sum and guard against negative outputs.\"\"\"
                    total = a + b
                    if total < 0:
                        raise ValueError("negative totals are not supported")
                    return total
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = CodebaseParser()
    context_port = TestcraftContextAdapter()
    import_resolver = ImportResolver()
    config = GenerationConfig.get_default_config()
    context_assembler = ContextAssembler(
        context_port=context_port,
        parser_port=parser,
        config=config,
        import_resolver=import_resolver,
    )

    # Index the module to exercise the context adapter in an end-to-end fashion
    context_port.index(module_path, project_root=tmp_path)

    builder = ContextPackBuilder(
        import_resolver=import_resolver,
        parser=parser,
        context_assembler=context_assembler,
    )

    context_pack = builder.build_context_pack(
        target_file=module_path,
        target_object="SampleService.calculate",
        project_root=tmp_path,
    )

    assert context_pack is not None
    assert context_pack.focal.signature.startswith("def calculate")
    assert not context_pack.focal.signature.startswith("#")
    assert context_pack.resolved_defs, "Expected resolved definitions for target"
    assert any(
        "calculate" in resolved.signature for resolved in context_pack.resolved_defs
    )
    assert context_pack.import_map is not None
    assert context_pack.import_map.target_import
