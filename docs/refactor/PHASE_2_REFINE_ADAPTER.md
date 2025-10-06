# Phase 2: Refactor RefineAdapter

**Objective**: Migrate RefineAdapter to use LLMOrchestrator exclusively

**Impact**: BREAKING CHANGE - RefineAdapter now requires ParserPort

---

## Current State Analysis

**File**: `testcraft/adapters/refine/main_adapter.py`

**Current Constructor** (lines 34-53):
```python
def __init__(
    self,
    llm: LLMPort,
    config: RefineConfig | None = None,
    writer_port: WriterPort | None = None,
    telemetry_port: TelemetryPort | None = None,
):
```

**Current Refinement** (lines 687-689):
```python
response = self.llm.refine_content(
    original_content=current_content,
    refinement_instructions=user_prompt,
)
```

**Problem**: Uses simple prompts, no orchestrator integration

---

## Task 2.1: Add Orchestrator Dependencies

### Step 1: Update Constructor

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...application.generation.services.llm_orchestrator import LLMOrchestrator

from ...ports.llm_port import LLMPort
from ...ports.parser_port import ParserPort  # NEW
from ...ports.writer_port import WriterPort
from ...ports.telemetry_port import TelemetryPort
from ...config.models import RefineConfig


class RefineAdapter:
    """Adapter for AI-powered test refinement using LLMOrchestrator."""

    def __init__(
        self,
        llm: LLMPort,
        parser_port: ParserPort,  # NEW - REQUIRED
        config: RefineConfig | None = None,
        writer_port: WriterPort | None = None,
        telemetry_port: TelemetryPort | None = None,
        llm_orchestrator: LLMOrchestrator | None = None,  # NEW - OPTIONAL
    ):
        """
        Initialize RefineAdapter with orchestrator support.

        Args:
            llm: LLM adapter for low-level operations
            parser_port: Parser for building context (REQUIRED for orchestrator)
            config: Refinement configuration with guardrails
            writer_port: Optional writer for safe file operations
            telemetry_port: Optional telemetry for observability
            llm_orchestrator: Optional pre-initialized orchestrator (lazy if None)
        """
        self.llm = llm
        self.parser_port = parser_port  # NEW
        self.config = config or RefineConfig()
        self.writer_port = writer_port
        self.telemetry_port = telemetry_port

        # Lazy initialization of orchestrator
        self._llm_orchestrator = llm_orchestrator
        self._orchestrator_initialized = llm_orchestrator is not None

        # Extract guardrails (existing code)
        guardrails = self.config.refinement_guardrails
        self.reject_empty = guardrails.get("reject_empty", True)
        self.reject_literal_none = guardrails.get("reject_literal_none", True)
        self.reject_identical = guardrails.get("reject_identical", True)
        self.validate_syntax = guardrails.get("validate_syntax", True)
        self.format_on_refine = guardrails.get("format_on_refine", True)
```

### Step 2: Implement Lazy Orchestrator Initialization

```python
def _ensure_orchestrator(self) -> LLMOrchestrator:
    """
    Ensure orchestrator is initialized (lazy loading).

    Lazy initialization avoids circular imports and allows
    dependency injection when needed.
    """
    if self._orchestrator_initialized:
        return self._llm_orchestrator  # type: ignore

    # Lazy import to avoid circular dependencies
    from ...application.generation.services.context_assembler import ContextAssembler
    from ...application.generation.services.context_pack import ContextPackBuilder
    from ...application.generation.services.symbol_resolver import SymbolResolver
    from ...application.generation.services.llm_orchestrator import LLMOrchestrator
    from ...config.models import OrchestratorConfig

    logger.debug("Lazy-initializing LLMOrchestrator for RefineAdapter")

    # Build orchestrator dependencies
    context_assembler = ContextAssembler(
        context_port=None,  # RefineAdapter doesn't need full context port
        parser_port=self.parser_port,
        config={},
    )
    context_pack_builder = ContextPackBuilder()
    symbol_resolver = SymbolResolver(parser_port=self.parser_port)
    orchestrator_config = OrchestratorConfig()

    self._llm_orchestrator = LLMOrchestrator(
        llm_port=self.llm,
        parser_port=self.parser_port,
        context_assembler=context_assembler,
        context_pack_builder=context_pack_builder,
        symbol_resolver=symbol_resolver,
        config=orchestrator_config,
    )
    self._orchestrator_initialized = True

    return self._llm_orchestrator
```

---

## Task 2.2: Refactor refine_from_failures()

### Replace Lines 687-689

**Current Code**:
```python
response = self.llm.refine_content(
    original_content=current_content,
    refinement_instructions=user_prompt,
)
```

**New Code**:
```python
try:
    # Get orchestrator and build context
    orchestrator = self._ensure_orchestrator()
    context_pack = self._build_context_pack_from_source(
        test_file=test_path,
        source_context=source_context,
    )

    # Prepare feedback dict for orchestrator
    feedback = {
        "result": "failed",
        "trace_excerpt": failure_output,
        "coverage_gaps": {},
        "notes": "Refinement triggered by pytest failures",
    }

    # Use orchestrator.refine_stage() instead of llm.refine_content()
    refined_code = orchestrator.refine_stage(
        context_pack=context_pack,
        existing_code=current_content,
        feedback=feedback,
        project_root=test_path.parent,
    )

except Exception as e:
    logger.exception("Orchestrator refinement failed: %s", e)
    return {
        "success": False,
        "error": f"Orchestrator refinement error: {e}",
        "iterations_used": 1,
        "final_status": "orchestrator_error",
    }
```

### Add Helper: _build_context_pack_from_source()

```python
def _build_context_pack_from_source(
    self,
    test_file: Path,
    source_context: dict[str, Any] | None,
) -> ContextPack:
    """
    Build ContextPack from source context information.

    Attempts to derive the source file being tested and build
    proper context for refinement.

    Args:
        test_file: Path to the test file needing refinement
        source_context: Optional source context dict with keys:
            - source_file: Path to source file
            - target_object: Name of object being tested

    Returns:
        ContextPack for orchestrator refinement

    Raises:
        Exception: If context pack building fails (caller should catch)
    """
    # Extract source file path from context
    source_file = None
    target_object = "test_module"

    if source_context:
        source_file = source_context.get("source_file")
        if source_file:
            source_file = Path(source_file)
        target_object = source_context.get("target_object", target_object)

    # Fallback: Infer source file from test file name
    if not source_file or not source_file.exists():
        source_file = self._infer_source_file(test_file)
        logger.debug(
            "Inferred source file %s from test file %s",
            source_file, test_file
        )

    # Use orchestrator's ContextPackBuilder
    orchestrator = self._ensure_orchestrator()

    try:
        context_pack = orchestrator._context_pack_builder.build_context_pack(
            target_file=source_file,
            target_object=target_object,
            project_root=test_file.parent.parent,
        )
        return context_pack
    except Exception as e:
        logger.warning("ContextPack building failed: %s. Using minimal context.", e)
        return self._create_minimal_context_pack(test_file)


def _infer_source_file(self, test_file: Path) -> Path:
    """
    Infer source file from test file name.

    Examples:
        tests/test_mymodule.py -> src/mymodule.py
        tests/unit/test_mymodule.py -> src/mymodule.py
        tests/test_package_module.py -> src/package/module.py

    Args:
        test_file: Path to test file

    Returns:
        Best-guess path to source file (may not exist)
    """
    # Remove test_ prefix and .py suffix
    stem = test_file.stem.replace("test_", "")

    # Common source directory patterns
    potential_paths = [
        test_file.parent.parent / "testcraft" / f"{stem}.py",
        test_file.parent.parent / "src" / f"{stem}.py",
        test_file.parent.parent / f"{stem}.py",
    ]

    # Handle package.module patterns (e.g., test_adapters_llm -> adapters/llm.py)
    if "_" in stem:
        parts = stem.split("_")
        potential_paths.extend([
            test_file.parent.parent / "testcraft" / "/".join(parts[:-1]) / f"{parts[-1]}.py",
            test_file.parent.parent / "src" / "/".join(parts[:-1]) / f"{parts[-1]}.py",
        ])

    # Return first existing path
    for path in potential_paths:
        if path.exists():
            logger.debug("Found source file: %s", path)
            return path

    # Last resort: use test file itself
    logger.warning(
        "Could not infer source file for %s, using test file as fallback",
        test_file
    )
    return test_file


def _create_minimal_context_pack(self, test_file: Path) -> ContextPack:
    """Create minimal ContextPack when building fails."""
    from ...domain.models import (
        ContextPack, Target, ImportMap, Focal,
        PropertyContext, Conventions, Budget
    )

    return ContextPack(
        target=Target(module_file=str(test_file), object="test_module"),
        import_map=ImportMap(target_import=f"import {test_file.stem}"),
        focal=Focal(source="", signature="", docstring=None),
        resolved_defs=[],
        property_context=PropertyContext(),
        conventions=Conventions(),
        budget=Budget(),
        context="Minimal context for refinement (context building failed)",
    )
```

### Add Validation: _validate_refined_code()

```python
def _validate_refined_code(
    self,
    original: str,
    refined: str,
) -> dict[str, Any]:
    """
    Apply RefineConfig guardrails to refined code.

    Returns:
        {"valid": bool, "reason": str | None}
    """
    # Empty check
    if self.reject_empty and not refined.strip():
        return {"valid": False, "reason": "Empty refinement rejected by guardrails"}

    # Literal None check
    if self.reject_literal_none and refined.strip().lower() in ("none", "null"):
        return {"valid": False, "reason": "Literal 'None' rejected by guardrails"}

    # Identical content check
    if self.reject_identical and refined.strip() == original.strip():
        return {"valid": False, "reason": "Identical content rejected by guardrails"}

    # Syntax validation
    if self.validate_syntax:
        try:
            import ast
            ast.parse(refined)
        except SyntaxError as e:
            return {"valid": False, "reason": f"Syntax validation failed: {e}"}

    return {"valid": True, "reason": None}
```

---

## Task 2.3: Update Dependency Injection

**File**: `testcraft/cli/dependency_injection.py`
**Location**: Around line 93

**Current**:
```python
container["refine_adapter"] = RefineAdapter(
    llm=container["llm_adapter"],
    config=refine_config,
    writer_port=container["writer_adapter"],
    telemetry_port=container["telemetry_adapter"],
)
```

**Updated**:
```python
container["refine_adapter"] = RefineAdapter(
    llm=container["llm_adapter"],
    parser_port=container["parser_adapter"],  # ADD THIS
    config=refine_config,
    writer_port=container["writer_adapter"],
    telemetry_port=container["telemetry_adapter"],
)
```

---

## Task 2.4: Update Tests

**File**: `tests/test_refine_adapters.py`

### Add Orchestrator Integration Test

```python
def test_orchestrator_integration(mock_llm, mock_parser):
    """Test that RefineAdapter integrates with orchestrator."""
    from testcraft.adapters.refine.main_adapter import RefineAdapter
    from testcraft.config.models import RefineConfig

    adapter = RefineAdapter(
        llm=mock_llm,
        parser_port=mock_parser,
        config=RefineConfig(),
    )

    # Verify orchestrator is lazily initialized
    assert not adapter._orchestrator_initialized

    # Trigger initialization
    orchestrator = adapter._ensure_orchestrator()
    assert adapter._orchestrator_initialized
    assert orchestrator is not None


def test_context_pack_building(tmp_path, mock_parser):
    """Test ContextPack building from source context."""
    # Create test file
    test_file = tmp_path / "tests" / "test_module.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_example(): pass")

    # Create source file
    source_file = tmp_path / "testcraft" / "module.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("def example(): pass")

    adapter = RefineAdapter(llm=mock_llm, parser_port=mock_parser)
    context_pack = adapter._build_context_pack_from_source(
        test_file=test_file,
        source_context={"source_file": str(source_file)},
    )

    assert context_pack is not None
    assert source_file.name in context_pack.target.module_file
```

### Update Existing Tests

```python
# Update all RefineAdapter instantiations to include parser_port:

adapter = RefineAdapter(
    llm=mock_llm,
    parser_port=mock_parser,  # ADD THIS
    config=RefineConfig(),
)
```

---

## Validation

### Test 1: Unit Tests
```bash
pytest tests/test_refine_adapters.py -v
```

**Expected**: All tests pass

### Test 2: Integration Test
```bash
pytest tests/test_pytest_refiner_integration.py -v
```

**Expected**: Refinement uses orchestrator

### Test 3: Manual Refinement
```bash
# Create failing test
echo 'def test_fails(): assert False' > /tmp/test_temp.py

# Run refinement (would need actual integration)
pytest /tmp/test_temp.py || echo "Test failed as expected"
```

---

## Rollback

If this breaks:
```bash
git checkout testcraft/adapters/refine/main_adapter.py
git checkout testcraft/cli/dependency_injection.py
git checkout tests/test_refine_adapters.py
```

---

## Completion Checklist

- [ ] Constructor updated with ParserPort and llm_orchestrator parameters
- [ ] Lazy orchestrator initialization implemented
- [ ] refine_from_failures() refactored to use orchestrator.refine_stage()
- [ ] Context pack building helpers implemented
- [ ] Validation helpers implemented
- [ ] Dependency injection updated
- [ ] All tests updated and passing
- [ ] Integration tests pass
- [ ] No regressions in refinement functionality

---

**Next**: [PHASE_3_REMOVE_FALLBACK.md](./PHASE_3_REMOVE_FALLBACK.md) (if created)
