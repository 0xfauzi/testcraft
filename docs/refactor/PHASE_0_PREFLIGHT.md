# Phase 0: Pre-Flight Fixes & Validation

**Priority**: CRITICAL
**Must Complete**: Before ANY refactoring

---

## Task 0.1: Fix --dry-run Mode

### Problem Analysis

**Current Behavior** (`testcraft/cli/main.py:303-309`):
```python
if ctx.obj.dry_run:
    ctx.obj.ui.display_info(
        "DRY RUN: No tests will actually be generated", "Dry Run Mode"
    )
    # ❌ PROBLEM: Only displays message, execution continues
```

**Root Cause**: `GenerateUseCase` has no awareness of dry_run flag

### Solution

#### Step 1: Update GenerateUseCase.__init__()

**File**: `testcraft/application/generate_usecase.py`

```python
def __init__(
    self,
    llm_port: LLMPort,
    writer_port: WriterPort,
    coverage_port: CoveragePort,
    refine_port: RefinePort,
    context_port: ContextPort,
    parser_port: ParserPort,
    state_port: StatePort,
    telemetry_port: TelemetryPort,
    file_discovery_service: FileDiscoveryService | None = None,
    config: dict[str, Any] | None = None,
    dry_run: bool = False,  # ADD THIS
):
    """Initialize GenerateUseCase with optional dry-run mode."""
    # ... existing init code ...
    self._dry_run = dry_run  # ADD THIS
```

#### Step 2: Add dry_run Check in generate_tests()

**File**: `testcraft/application/generate_usecase.py`
**Location**: Beginning of `generate_tests()` method (after line 220)

```python
async def generate_tests(
    self,
    project_path: str | Path,
    target_files: list[str | Path] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main entry point for test generation."""
    project_path = Path(project_path)

    # DRY-RUN EARLY EXIT
    if self._dry_run:
        return await self._execute_dry_run(project_path, target_files)

    # Normal execution continues...
```

#### Step 3: Implement _execute_dry_run()

**File**: `testcraft/application/generate_usecase.py`
**Location**: Add as private method

```python
async def _execute_dry_run(
    self,
    project_path: Path,
    target_files: list[str | Path] | None,
) -> dict[str, Any]:
    """
    Execute dry-run mode: analyze files without generating tests.

    Returns preview of what would be generated without actual execution.
    """
    logger.info("🔍 Dry-run mode: Analyzing files without generation")

    # Discover files (read-only operation)
    discovery_result = self._state_discovery.sync_and_discover(
        project_path, target_files
    )

    # Build preview data
    files_to_analyze = discovery_result["files"]
    preview = []

    for file_path in files_to_analyze:
        test_path = self._infer_test_path(file_path)
        preview.append({
            "source_file": str(file_path),
            "would_generate": str(test_path),
            "status": "dry_run_preview",
            "reason": "Dry-run mode - no actual generation",
        })

    return {
        "success": True,
        "dry_run": True,
        "files_analyzed": len(files_to_analyze),
        "files": preview,
        "message": "Dry run completed - no files written",
        "total_time": 0,
    }

def _infer_test_path(self, source_file: Path) -> Path:
    """Infer where test file would be created."""
    # Simple inference - can be enhanced
    stem = source_file.stem
    if source_file.parent.name == "src":
        test_dir = source_file.parent.parent / "tests"
    else:
        test_dir = source_file.parent.parent / "tests"

    return test_dir / f"test_{stem}.py"
```

#### Step 4: Update CLI to Pass dry_run

**File**: `testcraft/cli/dependency_injection.py`
**Location**: `create_dependency_container()` function

```python
# Around line 100, after refine_adapter creation:
container["generate_usecase"] = GenerateUseCase(
    llm_port=container["llm_adapter"],
    writer_port=container["writer_adapter"],
    coverage_port=container["coverage_adapter"],
    refine_port=container["refine_adapter"],
    context_port=container["context_adapter"],
    parser_port=container["parser_adapter"],
    state_port=container["state_adapter"],
    telemetry_port=container["telemetry_adapter"],
    file_discovery_service=container["file_discovery"],
    config=config.model_dump(),
    dry_run=False,  # ADD THIS - Will be overridden in CLI
)
```

**File**: `testcraft/cli/main.py`
**Location**: `generate()` command (around line 336)

```python
# Get use case from container
generate_usecase = ctx.obj.container["generate_usecase"]

# ✅ UPDATE: Override dry_run from CLI context
generate_usecase._dry_run = ctx.obj.dry_run

# Configure generation parameters...
```

### Validation

#### Test 1: Dry-run with single file
```bash
testcraft generate testcraft/domain/models.py --dry-run
```

**Expected**:
- Shows preview of test file that would be created
- No actual test files created
- Returns success with dry_run=true

#### Test 2: Dry-run with directory
```bash
testcraft generate testcraft/domain/ --dry-run
```

**Expected**:
- Lists all files that would be processed
- Shows test paths that would be created
- No file system modifications

#### Test 3: Normal mode still works
```bash
testcraft generate testcraft/domain/models.py
```

**Expected**:
- Actually generates test file
- Writes to file system
- Returns results with dry_run=false (or absent)

### Rollback

If this breaks:
```bash
git checkout testcraft/application/generate_usecase.py
git checkout testcraft/cli/dependency_injection.py
git checkout testcraft/cli/main.py
```

---

## Task 0.2: Create Rollback Branch

```bash
# Ensure you're on main/master
git checkout main
git pull origin main

# Create backup branch
git checkout -b refactor/pre-orchestrator-consolidation-backup
git push -u origin refactor/pre-orchestrator-consolidation-backup

# Create working branch
git checkout -b refactor/orchestrator-consolidation

# Confirm branches exist
git branch -a | grep refactor
```

**Expected Output**:
```
* refactor/orchestrator-consolidation
  refactor/pre-orchestrator-consolidation-backup
  remotes/origin/refactor/pre-orchestrator-consolidation-backup
```

---

## Task 0.3: Baseline Metrics Collection

### Step 1: Run Full Test Suite

```bash
# Activate environment
source .venv/bin/activate
uv sync --all-extras --all-groups

# Run tests with coverage
pytest tests/ -v \
    --cov=testcraft \
    --cov-report=html \
    --cov-report=json \
    --cov-report=term-missing \
    -o junit_family=xunit2 \
    --junitxml=baseline_results.xml

# Expected: All tests pass
```

### Step 2: Capture Metrics

```bash
# Test count
echo "Test Count:" > baseline_metrics.txt
pytest --collect-only tests/ | tail -1 >> baseline_metrics.txt

# Coverage
echo "\nCoverage:" >> baseline_metrics.txt
coverage json -o baseline_coverage.json
jq '.totals.percent_covered' baseline_coverage.json >> baseline_metrics.txt

# Lines of code
echo "\nLines of Code:" >> baseline_metrics.txt
find testcraft -name "*.py" | xargs wc -l | tail -1 >> baseline_metrics.txt

# File counts
echo "\nPrompt Files:" >> baseline_metrics.txt
grep -r "test_generation\|refinement\|llm_test_generation" testcraft --include="*.py" | wc -l >> baseline_metrics.txt
```

### Step 3: Document Baseline

Create `REFACTOR_BASELINE.md`:

```markdown
# Refactoring Baseline Metrics

## Date: 2025-10-05
## Branch: refactor/pre-orchestrator-consolidation-backup
## Commit: $(git rev-parse HEAD)

### Test Metrics
- Total tests: [FROM pytest --collect-only]
- Passing: [ALL - should be 100%]
- Coverage: [FROM coverage report]%

### Code Metrics
- Total LOC: [FROM wc -l]
- testcraft/ LOC: [FROM find testcraft | wc -l]
- Files with legacy prompts: [FROM grep count]

### Callsites (To Be Removed)
- generate_tests() high-level calls: [TBD from audit]
- refine_content() high-level calls: [TBD from audit]
- Legacy prompt references: [FROM grep]

### Performance Baseline
- Average test generation time: [TBD from benchmarks]
- Memory usage: [TBD]

---

**Purpose**: This baseline allows us to verify:
1. No functionality loss
2. Test coverage maintained
3. Code reduction achieved
4. Performance not degraded
```

### Step 4: Commit Baseline

```bash
git add baseline_*.* REFACTOR_BASELINE.md
git commit -m "chore: Capture baseline metrics before orchestrator refactor

- Full test suite results
- Coverage report
- Lines of code count
- Callsite audit preparation

This baseline enables validation that refactoring maintains
all functionality while reducing code complexity."

git push origin refactor/orchestrator-consolidation
```

---

## Phase 0 Completion Checklist

- [ ] Task 0.1: --dry-run mode fixed and tested
- [ ] Task 0.2: Rollback branch created and pushed
- [ ] Task 0.3: Baseline metrics captured and committed
- [ ] All tests passing
- [ ] Coverage ≥ baseline
- [ ] No regressions introduced

**Only proceed to Phase 1 when ALL items checked**

---

**Next**: [CALLSITE_AUDIT.md](./CALLSITE_AUDIT.md)
