# Manual Fix Guidance Feature

## Overview

The Manual Fix Guidance feature provides intelligent assistance when test generation refinement exhausts all retries and suspects a production bug. Instead of silently failing, TestCraft can generate:

1. **Deliberately Failing Test**: A pytest module that will pass only after the underlying code bug is fixed
2. **Comprehensive BUG NOTE**: Markdown documentation with root cause analysis, reproduction steps, and fix suggestions

## Architecture

### Components

- **Domain Models** (`testcraft/domain/manual_fix.py`):
  - `ManualFixRequest`: Input payload (target, trace excerpt, notes)
  - `ManualFixRecommendation`: Structured output (test_code, bug_note_markdown, metadata, hash)
  - `ManualFixContextAttachment`: LLM/prompt provenance metadata

- **Ports** (`testcraft/ports/manual_fix_port.py`):
  - `ManualFixGuidancePort`: Interface for requesting guidance
  - `ManualFixPresenterPort`: Interface for CLI/TUI acceptance

- **Use Case** (`testcraft/application/manual_fix_usecase.py`):
  - Orchestrates context building, LLM invocation, parsing, acceptance, and persistence
  - Emits comprehensive telemetry spans/metrics

- **Integration Points**:
  - `GenerateUseCase._run_manual_fix_after_failure`: Post-refinement trigger
  - CLI `manual-fix` command: Standalone invocation
  - CLI `generate --manual-fix-on-fail`: Auto-trigger mode

## Configuration

### TOML Configuration

```toml
[manual_fix]
enable = true                                # Enable manual-fix guidance features
on_fail = false                              # Auto-trigger after refinement failure (opt-in)
auto_accept = false                          # Auto-accept recommendations without prompt
output_dir = ".testcraft/manual_fixes"       # Directory for Markdown artifacts
model = ""                                   # Override model ID (optional)
max_tokens = 8000                            # Max tokens for guidance
temperature = 0.1                            # LLM temperature
prompt_version = ""                          # Prompt version override (optional)
```

### CLI Flags

```bash
# Standalone manual-fix command
testcraft manual-fix . \
  --target-file src/module.py \
  --target-object MyClass.method \
  --trace-excerpt "AssertionError: expected 10, got 42" \
  --notes "Repro steps" \
  --auto-accept-fixes

# Enable in generate workflow
testcraft generate . --manual-fix-on-fail --auto-accept-fixes
```

## Workflow

### Automatic Trigger (Opt-In)

When `manual_fix.on_fail = true`:

1. Test refinement exhausts max retries
2. `GenerateUseCase` detects failure
3. Builds `ManualFixRequest` with trace excerpt and context
4. Invokes `ManualFixGuidanceUseCase`
5. LLM generates failing test + BUG NOTE
6. Presents recommendation (if `auto_accept = false`)
7. On acceptance, writes Markdown artifact to `.testcraft/manual_fixes/<target>_<hash>.md`

### Manual Invocation

```bash
testcraft manual-fix . \
  --target-file path/to/broken_module.py \
  --target-object problematic_function \
  --trace-excerpt "Full traceback here..." \
  --auto-accept-fixes
```

## Artifact Structure

Markdown files are persisted in `<output_dir>/<target>_<hash>.md`:

```markdown
# Manual Fix Recommendation
- Target: /repo/src/module.py::func
- Hash: abc123def4567890
- Created: 2025-01-15T10:30:00Z
- Canonical Import: from my_pkg import module as _under_test
- Prompt Hash: fedcba9876543210
- Model: gpt-4

## Failing Test (will pass after bug is fixed)

\```python
def test_bug_func_returns_wrong_value():
    result = func()
    assert result == 10  # Expected 10, got 42
\```

## BUG NOTE

# BUG: func returns hardcoded constant

**Summary**: Function returns 42 instead of calculated value.

**Expected**: 10
**Actual**: 42

**Root Cause**: Line 5 of module.py - return statement uses constant

**Suggested Fix**: Replace `return 42` with proper calculation
```

## Deduplication

Artifacts are deduplicated by content hash (test_code + bug_note_markdown + prompt_hash). If the exact same recommendation is generated twice, the existing artifact is reused.

## Telemetry

Emits the following telemetry:

- **Spans**:
  - `manual_fix_guidance` (parent span)
  - `manual_fix_llm_call` (orchestrator stage)
  - `manual_fix_parse` (response parsing)
  - `manual_fix_present` (acceptance flow)
  - `manual_fix_persist` (artifact writing)

- **Metrics**:
  - `manual_fix_accepted` (count)
  - `manual_fix_files_touched` (count)

- **Attributes**:
  - `target_file`, `target_object`
  - `recommendation_hash`
  - `test_code_extracted`, `bug_note_extracted`
  - `auto_accept`, `accepted`
  - `artifact_path`, `artifact_written`

## Testing

Comprehensive test coverage in:

- `tests/test_manual_fix_domain.py`: Domain model unit tests (8 tests)
- `tests/test_manual_fix_usecase.py`: Use case unit tests (7 tests)
- `tests/integration/test_manual_fix_integration.py`: End-to-end integration test

All tests verify:
- Hash stability and deduplication
- Fenced block extraction (python + markdown)
- Auto-accept vs user acceptance flows
- Artifact persistence and structure
- Telemetry emission

## Future Enhancements (Out of Scope for v1)

- Textual TUI review screen with interactive editing
- LLM retry/backoff on malformed responses
- Model/token override from config
- Cross-reference to failing test annotations
