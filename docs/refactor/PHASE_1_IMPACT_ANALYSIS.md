# Phase 1: Detailed Impact Analysis

**Priority**: HIGH
**Must Complete**: Before implementing any changes

---

## Overview

This phase performs a comprehensive analysis of all code that will be affected by the orchestrator consolidation. The goal is to document every callsite, understand dependencies, and create a detailed action plan.

---

## Task 1.1: Complete Callsite Audit

### Objective
Identify and document every location where legacy prompts or high-level LLM methods are used.

### Execution

#### Step 1: Search for Legacy Prompt References

```bash
# Find all legacy prompt references
grep -rn "test_generation\|refinement" testcraft --include="*.py" > legacy_prompts.txt

# Find LLM adapter prompt references
grep -rn "llm_test_generation\|llm_code_analysis\|llm_content_refinement" testcraft --include="*.py" > llm_adapter_prompts.txt

# Find high-level method calls
grep -rn "\.generate_tests\(" testcraft --include="*.py" > generate_tests_calls.txt
grep -rn "\.refine_content\(" testcraft --include="*.py" > refine_content_calls.txt
```

#### Step 2: Categorize Findings

Organize findings into:
- **HIGH PRIORITY**: Production code using legacy patterns
- **MEDIUM PRIORITY**: Prompt registry definitions
- **LOW PRIORITY**: Tests and documentation

#### Step 3: Document Each Finding

For each finding, document:
- File path
- Line number(s)
- Type of usage
- Impact (HIGH/MEDIUM/LOW)
- Required action

**Output**: [CALLSITE_AUDIT.md](./CALLSITE_AUDIT.md)

---

## Task 1.2: Architecture Decision Document

### Objective
Create formal Architecture Decision Record (ADR) documenting the consolidation decision.

### Execution

#### Step 1: Draft ADR Structure

Create `docs/refactor/ADR-001-orchestrator-consolidation.md` with:
- Status (Proposed/Accepted/Deprecated)
- Context (why this decision)
- Decision (what we're doing)
- Consequences (positive and negative)
- Alternatives considered

#### Step 2: Document Rationale

Include:
- **Problem Statement**: Three parallel codepaths cause issues
- **Evidence**: Specific examples from codebase
- **Solution**: Orchestrator-only architecture
- **Benefits**: Quality, maintainability, observability
- **Risks**: Breaking changes, no fallback

#### Step 3: Document Mitigation Strategies

For each risk, document:
- **Risk**: What could go wrong
- **Likelihood**: HIGH/MEDIUM/LOW
- **Impact**: HIGH/MEDIUM/LOW
- **Mitigation**: How to prevent/handle

**Example**:
```markdown
### Risk: Breaking Changes

**Likelihood**: HIGH
**Impact**: MEDIUM
**Mitigation**:
- Comprehensive migration guide
- Update all internal callers first
- Keep rollback branch for 30 days
- Clear documentation
```

**Output**: [ADR-001-orchestrator-consolidation.md](./ADR-001-orchestrator-consolidation.md)

---

## Task 1.3: Dependency Analysis

### Objective
Map all dependencies between components to understand refactoring order.

### Execution

#### Step 1: Create Dependency Graph

```python
# deps_analysis.py
import ast
from pathlib import Path
from collections import defaultdict

def analyze_dependencies(root_dir: Path) -> dict:
    """Analyze Python file dependencies."""
    deps = defaultdict(list)

    for py_file in root_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        deps[str(py_file)].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        deps[str(py_file)].append(node.module)
        except:
            pass

    return deps

# Run analysis
deps = analyze_dependencies(Path("testcraft"))
# Save to deps_graph.json
```

#### Step 2: Identify Critical Dependencies

Key components to track:
- `LLMOrchestrator` dependencies
- `RefineAdapter` dependencies
- `GenerateUseCase` dependencies
- Prompt registry usage

#### Step 3: Determine Refactoring Order

Based on dependencies:
1. **First**: Components with no dependencies (prompt registry)
2. **Second**: Components with single dependencies (RefineAdapter)
3. **Last**: High-level orchestrators (GenerateUseCase)

**Output**: Create `docs/refactor/DEPENDENCY_ANALYSIS.md`

---

## Task 1.4: Risk Assessment

### Objective
Identify and quantify all risks associated with the refactoring.

### Execution

#### Risk Matrix

| Risk | Likelihood | Impact | Priority | Mitigation |
|------|-----------|--------|----------|------------|
| Breaking external integrations | HIGH | HIGH | P0 | Migration guide, clear docs |
| Performance degradation | MEDIUM | MEDIUM | P1 | Benchmark before/after |
| Test failures | MEDIUM | HIGH | P0 | Comprehensive testing |
| Incomplete migration | LOW | HIGH | P0 | Checklist, validation |
| Rollback needed | MEDIUM | MEDIUM | P1 | Backup branch, rollback plan |

#### Detailed Risk Analysis

**Risk 1: Breaking External Integrations**
- **Description**: External code calling removed prompts/methods
- **Detection**: N/A (external code unknown)
- **Prevention**: Clear breaking change warnings
- **Response**: Migration guide, support channel

**Risk 2: Performance Degradation**
- **Description**: Orchestrator slower than simple prompts
- **Detection**: Benchmark before/after (Task 1.5)
- **Prevention**: Performance testing in Phase 7
- **Response**: Optimize or rollback if >20% slower

**Risk 3: Silent Failures**
- **Description**: Tests pass but functionality broken
- **Detection**: Integration tests, manual validation
- **Prevention**: Comprehensive test coverage
- **Response**: Fix issues before merge

**Output**: Add to ADR-001 "Consequences" section

---

## Task 1.5: Baseline Performance Metrics

### Objective
Capture current performance metrics for comparison after refactoring.

### Execution

#### Step 1: Measure Test Generation Time

```bash
# Generate baseline performance data
time testcraft generate testcraft/domain/models.py --dry-run
time testcraft generate testcraft/adapters/llm/claude.py --dry-run
time testcraft generate testcraft/application/generate_usecase.py --dry-run

# Save results
echo "Baseline Performance Metrics" > baseline_performance.txt
# ... append results
```

#### Step 2: Measure Memory Usage

```python
# memory_baseline.py
import tracemalloc
from testcraft.cli.dependency_injection import create_generate_usecase

tracemalloc.start()

use_case = create_generate_usecase(config={})
# Run generation
result = use_case.execute(...)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

#### Step 3: Measure Token Usage

If telemetry is available:
```bash
# Check current token usage
grep "tokens_used" .testcraft/logs/*.log | awk '{sum+=$NF} END {print sum}'
```

#### Step 4: Document Baselines

Create `docs/refactor/BASELINE_METRICS.md`:
```markdown
# Baseline Metrics

## Date: 2025-10-05
## Branch: refactor/pre-orchestrator-consolidation-backup
## Commit: $(git rev-parse HEAD)

### Performance
- Test generation (models.py): 2.3s
- Test generation (claude.py): 3.1s
- Test generation (generate_usecase.py): 4.2s
- Average: 3.2s

### Memory
- Current usage: 45.2 MB
- Peak usage: 67.8 MB

### Token Usage (if available)
- Average per file: 3,500 tokens
- Total for 10 files: 35,000 tokens

### Test Metrics
- Total tests: 342
- Passing: 342 (100%)
- Coverage: 83.5%
```

---

## Task 1.6: Test Coverage Analysis

### Objective
Identify areas with insufficient test coverage that need attention.

### Execution

#### Step 1: Generate Coverage Report

```bash
pytest tests/ --cov=testcraft --cov-report=html --cov-report=json

# Identify low-coverage files
python -c "
import json
with open('coverage.json') as f:
    cov = json.load(f)
    for file, data in cov['files'].items():
        if data['summary']['percent_covered'] < 80:
            print(f'{file}: {data[\"summary\"][\"percent_covered\"]:.1f}%')
"
```

#### Step 2: Prioritize Test Additions

Focus on:
- `RefineAdapter` (will be heavily modified)
- `GenerateUseCase` fallback path (being removed)
- Prompt registry (being cleaned up)

#### Step 3: Add Missing Tests

**Before refactoring**, add tests for:
```python
# tests/test_generate_usecase_fallback.py
def test_fallback_path_is_used_when_context_unavailable():
    """Ensure fallback is currently working (before removal)."""
    # This test will be updated/removed in Phase 3
    pass

# tests/test_refine_adapter_current.py
def test_refine_uses_simple_prompts():
    """Ensure current refinement works (before orchestrator)."""
    # This test will be updated in Phase 2
    pass
```

**Goal**: Coverage ≥80% before starting refactor

---

## Task 1.7: External Integration Check

### Objective
Identify any external code that might depend on removed functionality.

### Execution

#### Step 1: Search for Public API Usage

```bash
# Check if testcraft is imported by other projects
# (if applicable)

# Check GitHub/GitLab for known integrations
# Search internal docs for integration examples
```

#### Step 2: Document Known Integrations

Create `docs/refactor/EXTERNAL_INTEGRATIONS.md`:
```markdown
# Known External Integrations

## Internal Projects
1. **Project A**: Uses GenerateUseCase directly
   - Contact: team-a@company.com
   - Action: Provide migration guide before refactor

2. **Project B**: CLI-only usage
   - Contact: team-b@company.com
   - Action: No changes needed (CLI unchanged)

## External Users
- Unknown external users may exist
- Mitigation: Clear breaking change warnings in release notes
```

#### Step 3: Communication Plan

If external integrations exist:
1. Send advance notice (2 weeks before)
2. Provide migration examples
3. Offer support channel
4. Keep rollback branch available for 30 days

---

## Phase 1 Completion Checklist

- [ ] Task 1.1: Callsite audit complete (CALLSITE_AUDIT.md)
- [ ] Task 1.2: ADR created (ADR-001-orchestrator-consolidation.md)
- [ ] Task 1.3: Dependency analysis complete
- [ ] Task 1.4: Risk assessment documented
- [ ] Task 1.5: Baseline performance metrics captured
- [ ] Task 1.6: Test coverage ≥80%
- [ ] Task 1.7: External integrations identified and notified

---

## Deliverables

1. ✅ [CALLSITE_AUDIT.md](./CALLSITE_AUDIT.md) - Complete callsite documentation
2. ✅ [ADR-001-orchestrator-consolidation.md](./ADR-001-orchestrator-consolidation.md) - Architecture decision
3. 📄 `DEPENDENCY_ANALYSIS.md` - Component dependency graph
4. 📄 `BASELINE_METRICS.md` - Performance/coverage baselines
5. 📄 `EXTERNAL_INTEGRATIONS.md` - Known integration points

---

## Validation

Before proceeding to Phase 2:
1. ✅ All deliverables created
2. ✅ All findings documented with line numbers
3. ✅ All risks identified and mitigated
4. ✅ Baseline metrics captured
5. ✅ Test coverage ≥80%

**Only proceed to Phase 2 when ALL items checked**

---

**Next**: [PHASE_2_REFINE_ADAPTER.md](./PHASE_2_REFINE_ADAPTER.md)
