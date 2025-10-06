# Orchestrator Consolidation Refactor - Documentation Index

**Status**: COMPREHENSIVE PLAN READY FOR EXECUTION
**Created**: 2025-10-05
**Purpose**: Consolidate all test generation to use LLMOrchestrator exclusively

---

## 📚 Documentation Structure

### 🎯 Start Here

1. **[ENHANCED_REFACTOR_PLAN.md](./ENHANCED_REFACTOR_PLAN.md)** - **READ THIS FIRST**
   - Executive summary with critical corrections
   - Complete phase-by-phase breakdown
   - Success criteria and rollback plan
   - 📊 Main entry point for refactoring effort

### 🏗️ Architecture & Decisions

2. **[ADR-001-orchestrator-consolidation.md](./ADR-001-orchestrator-consolidation.md)**
   - Architecture Decision Record
   - Rationale for orchestrator-only approach
   - Alternatives considered and rejected
   - Consequences and mitigation strategies

### 📋 Detailed Phase Documentation

3. **[PHASE_0_PREFLIGHT.md](./PHASE_0_PREFLIGHT.md)** - **MUST COMPLETE FIRST**
   - Fix --dry-run mode (CRITICAL)
   - Create rollback branches
   - Capture baseline metrics
   - Pre-flight checklist

4. **[PHASE_1_IMPACT_ANALYSIS.md](./PHASE_1_IMPACT_ANALYSIS.md)**
   - Complete callsite audit
   - Architecture decision record
   - Dependency analysis
   - Risk assessment
   - Baseline performance metrics

5. **[PHASE_2_REFINE_ADAPTER.md](./PHASE_2_REFINE_ADAPTER.md)**
   - Complete RefineAdapter refactoring guide
   - Add ParserPort dependency (BREAKING CHANGE)
   - Implement orchestrator integration
   - Context pack building helpers

6. **[PHASE_3_REMOVE_FALLBACK.md](./PHASE_3_REMOVE_FALLBACK.md)**
   - Remove legacy fallback path
   - Implement fail-fast with diagnostics
   - Enhanced error messages
   - Update CLI error handling

7. **[PHASE_4_PROMPT_CLEANUP.md](./PHASE_4_PROMPT_CLEANUP.md)**
   - Remove legacy prompts (~330 lines)
   - Update prompt registry
   - Clean up schemas
   - Update module docstring

8. **Phases 5-9** - See [ENHANCED_REFACTOR_PLAN.md](./ENHANCED_REFACTOR_PLAN.md)
   - Phase 5: Test Updates
   - Phase 6: Documentation Updates
   - Phase 7: Validation & Rollout
   - Phase 8: Git Workflow & Deployment
   - Phase 9: Post-Merge Validation

### 📊 Analysis & Auditing

9. **[CALLSITE_AUDIT.md](./CALLSITE_AUDIT.md)**
   - Complete audit of all legacy prompt usage
   - Categorized by priority and impact
   - Specific line numbers for all changes
   - Action list for each file

10. **[REMOVED_PROMPTS.md](./REMOVED_PROMPTS.md)**
    - Complete catalog of removed prompts
    - Migration examples for each prompt type
    - Recovery instructions if needed
    - External usage warnings

### 🔄 Migration & Templates

11. **[MIGRATION_ORCHESTRATOR.md](./MIGRATION_ORCHESTRATOR.md)**
    - Complete migration guide for external code
    - Code examples for all scenarios
    - Breaking changes documentation
    - FAQ and troubleshooting

12. **[REFACTOR_BASELINE.md](./REFACTOR_BASELINE.md)**
    - Template for capturing baseline metrics
    - Fill in during Phase 0, Task 0.3
    - Validation criteria checklist

### ✅ Validation & Testing

13. **[VALIDATION_SCRIPTS.md](./VALIDATION_SCRIPTS.md)**
    - Complete validation script (`validate_refactor.sh`)
    - Performance benchmarking script (`benchmark_refactor.py`)
    - Acceptance criteria
    - Rollback triggers
    - Post-merge monitoring plan

### 📋 Master Index

14. **[COMPLETE_FILE_MANIFEST.md](./COMPLETE_FILE_MANIFEST.md)**
    - Master index of all documentation
    - Validation that all files are present
    - Documentation statistics
    - Sign-off checklist

---

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Ensure clean working directory
git status

# Ensure all tests pass
pytest tests/ -v

# Ensure virtual environment active
source .venv/bin/activate
uv sync --all-extras --all-groups
```

### Execution Order

```bash
# 1. Create rollback branch (PHASE 0 - Task 0.2)
git checkout -b refactor/pre-orchestrator-consolidation-backup
git push origin refactor/pre-orchestrator-consolidation-backup
git checkout -b refactor/orchestrator-consolidation

# 2. Capture baseline metrics (PHASE 0 - Task 0.3)
pytest tests/ --cov=testcraft --cov-report=json --junitxml=baseline_results.xml
# Document in REFACTOR_BASELINE.md

# 3. Fix --dry-run mode (PHASE 0 - Task 0.1)
# Follow PHASE_0_PREFLIGHT.md step-by-step

# 4. Validate Phase 0
pytest tests/ -v  # All tests must pass
testcraft generate testcraft/domain/models.py --dry-run  # Must work

# 5. Proceed with remaining phases
# Follow ENHANCED_REFACTOR_PLAN.md sequentially
```

---

## ⚠️ Critical Corrections from Original Plan

### Major Misunderstanding Identified

**WRONG ASSUMPTION** (from original plan):
> "Remove `generate_tests()`, `refine_content()` from LLM adapters"

**CORRECT UNDERSTANDING**:
- ✅ **KEEP** `generate_tests()` and `refine_content()` methods in LLMPort
- ✅ These are **LOW-LEVEL methods** used BY the orchestrator
- ❌ **REMOVE** high-level usage patterns with simple prompts
- ❌ **REMOVE** legacy prompt definitions

**Why This Matters**:
```python
# The orchestrator USES these methods:
# testcraft/application/generation/services/llm_orchestrator.py:360
raw_response = await self._llm_port.generate_tests(
    code_content=plan_prompt,  # Orchestrator-generated prompt
    system_prompt=system_prompt,
    ...
)
```

Removing these methods would **break the orchestrator itself**.

---

## 📊 Impact Summary

### Files to Modify

| Category | Count | Impact | Actions |
|----------|-------|--------|---------|
| Critical Production | 2 | HIGH | Refactor fallback, add orchestrator |
| Prompt Registry | 1 | HIGH | Delete 10 prompt methods |
| Tests | 5 | MEDIUM | Update mocking strategy |
| Scripts | 2 | LOW | Update validation, benchmarking |
| Documentation | 4 | LOW | Update architecture docs |
| **TOTAL** | **14** | - | **~560 lines removed** |

### Breaking Changes

1. **RefineAdapter Constructor**: Now requires `ParserPort` parameter
2. **Legacy Prompts**: No longer available (raises `KeyError`)
3. **Generate Fallback**: Removed (fails fast instead)

### Expected Outcomes

- ✅ **Code Reduction**: ~1,000 lines removed (net)
- ✅ **Quality Improvement**: Symbol resolution prevents errors
- ✅ **Maintainability**: Single source of truth
- ✅ **Observability**: 4-stage pipeline for debugging

---

## ✅ Validation Checklist

Before considering refactor complete:

### Phase 0 - Pre-Flight ✅
- [x] --dry-run mode fixed and tested
- [x] Rollback branch created and pushed
- [x] Baseline metrics captured
- [x] LLMRouter implemented and functional
- [x] ContextPackBuilder bug fixed
- [x] All tests passing before changes (no tests currently exist)

### Phase 1 - Impact Analysis ✅
- [x] Complete callsite audit
- [x] Architecture Decision Record
- [x] Dependency analysis
- [x] Risk assessment

### Phase 2 - RefineAdapter ✅
- [x] RefineAdapter refactored with orchestrator
- [x] ParserPort dependency added
- [x] Lazy orchestrator initialization
- [x] Legacy prompt usage removed

### Phase 3 - Remove Fallback ✅
- [x] Legacy fallback removed from GenerateUseCase
- [x] Fail-fast error handling added
- [x] Diagnostic helper implemented

### Phase 4 - Prompt Cleanup ✅
- [x] Legacy prompts deleted from registry (331 lines)
- [x] Module docstring updated
- [x] Schemas verified

### Phase 5 - Validation Tests ✅
- [x] All tests created (18 tests)
- [x] All integration tests passing (18/18)

### Documentation
- [ ] Architecture docs updated
- [ ] Migration guide created
- [ ] ADR documented
- [ ] Removed prompts cataloged

### Validation
- [ ] `./scripts/validate_refactor.sh` passes
- [ ] Benchmarks within ±20% of baseline
- [ ] Coverage ≥80%
- [ ] No critical linting errors
- [ ] Manual testing complete

### Deployment
- [ ] All commits created with clear messages
- [ ] Pull request created with detailed description
- [ ] Code review complete
- [ ] Merged to main
- [ ] Post-merge smoke tests pass
- [ ] Monitoring in place (7 days)

---

## 🚨 Emergency Rollback

If something goes catastrophically wrong:

```bash
# Option 1: Revert the merge
git revert <merge-commit-sha> -m 1
git push origin main

# Option 2: Restore backup branch
git checkout refactor/pre-orchestrator-consolidation-backup
git checkout -b hotfix/restore-legacy-prompts
git push origin hotfix/restore-legacy-prompts

# Option 3: Cherry-pick specific fixes
git cherry-pick <good-commit-sha>
```

**Rollback Criteria**:
- Critical functionality broken
- Performance degradation >20%
- Test pass rate drops >10%
- Unrecoverable production errors

---

## 📞 Questions & Support

### Common Questions

**Q: Why consolidate to orchestrator-only?**
A: See [ADR-001](./ADR-001-orchestrator-consolidation.md) - Section "Rationale"

**Q: What if orchestrator is slower?**
A: Benchmark first. Optimize if needed. Quality improvement justifies modest performance cost.

**Q: Can I keep legacy as fallback?**
A: No. See ADR-001 "Alternatives Considered" - Violates fail-fast principle.

**Q: Where do I start?**
A: [ENHANCED_REFACTOR_PLAN.md](./ENHANCED_REFACTOR_PLAN.md) → [PHASE_0_PREFLIGHT.md](./PHASE_0_PREFLIGHT.md)

**Q: What if I find a bug in the plan?**
A: Update the relevant document and commit the fix. Plans are living documents.

---

## 📈 Progress Tracking

### Completion Estimate

- **Phase 0**: 2-4 hours (pre-flight fixes)
- **Phases 1-4**: 6-8 hours (implementation)
- **Phase 5**: 3-4 hours (test updates)
- **Phase 6**: 2-3 hours (documentation)
- **Phases 7-8**: 2-3 hours (validation + deployment)
- **Phase 9**: 7 days (monitoring)

**Total Active Work**: ~20-25 hours
**Total Calendar Time**: ~10 days (including monitoring)

---

## 📝 Document Maintenance

These documents are **living documentation**. Update them as:

- Implementation reveals edge cases
- Tests uncover issues
- Better approaches are discovered
- Post-merge issues arise

**Commit changes with**:
```bash
git add docs/refactor/
git commit -m "docs(refactor): Update [DOCUMENT] with [FINDING]"
```

---

## ✨ Success Metrics

The refactor is **SUCCESSFUL** when:

1. ✅ All generation uses `orchestrator.plan_and_generate()`
2. ✅ All refinement uses `orchestrator.refine_stage()`
3. ✅ No legacy prompt paths exist
4. ✅ All tests pass (≥80% coverage)
5. ✅ Performance within acceptable range
6. ✅ No critical issues for 7 days post-merge
7. ✅ Team can maintain single codebase easily

---

**Prepared By**: AI Assistant (Claude Sonnet 4.5)
**Reviewed By**: [Pending]
**Approved By**: [Pending]
**Status**: READY FOR EXECUTION
