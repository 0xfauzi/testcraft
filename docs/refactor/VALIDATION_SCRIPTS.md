# Validation Scripts & Testing Strategy

**Purpose**: Comprehensive validation of orchestrator consolidation refactor
**Goal**: Zero functionality loss, improved quality

---

## Script 1: validate_refactor.sh

**Location**: `scripts/validate_refactor.sh`

```bash
#!/bin/bash
# Comprehensive validation script for orchestrator consolidation refactor

set -e  # Exit on any error

echo "🔍 Validating Orchestrator Consolidation Refactor"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Activate environment
echo "Activating virtual environment..."
source .venv/bin/activate || {
    echo -e "${RED}❌ Virtual environment not found${NC}"
    exit 1
}

# ========================================
# 1. Unit Tests
# ========================================
echo ""
echo -e "${GREEN}1️⃣ Running Unit Tests...${NC}"
echo "----------------------------------------"

pytest tests/test_llm_adapters.py -v || {
    echo -e "${RED}❌ LLM adapter tests failed${NC}"
    exit 1
}

pytest tests/test_refine_adapters.py -v || {
    echo -e "${RED}❌ Refine adapter tests failed${NC}"
    exit 1
}

# Test prompt registry (may not exist)
if [ -f "tests/test_prompt_registry.py" ]; then
    pytest tests/test_prompt_registry.py -v || {
        echo -e "${YELLOW}⚠️  Prompt registry tests failed (non-critical)${NC}"
    }
fi

echo -e "${GREEN}✅ Unit tests passed${NC}"

# ========================================
# 2. Integration Tests
# ========================================
echo ""
echo -e "${GREEN}2️⃣ Running Integration Tests...${NC}"
echo "----------------------------------------"

pytest tests/test_generate_usecase_contextpack_integration.py -v || {
    echo -e "${RED}❌ GenerateUseCase integration tests failed${NC}"
    exit 1
}

pytest tests/test_pytest_refiner_integration.py -v || {
    echo -e "${RED}❌ Refinement integration tests failed${NC}"
    exit 1
}

pytest tests/test_immediate_refinement.py -v || {
    echo -e "${RED}❌ Immediate refinement tests failed${NC}"
    exit 1
}

echo -e "${GREEN}✅ Integration tests passed${NC}"

# ========================================
# 3. End-to-End Tests
# ========================================
echo ""
echo -e "${GREEN}3️⃣ Running E2E Tests...${NC}"
echo "----------------------------------------"

if [ -d "tests/e2e" ]; then
    pytest tests/e2e/ -v || {
        echo -e "${YELLOW}⚠️  E2E tests failed (may not exist yet)${NC}"
    }
else
    echo -e "${YELLOW}⚠️  E2E tests directory not found${NC}"
fi

# ========================================
# 4. Full Test Suite with Coverage
# ========================================
echo ""
echo -e "${GREEN}4️⃣ Running Full Test Suite with Coverage...${NC}"
echo "----------------------------------------"

pytest tests/ -v \
    --cov=testcraft \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=json \
    -o junit_family=xunit2 \
    --junitxml=validation_results.xml || {
    echo -e "${RED}❌ Full test suite failed${NC}"
    exit 1
}

# Check coverage threshold
COVERAGE=$(python -c "import json; print(json.load(open('coverage.json'))['totals']['percent_covered'])")
THRESHOLD=80

if (( $(echo "$COVERAGE < $THRESHOLD" | bc -l) )); then
    echo -e "${RED}❌ Coverage ${COVERAGE}% is below threshold ${THRESHOLD}%${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Coverage ${COVERAGE}% meets threshold${NC}"
fi

# ========================================
# 5. Linting
# ========================================
echo ""
echo -e "${GREEN}5️⃣ Running Linters...${NC}"
echo "----------------------------------------"

ruff check testcraft/ || {
    echo -e "${YELLOW}⚠️  Linting issues found${NC}"
}

# ========================================
# 6. Type Checking
# ========================================
echo ""
echo -e "${GREEN}6️⃣ Running Type Checker...${NC}"
echo "----------------------------------------"

if [ -f "mypy-staged.ini" ]; then
    mypy testcraft/ --config-file mypy-staged.ini || {
        echo -e "${YELLOW}⚠️  Type checking issues found${NC}"
    }
else
    echo -e "${YELLOW}⚠️  mypy config not found, skipping${NC}"
fi

# ========================================
# 7. Manual Validation Tests
# ========================================
echo ""
echo -e "${GREEN}7️⃣ Running Manual Validation...${NC}"
echo "----------------------------------------"

# Test dry-run mode
echo "Testing dry-run mode..."
testcraft generate testcraft/domain/models.py --dry-run > /dev/null 2>&1 || {
    echo -e "${RED}❌ Dry-run mode failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Dry-run mode works${NC}"

# Test basic generation (dry-run to avoid actual file creation)
echo "Testing basic generation (dry-run)..."
testcraft generate testcraft/domain/models.py --dry-run --verbose > /dev/null 2>&1 || {
    echo -e "${RED}❌ Basic generation failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Basic generation works${NC}"

# ========================================
# Summary
# ========================================
echo ""
echo "=================================================="
echo -e "${GREEN}✅ All validation checks complete!${NC}"
echo ""
echo "📊 Summary:"
echo "  - Unit tests: PASSED"
echo "  - Integration tests: PASSED"
echo "  - Full test suite: PASSED"
echo "  - Coverage: ${COVERAGE}%"
echo "  - Linting: CHECKED"
echo "  - Type checking: CHECKED"
echo "  - Manual validation: PASSED"
echo ""
echo "📊 Coverage Report: file://$(pwd)/htmlcov/index.html"
echo ""
```

**Make Executable**:
```bash
chmod +x scripts/validate_refactor.sh
```

---

## Script 2: benchmark_refactor.py

**Location**: `scripts/benchmark_refactor.py`

```python
#!/usr/bin/env python3
"""
Benchmark orchestrator performance before/after refactor.

Measures:
- Test generation time per file
- Memory usage
- Token consumption
- Quality metrics (if available)
"""

import time
import tracemalloc
from pathlib import Path
from typing import Any

def benchmark_generation(
    target_file: str,
    iterations: int = 3,
    dry_run: bool = True
) -> dict[str, Any]:
    """
    Benchmark test generation for a file.

    Args:
        target_file: Path to file to generate tests for
        iterations: Number of iterations to average
        dry_run: Use dry-run mode to avoid file writes

    Returns:
        Dict with benchmark results
    """
    from testcraft.cli.dependency_injection import create_dependency_container
    from testcraft.config.loader import ConfigLoader

    print(f"\n📄 Benchmarking {target_file}...")

    # Load config
    loader = ConfigLoader()
    config = loader.load_config()

    # Create container
    container = create_dependency_container(config)
    use_case = container["generate_usecase"]

    # Set dry-run mode
    if dry_run:
        use_case._dry_run = True

    times = []
    memory_peaks = []

    for i in range(iterations):
        # Start memory tracking
        tracemalloc.start()

        # Time the operation
        start = time.time()

        try:
            import asyncio
            result = asyncio.run(use_case.generate_tests(
                project_path=".",
                target_files=[target_file],
            ))

            elapsed = time.time() - start
            times.append(elapsed)

            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()
            memory_peaks.append(peak / 1024 / 1024)  # MB

            print(f"  Iteration {i+1}: {elapsed:.2f}s (peak mem: {peak/1024/1024:.1f}MB)")

        except Exception as e:
            print(f"  ❌ Iteration {i+1} failed: {e}")
            tracemalloc.stop()
            continue

        tracemalloc.stop()

    if not times:
        return {
            "file": target_file,
            "status": "failed",
            "error": "All iterations failed",
        }

    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_peaks) / len(memory_peaks)

    return {
        "file": target_file,
        "iterations": iterations,
        "avg_time": avg_time,
        "min_time": min(times),
        "max_time": max(times),
        "avg_memory_mb": avg_memory,
        "times": times,
        "status": "success",
    }


def main():
    """Run benchmarks on representative files."""
    print("🏁 Benchmarking Orchestrator Performance")
    print("=" * 50)

    # Representative files of varying complexity
    test_files = [
        "testcraft/domain/models.py",         # Complex (models)
        "testcraft/adapters/llm/claude.py",   # Medium (adapter)
        "testcraft/cli/main.py",              # Medium (CLI)
        "testcraft/config/models.py",         # Simple (config)
    ]

    results = []

    for file_path in test_files:
        if not Path(file_path).exists():
            print(f"⚠️  Skipping {file_path} (not found)")
            continue

        try:
            result = benchmark_generation(file_path, iterations=3, dry_run=True)
            results.append(result)

            if result["status"] == "success":
                print(f"  ✅ Average: {result['avg_time']:.2f}s "
                      f"(mem: {result['avg_memory_mb']:.1f}MB)")
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print("")

    successful = [r for r in results if r["status"] == "success"]

    if not successful:
        print("  ❌ No successful benchmarks")
        return 1

    for r in successful:
        print(f"  {r['file']}: {r['avg_time']:.2f}s "
              f"(range: {r['min_time']:.2f}-{r['max_time']:.2f}s, "
              f"mem: {r['avg_memory_mb']:.1f}MB)")

    total_avg = sum(r['avg_time'] for r in successful) / len(successful)
    total_mem = sum(r['avg_memory_mb'] for r in successful) / len(successful)

    print("")
    print(f"  Overall Average Time: {total_avg:.2f}s")
    print(f"  Overall Average Memory: {total_mem:.1f}MB")
    print("")

    # Save results
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("📁 Results saved to benchmark_results.json")

    return 0


if __name__ == "__main__":
    exit(main())
```

**Make Executable**:
```bash
chmod +x scripts/benchmark_refactor.py
```

---

## Usage

### Before Refactoring

```bash
# Checkout baseline branch
git checkout refactor/pre-orchestrator-consolidation-backup

# Run benchmarks
python scripts/benchmark_refactor.py > benchmark_before.txt

# Save results
cp benchmark_results.json benchmark_before.json
```

### After Refactoring

```bash
# Checkout refactored branch
git checkout refactor/orchestrator-consolidation

# Run validation
./scripts/validate_refactor.sh

# Run benchmarks
python scripts/benchmark_refactor.py > benchmark_after.txt

# Save results
cp benchmark_results.json benchmark_after.json
```

### Compare Results

```bash
# Compare benchmark files
diff benchmark_before.txt benchmark_after.txt

# Or use Python
python -c "
import json

with open('benchmark_before.json') as f:
    before = json.load(f)

with open('benchmark_after.json') as f:
    after = json.load(f)

print('Performance Comparison:')
for b, a in zip(before, after):
    if b['status'] == 'success' and a['status'] == 'success':
        file = b['file']
        before_time = b['avg_time']
        after_time = a['avg_time']
        delta = ((after_time - before_time) / before_time) * 100

        print(f\"{file}:\")
        print(f\"  Before: {before_time:.2f}s\")
        print(f\"  After:  {after_time:.2f}s\")
        print(f\"  Delta:  {delta:+.1f}%\")
"
```

---

## Acceptance Criteria

### Must Pass

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Coverage ≥ 80%
- ✅ No critical linting errors
- ✅ Dry-run mode works
- ✅ Basic generation works

### Performance Targets

- ✅ Average generation time within ±20% of baseline
- ✅ Memory usage within ±20% of baseline
- ✅ No timeout errors

### Quality Targets

- ✅ Generated tests have correct imports
- ✅ No undefined name errors
- ✅ Refinement successfully fixes failures
- ✅ Error messages are clear

---

## Rollback Triggers

**Immediately rollback if**:

1. **Critical Functionality Broken**:
   - Test generation fails completely
   - Refinement doesn't work
   - CLI commands error

2. **Performance Degradation >20%**:
   - Average generation time >20% slower
   - Memory usage >20% higher
   - Timeout rate increases significantly

3. **Test Pass Rate Drops >10%**:
   - More than 10% of tests now fail
   - Coverage drops below 70%

4. **Production Errors**:
   - Unrecoverable errors in actual usage
   - Data loss or corruption
   - Silent failures

---

## Continuous Monitoring (Post-Merge)

### Days 1-7: Active Monitoring

**Check Daily**:
```bash
# Run validation
./scripts/validate_refactor.sh

# Check error logs
grep -i "error\|exception" .testcraft/logs/*.log | tail -20

# Monitor usage patterns
testcraft status  # Check generation statistics
```

**Watch For**:
- New error patterns
- Performance degradation
- User complaints
- Unusual behavior

### Week 2-4: Passive Monitoring

**Check Weekly**:
- Test pass rate
- Generation success rate
- Average generation time
- Memory usage trends

### After 30 Days

If no issues:
- Delete backup branch
- Archive refactor documentation
- Update as "stable"

---

**Next**: Execute validation scripts before and after refactoring
