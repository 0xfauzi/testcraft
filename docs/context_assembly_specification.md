# Repository-Aware Python Test Generation (Spec)

## Goals

* Generate **repository-aware**, **compilable**, **deterministic** `pytest` tests with **correct imports** for the focal code.
* Reduce hallucinations via **precise context injection** (on-demand symbol definitions) and **property-based retrieval** (GIVEN/WHEN/THEN from related code & existing tests). &#x20;
* Use **execution feedback** (tracebacks, coverage, mutation) to repair/strengthen tests.
* If failures indicate a **real product bug**, emit a **deliberately failing** test + a **BUG NOTE**; the test passes only after the code is fixed.

Pipeline: **plan → generate → refine → ask for manual fix steps**

---

## Architecture

### 1) Analyzers (offline)

* **RepoLayoutDetector**
  Detect packaging and module roots:

  * `pyproject.toml` (`[project]`, `[tool.setuptools]`, Poetry, Hatch)
  * `setup.cfg` / `setup.py` (`packages`, `package_dir`)
  * Presence of `src/` layout, top-level packages (dirs with `__init__.py`), monorepo subpackages
  * Fallback heuristics: infer package roots by walking parents with `__init__.py`; otherwise treat parent dir as import root.

* **ImportResolver (Canonical Import Mapper)**
  For a given file (target), compute its **canonical absolute import path** (e.g., `my_pkg.sub.module`) and the **bootstrap** required (extra sys.path entries). Emits:

  * `target_import`: e.g., `from my_pkg.sub import module as _under_test` or `import my_pkg.sub.module as _under_test`
  * `sys_path_roots`: list of directories to add (e.g., `<repo>/src`, `<repo>` for flat layout)
  * `needs_bootstrap`: bool
  * `bootstrap_conftest`: minimal `tests/conftest.py` to add roots to `sys.path`

* **MetainfoBuilder**
  Parse AST (and/or Jedi/Pyright) → tables: packages, modules, classes, functions, signatures, docstrings, constants, fixtures, imports.

* **TestCaseIndexer**
  Build **test bundles**: each test paired with the fixtures/imports/mocks/assertions it actually relies on (not entire files).

* **PropertyAnalyzer (APT-style)**
  Identify **GIVEN/WHEN/THEN** property relationships for the focal method across intra-class and repo-level, rank: intra-class complete > intra-class G/W/T > repo-level complete > repo-level G/W/T; extract minimal G/W/T snippets + related test bundles. &#x20;

### 2) Context Assembler (per target)

Build a **ContextPack**:

* Focal code (source + signature + docstring)
* **Resolved defs** for identifiers the model may call (signatures/docstrings; minimal bodies only when essential) — injected **precisely on demand** during planning/generation to avoid bloat. &#x20;
* Property context: ranked methods, G/W/T snippets, selected test bundles.&#x20;
* Conventions (pytest, determinism, IO policy)
* **ImportMap** (from ImportResolver): `target_import`, `sys_path_roots`, `needs_bootstrap`, `bootstrap_conftest`

### 3) LLM Orchestrator

* **PLAN** → collect missing symbols; resolve & repack
* **GENERATE** → produce tests using **canonical import** (never `src.my_module`)
* Run: `pytest` (+ coverage/mutation) with **bootstrap** (env or conftest) applied
* **REFINE** → repair compile/runtime, increase coverage, kill mutants
* If bug: **ASK FOR MANUAL FIX STEPS** → failing test + BUG NOTE

### 4) Runners

* `pytest -q --maxfail=1`
* `coverage run -m pytest && coverage json`
* Optional quick mutation pass (e.g., `mutmut`/`cosmic-ray`) to surface weak oracles.

---

## Data Contracts

### ContextPack (input to LLM)

```json
{
  "target": {"module_file": "path/to/src/my_pkg/sub/module.py", "object": "Class.method|function"},
  "import_map": {
    "target_import": "from my_pkg.sub import module as _under_test",
    "sys_path_roots": ["/abs/repo/src"],
    "needs_bootstrap": true,
    "bootstrap_conftest": "PY CODE STRING"
  },
  "focal": {"source": "str", "signature": "str", "docstring": "str|null"},
  "resolved_defs": [{"name": "Symbol", "kind": "class|func|const|enum|fixture", "signature": "str", "doc": "str|null", "body": "str|omitted"}],
  "property_context": {
    "ranked_methods": [{"qualname": "pkg.Class.method", "level": "intra|repo", "relation": "complete|G|W|T"}],
    "gwt_snippets": {"given": ["..."], "when": ["..."], "then": ["..."]},
    "test_bundles": [{"test_name": "test_x", "imports": ["..."], "fixtures": ["..."], "mocks": ["..."], "assertions": ["..."]}]
  },
  "conventions": {
    "test_style": "pytest",
    "allowed_libs": ["pytest","hypothesis"],
    "determinism": {"seed": 1337, "tz": "UTC", "freeze_time": true},
    "io_policy": {"network": "forbidden", "fs": "tmp_path_only"}
  },
  "budget": {"max_input_tokens": 60000}
}
```

### ExecutionFeedback (into REFINE / MANUAL FIX)

```json
{
  "result": "pass|fail|error|flake",
  "trace_excerpt": "short traceback or assertion diff",
  "coverage": {"file": "path", "uncovered_lines": [12,19], "missing_branches": ["L42: condB"]},
  "mutants_survived": [{"op": "AOR", "location": "pkg/mod.py:88", "hint": "add stronger oracle"}],
  "notes": "flakiness/timeouts/module import issues, if any"
}
```

---

## Module Resolution & Import Policy (first-class)

**Canonical import, never `src.*`:**

1. **Packaged project at root (no `src/`)**

   * Canonical: `top_pkg.sub.module`
   * `sys_path_roots`: \[`<repo>`] (pytest usually adds this; we still bootstrap if needed)

2. **`src/` layout**

   * Canonical: `top_pkg.sub.module` (package lives under `src/`)
   * `sys_path_roots`: \[`<repo>/src`] (must be added)
   * Tests **must not** write `import src.top_pkg...`

3. **Monorepo / multi-package**

   * Pick package root containing the file; compute canonical absolute import for that package.
   * `sys_path_roots`: add each implicated package root (e.g., `<repo>/libs/pkg_a/src`, `<repo>/libs/pkg_b/src`)

4. **Unpackaged loose modules (no `__init__.py`)**

   * Prefer adding their parent directories to `sys.path` and still import absolutely by derived module name.
   * Only if impossible, fallback to `importlib.machinery.SourceFileLoader` in a tiny helper. (Discouraged; use bootstrap if at all possible.)

**Bootstrap choices (applied automatically by the controller):**

* **Conftest bootstrap (default, hermetic):**

  ```python
  # tests/conftest.py (generated once)
  import sys, pathlib
  for p in [{{ sys_path_roots }}]:
      p = pathlib.Path(p).resolve()
      if str(p) not in sys.path:
          sys.path.insert(0, str(p))
  ```
* **Env bootstrap (CI-friendly alternative):**

  * Run with `PYTHONPATH={{':'.join(sys_path_roots)}} pytest ...`
* The controller creates/updates `tests/conftest.py` **once** if `needs_bootstrap`.

**LLM guardrails in prompts:** model receives **`target_import`** and must import using it; generating any other path is treated as an error in REFINE.

---

## Stage Prompts

### (A) PLAN

```
SYSTEM:
You are a senior Python test engineer. You write small, correct, deterministic pytest tests.
Do NOT guess missing symbols. List them.

USER:
TARGET
- File: {{target.module_file}}
- Object: {{target.object}}
- Canonical import to use in tests (must use exactly this):
  {{import_map.target_import}}

Focal code:
{{focal.source}}

Signature/docstring:
{{focal.signature}}
{{focal.docstring}}

Precise repository context (curated):
Resolved definitions you can rely on:
{{resolved_defs_compact}}

Property-related examples (GIVEN/WHEN/THEN):
GIVEN:
{{gwt_snippets.given}}
WHEN:
{{gwt_snippets.when}}
THEN:
{{gwt_snippets.then}}

Repo conventions:
{{conventions}}

TASK:
1) Produce a TEST PLAN (cases, boundaries, exceptions, side-effects, fixtures/mocks).
2) List "missing_symbols" you need (fully qualified where possible).
3) Confirm the import you will write at the top of the test file.
Output strictly as JSON: {"plan":[...], "missing_symbols":[...], "import_line":"..."}
```

*Controller:* if `missing_symbols` non-empty → resolve via Jedi/Pyright; repack and (optionally) re-PLAN. (This mirrors precise on-demand context injection shown to cut hallucinations and improve mutation kills.  )

### (B) GENERATE

```
SYSTEM:
Output a single runnable pytest module. Use ONLY the provided canonical import.
No network. Use tmp_path for FS. Keep imports minimal.

USER:
Canonical import (must appear at top of the file):
{{import_map.target_import}}

Focal code (trimmed):
{{focal.source}}

Resolved definitions (only what you can call):
{{resolved_defs_compact}}

Property-related patterns and test-bundle fragments:
{{property_context_compact}}

Repo conventions / determinism:
{{conventions}}

Approved TEST PLAN:
{{approved_plan_json}}

REQUIREMENTS:
- Use EXACTLY the canonical import above.
- Prefer pytest parametrization for partitions/boundaries.
- Assertions must check behavior (not just “no exception”).
- If side-effects occur, assert on state/IO/logs accordingly.
- Name tests `test_<target_simplename>_<behavior>`.
- Output ONLY the complete test module in one fenced block.
```

*(The APT idea—ranked G/W/T property snippets + test bundles—feeds task-specific context the model can actually reuse. )*

### (C) REFINE

```
SYSTEM:
You repair Python tests with minimal edits. Keep style and canonical import unchanged.

USER:
Last tests (trim to failing parts):
{{last_tests_excerpt}}

Focal code (trimmed):
{{focal.source_trimmed}}

Canonical import (DO NOT change):
{{import_map.target_import}}

Execution feedback:
- Result: {{result}}
- Trace excerpt: {{trace_excerpt}}   # includes ModuleNotFoundError etc.
- Coverage gaps: {{coverage_summary}}
- Surviving mutants: {{mutants_survived}}

Constraints:
- Do NOT introduce new undefined symbols. If truly needed, output {"missing_symbols":[...]} and nothing else.

TASK:
1) Brief rationale of changes (compile fix / wrong assumption / new branch case / stronger oracle).
2) Output the corrected full test module.
3) If you require new symbols, output only {"missing_symbols":[...]}.
```

### (D) ASK FOR MANUAL FIX STEPS (real bug)

```
SYSTEM:
When code has a real defect, deliver:
(1) a deliberately failing, high-signal pytest test (it will PASS once code is fixed),
(2) a concise BUG NOTE for engineers.

USER:
Canonical import:
{{import_map.target_import}}

Focal code:
{{focal.source}}

Property-related THEN snippets that justify expected behavior:
{{gwt_snippets.then}}

Trace excerpt / outputs:
{{trace_excerpt}}

Repo conventions:
{{conventions}}

Output:
1) One fenced code block (python) with a single test file named like
   test_bug_<target_simplename>_<symptom>.py using the canonical import.
2) One fenced code block (markdown) with the BUG NOTE: Title, Summary, Steps to Reproduce,
   Expected vs Actual, Suspected root-cause (file:line), Related methods/tests, Risk/Blast radius,
   Suggested fix sketch.
```

---

## Controller Algorithm (import-aware)

```python
def detect_repo_layout(repo_root):
    # Prefer pyproject/setup.cfg; else heuristics
    # returns { "src_roots": [...], "packages": [...], "monorepo_pkgs": [...] }

def compute_canonical_import(repo_info, module_file):
    # Walk parents to locate package root (__init__.py chain)
    # Map through package_dir / src roots if configured
    # return { "target_import", "sys_path_roots", "needs_bootstrap" }

def ensure_bootstrap(import_map, tests_dir):
    if import_map["needs_bootstrap"]:
        write_conftest_py(tests_dir, import_map["sys_path_roots"])

def build_context_pack(target):
    repo_info = detect_repo_layout(REPO)
    import_map = compute_canonical_import(repo_info, target.module_file)
    ensure_bootstrap(import_map, tests_dir="tests")
    focal = parse_focal(target)
    props = property_analyzer(target)          # APT: rank G/W/T + bundles
    bundles = test_case_indexer(props.methods)
    resolved = resolve_defs_needed(focal, bundles)  # precise injections (RATester-like)
    return ContextPack(import_map, focal, resolved, props, conventions())

def plan_stage(ctx):
    plan = call_llm(PLAN_PROMPT, ctx)
    if plan["import_line"].strip() != ctx.import_map["target_import"].strip():
        # force alignment
        plan["import_line"] = ctx.import_map["target_import"]
    if plan["missing_symbols"]:
        ctx.resolved_defs += fetch_defs(plan["missing_symbols"])
        plan = call_llm(PLAN_PROMPT, ctx)
    return plan

def generate_stage(ctx, plan):
    return call_llm(GENERATE_PROMPT, ctx | {"approved_plan_json": plan})

def run_tests(code_path, sys_path_roots):
    env = os.environ.copy()
    if sys_path_roots:
        env["PYTHONPATH"] = os.pathsep.join(sys_path_roots + [env.get("PYTHONPATH","")])
    # run pytest/coverage/mutation with env
    return collect_feedback()

def refine_stage(ctx, code, feedback):
    if feedback.result == "pass":
        return code
    refined = call_llm(REFINE_PROMPT, ctx | {"last_tests_excerpt": trim(code), **feedback.to_dict()})
    if isinstance(refined, dict) and "missing_symbols" in refined:
        ctx.resolved_defs += fetch_defs(refined["missing_symbols"])
        refined = call_llm(REFINE_PROMPT, ctx | {"last_tests_excerpt": trim(code), **feedback.to_dict()})
    return refined

def manual_fix_stage(ctx, feedback):
    return call_llm(MANUAL_FIX_PROMPT, ctx | feedback.to_dict())
```

---

## Quality Gates

1. **Import gate:** The generated file **must** contain `{{import_map.target_import}}` as the first non-comment import; reject otherwise.
2. **Bootstrap gate:** If `needs_bootstrap`, ensure `tests/conftest.py` exists and adds `sys_path_roots` to `sys.path`.
3. **Compile gate:** `pytest -q` imports succeed (no `ModuleNotFoundError`).
4. **Determinism gate:** `pytest -q -k <new tests>` twice with the same seed → identical results.
5. **Coverage gate:** new lines/branches for target increased vs baseline; uncovered branches feed the next REFINE round.
6. **Mutation gate (fast sample):** aim to reduce survivors each REFINE loop (precise injections + property retrieval are shown to improve mutation killing and correctness). &#x20;

---

## Why this design

* **Property-based retrieval (APT)** supplies **task-specific** G/W/T context + test bundles, improving correctness/maintainability over generic RAG. &#x20;
* **Precise, on-demand context injection (RATester)** curbs hallucinated calls/arity errors and strengthens tests in mutation analyses. &#x20;
* **Import policy + bootstrap** guarantees the model uses the **right** module path (`my_pkg...`) and that the runner can import it across `src/`/flat/monorepo layouts—eliminating the classic `src.my_module` anti-pattern.
