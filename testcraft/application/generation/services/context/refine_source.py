from __future__ import annotations

from pathlib import Path
from typing import Any


def build_refinement_context(
    parser: Any,
    context_port: Any,
    config: dict[str, Any],
    test_file: Path,
    test_content: str,
    structure_builder: Any,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "test_file_path": str(test_file),
        "test_content": test_content,
        "related_source_files": [],
        "imports_context": [],
        "dependency_analysis": {},
        "retrieved_context": [],
        "project_structure": {},
    }

    if not config.get("enable_context", True):
        return context

    try:
        dependency_analysis = parser.analyze_dependencies(test_file)
        context["dependency_analysis"] = dependency_analysis
        imports = dependency_analysis.get("imports", [])
        internal_deps = dependency_analysis.get("internal_deps", [])
        if not internal_deps:
            try:
                derived = _derive_modules_from_test_ast(test_content)
                merged = list(dict.fromkeys(list(internal_deps) + derived))
                internal_deps = merged
                dependency_analysis["internal_deps"] = internal_deps
                context["dependency_analysis"] = dependency_analysis
            except Exception:
                pass

        try:
            context_port.index(test_file, content=test_content)
            related_context = context_port.get_related_context(test_file, relationship_type="all")
            for related_file_path in related_context.get("related_files", []):
                related_path = Path(related_file_path)
                if related_path.exists() and related_path.suffix == ".py":
                    try:
                        source_content = related_path.read_text(encoding="utf-8")
                        context["related_source_files"].append(
                            {
                                "path": str(related_path),
                                "content": source_content[:2000],
                                "relationship": "context_analysis",
                            }
                        )
                    except Exception:
                        pass
            retrieval_queries = _extract_test_context_queries(test_file, test_content)
            for query in retrieval_queries[:3]:
                try:
                    retrieval_result = context_port.retrieve(query=query, context_type="general", limit=3)
                    if retrieval_result.get("results"):
                        context["retrieved_context"].append({"query": query, "results": retrieval_result["results"][:2]})
                except Exception:
                    pass
            for dep in internal_deps:
                for source_path in _find_source_files_for_module(test_file, dep):
                    if source_path.exists():
                        try:
                            source_content = source_path.read_text(encoding="utf-8")
                            context["related_source_files"].append(
                                {
                                    "path": str(source_path),
                                    "content": source_content[:2000],
                                    "relationship": f"import_dependency: {dep}",
                                }
                            )
                        except Exception:
                            pass
            for import_info in imports:
                context["imports_context"].append(
                    {
                        "module": import_info.get("module", ""),
                        "items": import_info.get("items", []),
                        "alias": import_info.get("alias", ""),
                        "is_internal": import_info.get("module", "") in internal_deps,
                    }
                )
        except Exception:
            pass

        try:
            project_root = test_file.parent.parent if test_file.parent != test_file.parent.parent else test_file.parent
            directory_config = config.get("context_budgets", {}).get("directory_tree", {})
            max_depth = directory_config.get("max_depth", 3)
            max_entries_per_dir = directory_config.get("max_entries_per_dir", 150)
            include_py_only = directory_config.get("include_py_only", True)
            context["project_structure"] = structure_builder.build_tree_recursive(
                project_root, max_depth, max_entries_per_dir, include_py_only
            )
        except Exception:
            pass

    except Exception:
        return context

    return context


def _derive_modules_from_test_ast(test_content: str) -> list[str]:
    modules: list[str] = []
    try:
        tree = ast.parse(test_content)
    except Exception:
        return modules

    def add_module(mod: str) -> None:
        if not isinstance(mod, str):
            return
        mod = mod.strip()
        if not mod:
            return
        top = mod.split('.')[0]
        filtered = {
            "pytest", "unittest", "json", "re", "os", "sys", "pathlib", "typing",
            "datetime", "time", "collections", "itertools", "functools", "math",
            "rich", "logging", "schedule",
        }
        if top in filtered:
            return
        if mod not in modules:
            modules.append(mod)

    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if getattr(node, 'level', 0) == 0 and isinstance(getattr(node, 'module', None), str):
                add_module(node.module)
        elif isinstance(node, ast.Import):
            for alias in getattr(node, 'names', []) or []:
                name = getattr(alias, 'name', None)
                if isinstance(name, str):
                    add_module(name)
    dotted = [m for m in modules if "." in m]
    single = [m for m in modules if "." not in m]
    return dotted + single


def _extract_test_context_queries(test_file: Path, test_content: str) -> list[str]:
    import re, ast
    queries: list[str] = []
    try:
        try:
            tree = ast.parse(test_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    clean_name = node.name[5:]
                    query_words = re.findall(r"[a-z]+", clean_name.lower())
                    if query_words:
                        queries.append(" ".join(query_words))
                    for child in ast.walk(node):
                        if isinstance(child, ast.Constant) and isinstance(child.value, str):
                            if 3 <= len(child.value) <= 30 and child.value.replace("_", "").isalnum():
                                queries.append(child.value)
        except SyntaxError:
            pass
        if not queries:
            test_name = test_file.stem
            if test_name.startswith("test_"):
                clean_name = test_name[5:]
                query_words = re.findall(r"[a-z]+", clean_name.lower())
                if query_words:
                    queries.append(" ".join(query_words))
        unique = list(dict.fromkeys(queries))
        return unique[:5]
    except Exception:
        return [test_file.stem.replace("test_", "").replace("_", " ")]


def _find_source_files_for_module(test_file: Path, module_name: str) -> list[Path]:
    potential_paths: list[Path] = []
    try:
        module_parts = module_name.split(".")
        base_dirs = [test_file.parent, test_file.parent.parent]
        if "tests" in test_file.parts:
            tests_index = None
            for i, part in enumerate(test_file.parts):
                if part == "tests":
                    tests_index = i
                    break
            if tests_index is not None and tests_index > 0:
                project_root = Path(*test_file.parts[:tests_index])
                base_dirs.extend([project_root, project_root / test_file.parts[0], project_root / "src", project_root / "lib"])
        for base_dir in base_dirs:
            if not base_dir or not base_dir.exists():
                continue
            direct_file = base_dir / f"{module_parts[-1]}.py"
            if direct_file.exists():
                potential_paths.append(direct_file)
            if len(module_parts) > 1:
                package_file = base_dir
                for part in module_parts:
                    package_file = package_file / part
                package_file = package_file.with_suffix(".py")
                if package_file.exists():
                    potential_paths.append(package_file)
            package_init = base_dir / module_parts[0] / "__init__.py"
            if package_init.exists():
                potential_paths.append(package_init)
        seen: set[str] = set()
        unique_paths: list[Path] = []
        for path in potential_paths:
            path_str = str(path)
            if path_str not in seen:
                seen.add(path_str)
                unique_paths.append(path)
        return unique_paths
    except Exception:
        return []



