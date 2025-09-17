from __future__ import annotations

import ast
import re
from typing import Any

from .....domain.models import TestGenerationPlan


def get_contract_context(parser_port: Any, source_path: Any, plan: TestGenerationPlan) -> list[str]:
    items: list[str] = []
    try:
        if source_path is None:
            return items
        parse_result = parser_port.parse_file(source_path)
        tree = parse_result.get("ast")
        source_lines = parse_result.get("source_lines", []) or []
        source_code = "\n".join(source_lines) if source_lines else ""

        for element in plan.elements_to_test[:5]:
            header = f"# Contract: {getattr(element.type, 'value', str(element.type))} {element.name}"
            parts = [header]

            node = _find_node_for_element(tree, element)
            signature = _get_signature(node, element, source_lines)
            if signature:
                parts.append(f"signature: {signature[:200]}")

            invariants, raises = _extract_invariants_and_raises(node, source_code)

            doc = getattr(element, "docstring", None) or ""
            if doc:
                info = _parse_docstring(doc)
            else:
                info = {"summary": "", "params": [], "returns": "", "raises": []}

            if info.get("params"):
                parts.append(f"params: {list(info['params'])[:8]}")
            if info.get("returns"):
                parts.append(f"returns: {str(info['returns'])[:120]}")

            doc_raises = info.get("raises") or []
            if doc_raises or raises:
                combined_raises = list(dict.fromkeys(list(doc_raises) + raises))[:8]
                if combined_raises:
                    parts.append(f"raises: {combined_raises}")
            if invariants:
                parts.append("invariants: [" + ", ".join(invariants[:3]) + "]")
            if info.get("summary"):
                parts.append(f"doc: {info['summary'][:300]}")

            items.append("\n".join(parts)[:600])
    except Exception:
        pass
    return items


def _find_node_for_element(ast_tree: Any, elem: Any) -> Any | None:
    try:
        name = getattr(elem, "name", "")
        start = getattr(elem, "line_range", (0, 0))[0]
        if not ast_tree:
            return None
        if "." in name:
            cls_name, meth_name = name.split(".", 1)
            for node in getattr(ast_tree, "body", []):
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for sub in node.body:
                        if isinstance(sub, ast.FunctionDef) and sub.name == meth_name:
                            return sub
        for node in getattr(ast_tree, "body", []):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        for node in ast.walk(ast_tree):
            if hasattr(node, "lineno") and getattr(node, "lineno", -1) == start:
                return node
    except Exception:
        return None
    return None


def _get_signature(node: Any, element: Any, source_lines: list[str]) -> str:
    try:
        if isinstance(node, ast.FunctionDef):
            return _signature_of_function(node, source_lines)
        elif isinstance(node, ast.AsyncFunctionDef):
            sig_core = _signature_of_function(node, source_lines)
            return ("async " + sig_core) if sig_core else ""
        elif isinstance(node, ast.ClassDef):
            return _signature_of_class(node, source_lines)
    except Exception:
        pass
    try:
        start = getattr(element, "line_range", (0, 0))[0]
        if start and start - 1 < len(source_lines):
            return source_lines[start - 1].strip()
    except Exception:
        pass
    return ""


def _signature_of_function(fn: ast.FunctionDef, source_lines: list[str]) -> str:
    try:
        if source_lines and hasattr(fn, "lineno") and fn.lineno - 1 < len(source_lines):
            return source_lines[fn.lineno - 1].strip()
    except Exception:
        pass
    return f"def {fn.name}(...):"


def _signature_of_class(cls: ast.ClassDef, source_lines: list[str]) -> str:
    try:
        if source_lines and hasattr(cls, "lineno") and cls.lineno - 1 < len(source_lines):
            return source_lines[cls.lineno - 1].strip()
    except Exception:
        pass
    return f"class {cls.name}:"


def _extract_invariants_and_raises(node: Any, source_code: str) -> tuple[list[str], list[str]]:
    invariants: list[str] = []
    raises: list[str] = []
    try:
        target_fn = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
        if target_fn is not None:
            for sub in ast.walk(target_fn):
                if isinstance(sub, ast.Assert) and len(invariants) < 3:
                    try:
                        expr_txt = ast.get_source_segment(source_code, sub.test) or ""
                        if expr_txt:
                            invariants.append(expr_txt[:60])
                    except Exception:
                        pass
                if isinstance(sub, ast.Raise) and sub.exc is not None:
                    try:
                        exc_txt = ast.get_source_segment(source_code, sub.exc) or ""
                        if exc_txt:
                            exc_name = exc_txt.split("(")[0].strip()
                            raises.append(exc_name)
                    except Exception:
                        pass
            raises = list(dict.fromkeys([r for r in raises if r]))[:5]
            invariants = list(dict.fromkeys([i for i in invariants if i]))
    except Exception:
        pass
    return invariants, raises


def _parse_docstring(doc: str) -> dict[str, Any]:
    info: dict[str, Any] = {"summary": "", "params": [], "returns": "", "raises": []}
    if not isinstance(doc, str) or not doc.strip():
        return info
    try:
        lines = [l.rstrip() for l in doc.splitlines()]
        for ln in lines:
            if ln.strip():
                info["summary"] = re.sub(r"\s+", " ", ln.strip())
                break
        for m in re.finditer(r":param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:", doc):
            info["params"].append(m.group(1))
        m = re.search(r":return[s]?\s*:\s*(.+)", doc)
        if m:
            info["returns"] = re.sub(r"\s+", " ", m.group(1)).strip()
        for m in re.finditer(r":raise[s]?\s+([A-Za-z_][A-Za-z0-9_]*)\s*:?", doc):
            info["raises"].append(m.group(1))

        def _collect_section(header: str) -> list[str]:
            items: list[str] = []
            try:
                idx = next(i for i, l in enumerate(lines) if l.strip().lower().startswith(header))
            except StopIteration:
                return items
            for l in lines[idx + 1 : idx + 12]:
                if not l.strip():
                    break
                m2 = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:\(]", l)
                if m2:
                    items.append(m2.group(1))
            return items

        if not info["params"]:
            info["params"] = _collect_section("args:") or _collect_section("parameters:")
        if not info["raises"]:
            info["raises"] = _collect_section("raises:")
        if not info["returns"]:
            try:
                idx = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("returns"))
                if idx + 1 < len(lines):
                    info["returns"] = re.sub(r"\s+", " ", lines[idx + 1].strip())
            except StopIteration:
                pass
    except Exception:
        pass
    return info



