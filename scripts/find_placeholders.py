#!/usr/bin/env python3

"""
Throw-away scanner to surface incomplete functionality across the codebase.

Goal: Identify and report placeholder/stub/incomplete code with maximum detail
to enable a follow-up agent to fill the gaps. This script intentionally uses
regex/grep-style heuristics (no AST) to align with the task constraints.

Outputs (by default):
- Markdown report: build/reports/placeholders.md
- JSON report:     build/reports/placeholders.json

Scope defaults:
- Include:  testcraft/
- Exclude:  tests/, build/, htmlcov/, test-env/, testcraft.egg-info/

Usage examples:
  python scripts/find_placeholders.py
  python scripts/find_placeholders.py --paths testcraft/ --md-out build/reports/placeholders.md --json-out build/reports/placeholders.json

This is a temporary, throw-away tool.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ------------------------------ Patterns ------------------------------------


@dataclasses.dataclass
class PatternSpec:
    name: str
    description: str
    regex: Optional[re.Pattern]
    category: str
    severity: str  # one of: critical | high | medium | low
    # If regex is None, the pattern is handled procedurally (e.g., next-line checks)


def compile_patterns() -> List[PatternSpec]:
    flags = re.MULTILINE | re.IGNORECASE
    return [
        PatternSpec(
            name="not_implemented_usage",
            description="raise/return NotImplemented or NotImplementedError",
            regex=re.compile(r"\b(raise|return)\s+NotImplemented(Error)?\b", flags),
            category="not_implemented",
            severity="critical",
        ),
        PatternSpec(
            name="except_pass_same_line",
            description="except ...: pass on the same line",
            regex=re.compile(r"^\s*except[\s\w\.,\(\)\:]*:\s*pass\b", flags),
            category="except_pass",
            severity="high",
        ),
        PatternSpec(
            name="def_pass_same_line",
            description="def ...: pass on the same line",
            regex=re.compile(
                r"^\s*def\s+\w+\s*\([^\)]*\)\s*(?:->[^:]*)?:\s*pass\b",
                flags,
            ),
            category="pass_body",
            severity="high",
        ),
        PatternSpec(
            name="class_pass_same_line",
            description="class ...: pass on the same line",
            regex=re.compile(
                r"^\s*class\s+\w+(?:\([^\)]*\))?\s*:\s*pass\b",
                flags,
            ),
            category="pass_body",
            severity="high",
        ),
        PatternSpec(
            name="ellipsis_return",
            description="return ... placeholder",
            regex=re.compile(r"^\s*return\s+\.\.\.\s*(?:#.*)?$", flags),
            category="ellipsis",
            severity="critical",
        ),
        PatternSpec(
            name="standalone_ellipsis",
            description="standalone ... line",
            regex=re.compile(r"^\s*\.\.\.\s*(?:#.*)?$", flags),
            category="ellipsis",
            severity="high",
        ),
        PatternSpec(
            name="pass_with_todo_comment",
            description="pass followed by TODO/FIXME/XXX comment",
            regex=re.compile(r"\bpass\b.*#.*\b(TODO|FIXME|XXX)\b", flags),
            category="todo_stub",
            severity="high",
        ),
        # The following two are handled procedurally to ensure both tokens appear
        PatternSpec(
            name="todo_with_stub_words",
            description="TODO/FIXME/XXX combined with implement/placeholder/stub/wip/tbd",
            regex=None,
            category="todo_stub",
            severity="medium",
        ),
        PatternSpec(
            name="placeholder_or_stub_word",
            description="lines containing 'placeholder' or 'stub'",
            regex=None,
            category="todo_stub",
            severity="low",
        ),
        # Procedural: next-line pass for except/def/class
        PatternSpec(
            name="except_pass_next_line",
            description="except ... then next indented line is pass",
            regex=None,
            category="except_pass",
            severity="high",
        ),
        PatternSpec(
            name="def_or_class_pass_next_line",
            description="def/class header then next indented line is pass",
            regex=None,
            category="pass_body",
            severity="high",
        ),
    ]


TODO_WORDS = re.compile(r"\b(TODO|FIXME|XXX)\b", re.IGNORECASE)
STUB_WORDS = re.compile(r"\b(implement|placeholder|stub|wip|tbd)\b", re.IGNORECASE)
PLACEHOLDER_OR_STUB_WORD = re.compile(r"\b(placeholder|stub)\b", re.IGNORECASE)


# ------------------------------ Findings ------------------------------------


@dataclasses.dataclass
class Finding:
    file_path: str
    line_number: int
    matched_text: str
    category: str
    severity: str
    pattern_names: List[str]
    context_symbol: Optional[str]
    context_kind: Optional[str]  # "function" | "class" | None
    module_path: Optional[str]
    code_snippet: str
    code_start_line: int

    def to_json(self) -> Dict[str, object]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "category": self.category,
            "severity": self.severity,
            "pattern_names": self.pattern_names,
            "context_symbol": self.context_symbol,
            "context_kind": self.context_kind,
            "module_path": self.module_path,
            "code_snippet": self.code_snippet,
            "code_start_line": self.code_start_line,
            "matched_text": self.matched_text,
        }


# ------------------------------ Utilities -----------------------------------


def is_python_file(path: Path) -> bool:
    return path.suffix == ".py"


def should_exclude(path: Path, exclude_dirs: Sequence[str]) -> bool:
    parts = set(p for p in path.parts)
    for excl in exclude_dirs:
        if excl in parts:
            return True
    return False


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for odd encodings; treat as binary skip
        return ""


def derive_module_path(repo_root: Path, file_path: Path) -> Optional[str]:
    try:
        rel = file_path.relative_to(repo_root)
    except ValueError:
        rel = file_path
    if rel.suffix != ".py":
        return None
    parts = list(rel.with_suffix("").parts)
    # Convert path to dotted module; drop leading non-package segments if any
    return ".".join(parts)


def find_enclosing_symbol(lines: List[str], start_index: int) -> Tuple[Optional[str], Optional[str]]:
    """Find the nearest enclosing def/class name above start_index.

    Returns (symbol_name, kind) where kind is "function" or "class".
    """
    header_def = re.compile(r"^\s*def\s+(\w+)\s*\(")
    header_class = re.compile(r"^\s*class\s+(\w+)\s*(?:\(|:)\s*")
    # The regexes above are intentionally simple; we only need best-effort context.

    for i in range(start_index, -1, -1):
        line = lines[i]
        m_def = re.match(r"^\s*def\s+(\w+)\s*\(", line)
        if m_def:
            return m_def.group(1), "function"
        m_cls = re.match(r"^\s*class\s+(\w+)\s*(?:\(|:)\s*", line)
        if m_cls:
            return m_cls.group(1), "class"
    return None, None


def extract_snippet(lines: List[str], center_index: int, context: int = 3) -> Tuple[str, int]:
    start = max(0, center_index - context)
    end = min(len(lines), center_index + context + 1)
    snippet_lines = []
    for i in range(start, end):
        # Preserve original text; add simple line markers
        prefix = "-> " if i == center_index else "   "
        snippet_lines.append(f"{prefix}{lines[i].rstrip()}")
    snippet_text = "\n".join(snippet_lines)
    return snippet_text, start + 1  # return 1-based start line


def add_or_merge_finding(
    findings_by_key: Dict[Tuple[str, int], Finding],
    file_path: Path,
    line_index: int,
    line_text: str,
    pattern: PatternSpec,
    lines: List[str],
    repo_root: Path,
    context_lines: int,
) -> None:
    key = (str(file_path), line_index + 1)
    if key not in findings_by_key:
        symbol, kind = find_enclosing_symbol(lines, line_index)
        snippet, snippet_start = extract_snippet(lines, line_index, context_lines)
        findings_by_key[key] = Finding(
            file_path=str(file_path),
            line_number=line_index + 1,
            matched_text=line_text.rstrip(),
            category=pattern.category,
            severity=pattern.severity,
            pattern_names=[pattern.name],
            context_symbol=symbol,
            context_kind=kind,
            module_path=derive_module_path(repo_root, file_path),
            code_snippet=snippet,
            code_start_line=snippet_start,
        )
    else:
        existing = findings_by_key[key]
        if pattern.name not in existing.pattern_names:
            existing.pattern_names.append(pattern.name)
        # Elevate severity if the new pattern is more severe
        severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if severity_rank.get(pattern.severity, 0) > severity_rank.get(existing.severity, 0):
            existing.severity = pattern.severity
        # If categories differ, keep the more urgent one
        category_priority = {"not_implemented": 3, "ellipsis": 2, "except_pass": 2, "pass_body": 2, "todo_stub": 1}
        if category_priority.get(pattern.category, 0) > category_priority.get(existing.category, 0):
            existing.category = pattern.category


# ------------------------------ Scanner Core --------------------------------


def _is_inside_quotes(line: str, index: int) -> bool:
    """Return True if the given character index is within a quoted string on this line.
    Handles simple single-line quoted strings with escape characters.
    """
    in_single = False
    in_double = False
    escape = False
    for i, ch in enumerate(line):
        if i >= index:
            break
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "'" and not in_double:
            in_single = not in_single
    return in_single or in_double


def scan_file(
    repo_root: Path,
    file_path: Path,
    patterns: Sequence[PatternSpec],
    context_lines: int,
) -> List[Finding]:
    text = read_text(file_path)
    if not text:
        return []

    lines = text.splitlines()
    findings_by_key: Dict[Tuple[str, int], Finding] = {}

    # Regex-based simple matches
    for pat in patterns:
        if pat.regex is None:
            continue
        for match in pat.regex.finditer(text):
            # Compute the line index for this match
            start_pos = match.start()
            line_index = text.count("\n", 0, start_pos)
            line_text = lines[line_index] if 0 <= line_index < len(lines) else ""
            # Column offset within the current line
            last_newline = text.rfind("\n", 0, start_pos)
            col = start_pos - (last_newline + 1 if last_newline != -1 else 0)

            # Heuristic: skip if match is inside a quoted string (applies to NotImplemented pattern)
            if pat.name == "not_implemented_usage" and _is_inside_quotes(line_text, col):
                continue

            # Heuristic: skip if match is fully inside a quoted string (very rough)
            # We avoid complex parsing here; only skip obvious single-line quoted literals.
            # (Kept for future patterns if needed)
            # if line_text.count('"') % 2 == 1 or line_text.count("'") % 2 == 1:
            #     pass

            add_or_merge_finding(
                findings_by_key,
                file_path,
                line_index,
                line_text,
                pat,
                lines,
                repo_root,
                context_lines,
            )

    # Procedural detections
    for i, line in enumerate(lines):
        stripped = line.strip()

        # TODO words + stub words on the same line
        if TODO_WORDS.search(stripped) and STUB_WORDS.search(stripped):
            pat = next(p for p in patterns if p.name == "todo_with_stub_words")
            add_or_merge_finding(findings_by_key, file_path, i, line, pat, lines, repo_root, context_lines)

        # Placeholder or stub words (low severity)
        if PLACEHOLDER_OR_STUB_WORD.search(stripped):
            pat = next(p for p in patterns if p.name == "placeholder_or_stub_word")
            add_or_merge_finding(findings_by_key, file_path, i, line, pat, lines, repo_root, context_lines)

        # except ... then next significant line is 'pass'
        if re.match(r"^\s*except[\s\w\.,\(\)\:]*:\s*$", line, re.IGNORECASE):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and re.match(r"^\s*pass\b", lines[j]):
                pat = next(p for p in patterns if p.name == "except_pass_next_line")
                add_or_merge_finding(findings_by_key, file_path, j, lines[j], pat, lines, repo_root, context_lines)

        # def/class header then next significant line is 'pass'
        if re.match(r"^\s*def\s+\w+\s*\([^\)]*\)\s*(?:->[^:]*)?:\s*$", line):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and re.match(r"^\s*pass\b", lines[j]):
                pat = next(p for p in patterns if p.name == "def_or_class_pass_next_line")
                add_or_merge_finding(findings_by_key, file_path, j, lines[j], pat, lines, repo_root, context_lines)

        if re.match(r"^\s*class\s+\w+(?:\([^\)]*\))?\s*:\s*$", line):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and re.match(r"^\s*pass\b", lines[j]):
                pat = next(p for p in patterns if p.name == "def_or_class_pass_next_line")
                add_or_merge_finding(findings_by_key, file_path, j, lines[j], pat, lines, repo_root, context_lines)

    return list(findings_by_key.values())


def iter_python_files(paths: Sequence[Path], exclude_dirs: Sequence[str]) -> Iterable[Path]:
    for base in paths:
        if base.is_file() and is_python_file(base) and not should_exclude(base, exclude_dirs):
            yield base
            continue
        if base.is_dir():
            for root, dirnames, filenames in os.walk(base):
                # prune excluded directories
                dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
                root_path = Path(root)
                for fname in filenames:
                    fpath = root_path / fname
                    if is_python_file(fpath) and not should_exclude(fpath, exclude_dirs):
                        yield fpath


# ------------------------------ Reporting -----------------------------------


def ensure_parent_dir(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def write_json(findings: List[Finding], json_path: Path) -> None:
    ensure_parent_dir(json_path)
    data = [f.to_json() for f in findings]
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_rg_reproduction(exclude_dirs: Sequence[str]) -> str:
    exclude_args = " ".join(f"-g '!{d}/**'" for d in exclude_dirs)
    cmds = [
        f"rg -n --no-ignore {exclude_args} "
        r'-e "\b(raise|return)\s+NotImplemented(Error)?\b" '
        r'-e "^\s*except[\s\w\.,\(\)\:]*:\s*pass\b" '
        r'-e "^\s*def\s+\w+\s*\([^\)]*\)\s*(?:->[^:]*)?:\s*pass\b" '
        r'-e "^\s*class\s+\w+(?:\([^\)]*\))?\s*:\s*pass\b" '
        r'-e "^\s*return\s+\.\.\.\s*(?:#.*)?$" '
        r'-e "^\s*\.\.\.\s*(?:#.*)?$" '
        r'-e "\bpass\b.*#.*\b(TODO|FIXME|XXX)\b" '
        r'--glob "**/*.py"',
        "# For next-line 'pass' after except/def/class, search headers then inspect next line manually",
        f"rg -n --no-ignore {exclude_args} "
        r'-e "^\s*(except[\s\w\.,\(\)\:]*:|def\s+\w+\s*\([^\)]*\)\s*(?:->[^:]*)?:|class\s+\w+(?:\([^\)]*\))?:)\s*$" '
        r'--glob "**/*.py"',
        "# For TODO+stub words on same line",
        f"rg -n --no-ignore {exclude_args} "
        r'-e "\b(TODO|FIXME|XXX)\b.*\b(implement|placeholder|stub|wip|tbd)\b" '
        r'--glob "**/*.py"',
    ]
    return "\n".join(cmds)


def write_markdown(
    findings: List[Finding],
    md_path: Path,
    repo_root: Path,
    exclude_dirs: Sequence[str],
) -> None:
    ensure_parent_dir(md_path)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    by_category: Dict[str, int] = defaultdict(int)
    by_severity: Dict[str, int] = defaultdict(int)
    for f in findings:
        by_category[f.category] += 1
        by_severity[f.severity] += 1

    lines: List[str] = []
    lines.append(f"# Incomplete Functionality Scan Report")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Repo root: {repo_root}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Findings: {}".format(len(findings)))
    lines.append("- By severity:")
    for sev in ["critical", "high", "medium", "low"]:
        if sev in by_severity:
            lines.append(f"  - {sev}: {by_severity[sev]}")
    lines.append("- By category:")
    for cat, count in sorted(by_category.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  - {cat}: {count}")
    lines.append("")

    lines.append("## Ripgrep reproduction commands")
    lines.append("")
    lines.append("```bash")
    lines.append(generate_rg_reproduction(exclude_dirs))
    lines.append("```")
    lines.append("")

    lines.append("## Detailed Findings")
    lines.append("")
    for idx, f in enumerate(sorted(findings, key=lambda x: (x.file_path, x.line_number)), start=1):
        title = f"{idx}. {f.file_path}:{f.line_number} — {f.category} [{f.severity}]"
        lines.append(f"### {title}")
        lines.append("")
        if f.module_path:
            lines.append(f"- Module: `{f.module_path}`")
        if f.context_symbol:
            lines.append(f"- Context: {f.context_kind} `{f.context_symbol}`")
        if f.pattern_names:
            lines.append(f"- Matched patterns: `{', '.join(sorted(f.pattern_names))}`")
        lines.append(f"- Matched line: `{f.matched_text.strip()}`")
        lines.append("")
        lines.append("```python")
        lines.append(f"# snippet starting at line {f.code_start_line}")
        lines.append(f.code_snippet)
        lines.append("```")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------ CLI -----------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan for incomplete functionality (placeholder/stub) patterns.")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["testcraft"],
        help="Paths/directories to include (default: testcraft)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["tests", "build", "htmlcov", "test-env", "testcraft.egg-info"],
        help="Directory names to exclude (exact matches on path parts).",
    )
    parser.add_argument(
        "--md-out",
        default="build/reports/placeholders.md",
        help="Path to write Markdown report.",
    )
    parser.add_argument(
        "--json-out",
        default="build/reports/placeholders.json",
        help="Path to write JSON report (set empty to disable).",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="Number of context lines before/after match.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce stdout output; still writes reports.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    include_paths = [repo_root / Path(p) for p in args.paths]
    exclude_dirs = list(args.exclude)

    pattern_specs = compile_patterns()

    all_findings: List[Finding] = []
    for py_file in iter_python_files(include_paths, exclude_dirs):
        file_findings = scan_file(repo_root, py_file, pattern_specs, args.context_lines)
        all_findings.extend(file_findings)

    # Sort and stable-dedupe (already deduped per file:line)
    all_findings.sort(key=lambda f: (f.file_path, f.line_number, f.category, f.severity))

    md_path = repo_root / Path(args.md_out)
    write_markdown(all_findings, md_path, repo_root, exclude_dirs)

    if args.json_out:
        json_path = repo_root / Path(args.json_out)
        write_json(all_findings, json_path)

    if not args.quiet:
        print(f"Wrote Markdown report to: {md_path}")
        if args.json_out:
            print(f"Wrote JSON report to: {json_path}")
        print(f"Total findings: {len(all_findings)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


