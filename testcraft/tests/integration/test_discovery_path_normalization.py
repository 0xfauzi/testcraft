from pathlib import Path

from testcraft.adapters.io.file_discovery import FileDiscoveryService


def test_path_normalization_does_not_exclude_valid_files(tmp_path: Path, monkeypatch):
    # Create a simple project structure
    project = tmp_path
    src = project / "src" / "mod"
    src.mkdir(parents=True)
    f = src / "x.py"
    f.write_text("def x():\n    return 1\n", encoding="utf-8")

    discovery = FileDiscoveryService()

    # Simulate macOS /private prefix behavior by wrapping Path.resolve for project only
    original_resolve = Path.resolve

    def resolve_with_private(self: Path):  # type: ignore[override]
        p = original_resolve(self)
        s = str(p)
        if s.startswith("/") and not s.startswith("/private/"):
            # Pretend OS adds /private; our code should still include files
            return Path("/private" + s)
        return p

    monkeypatch.setattr(Path, "resolve", resolve_with_private)

    files = discovery.discover_source_files(
        project, file_patterns=["*.py"], include_test_files=False
    )

    # Even with path prefix mismatch, we should still include the file after lenient fallback
    assert any("x.py" in fp for fp in files)
