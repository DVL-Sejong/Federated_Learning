"""Canonical source identities across checkout, editable and wheel installations."""
from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath


def source_files() -> dict[str, Path]:
    """Map logical package paths to unique physical source files.

    Source folders are separate in the repository and merged in a wheel. Both
    forms must hash the same complete set of module files, including graph and
    metric implementations outside the primary physical package directory.
    """
    import defai
    roots = {Path(value).resolve() for value in defai.__path__ if Path(value).is_dir()}
    for name in ("defai.federated", "defai.metrics"):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            roots.update(Path(value).resolve().parent for value in spec.submodule_search_locations)
    files = {}
    for root in sorted(roots):
        for path in sorted(root.rglob("*.py")):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Source file escapes its package root")
            logical = path.relative_to(root).as_posix()
            actual = path.resolve()
            if logical in files and files[logical] != actual:
                raise ValueError(f"Conflicting implementations of package source: {logical}")
            files[logical] = actual
    return dict(sorted(files.items()))


def resolve_source(logical: str) -> Path:
    """Resolve a recorded logical path, rejecting traversal and unknown modules."""
    path = PurePosixPath(logical)
    if not isinstance(logical, str) or not logical or "\\" in logical or path.is_absolute() or any(part in {".", ".."} for part in logical.split("/")):
        raise ValueError("Expected a relative canonical package source path")
    try:
        return source_files()[logical]
    except KeyError as exc:
        raise FileNotFoundError(f"Unknown package source: {logical}") from exc
