"""Resolve installed sources separately from user-owned generated artifacts."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("DEFAI_WORKSPACE", str(Path.cwd()))).expanduser().resolve()


def source_path(name: str) -> Path:
    """Find canonical sources without depending on an original project checkout."""
    if name == "api.py":
        return Path(__file__).resolve().parents[1] / "api.py"
    if name == "graph.py":
        from defai.federated import graph
        return Path(graph.__file__).resolve()
    if name == "learning.py":
        from defai.federated import learning
        return Path(learning.__file__).resolve()
    path = Path(__file__).with_name(name)
    if path.name != name or not path.is_file():
        raise FileNotFoundError(f"Unknown canonical study source: {name}")
    return path


def relative_path(path: Path) -> str:
    """Return a workspace-relative artifact ID or an absolute installed-source ID."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
