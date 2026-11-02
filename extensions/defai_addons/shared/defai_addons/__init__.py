"""Integrated deFAI package with component-owned source directories."""
from pathlib import Path
__version__ = "3.2.0"
_root = Path(__file__).resolve().parents[2]
for _name in ("01_visualization", "02_comparison", "03_multimodal"):
    _part = _root / _name / "defai_addons"
    if _part.is_dir():
        __path__.append(str(_part))
