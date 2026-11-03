import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SMALL = {"dataset": "synthetic", "max-samples": 1200, "local-epochs": 1.0}


@pytest.fixture
def small():
    return dict(SMALL)


def kist_ready():
    from fl_core.backends import kist_available

    return kist_available()
