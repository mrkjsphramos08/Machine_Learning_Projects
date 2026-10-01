"""Make this project's `src` package importable when pytest runs.

pytest's rootdir is the repository root, but the code under test is a package
rooted HERE. Pytest automatically loads a conftest.py from the rootdir down to
each test file, so inserting the project folder onto sys.path here makes
`from src.data import ...` resolve no matter which directory you invoke pytest
from.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Evict any 'src' modules cached from another project in the monorepo
for mod in list(sys.modules):
    if mod == "src" or mod.startswith("src."):
        del sys.modules[mod]


@pytest.fixture(autouse=True)
def ensure_project_src():
    """Ensure this project's src is always active during test execution."""
    if str(PROJECT_ROOT) in sys.path:
        sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))
    yield
