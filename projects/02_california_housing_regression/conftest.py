"""Make this project's `src` package importable when pytest runs.

pytest's rootdir is the repository root, but the code under test is a package
rooted HERE. Pytest automatically loads a conftest.py from the rootdir down to
each test file, so inserting the project folder onto sys.path here makes
`from src.data import ...` resolve no matter which directory you invoke pytest
from.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
