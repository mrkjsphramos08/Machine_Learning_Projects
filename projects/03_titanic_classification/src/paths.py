"""
Path resolution for this project.

Every path in this project is computed from THIS file's location on disk, never
from the current working directory (CWD). That is what makes the pipeline behave
identically whether you run it from the repo root, from the project folder, or
from DVC (which sets its own working directory).

Directory layout assumed by this module:
    <repo_root>/projects/<project_name>/src/paths.py
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # .../projects/<project_name>
REPO_ROOT: Path = PROJECT_ROOT.parents[1]  # .../MLOps (shared venv, mlflow.db, .dvc)


def resolve(relative_path: str) -> Path:
    """Resolve a config path against the project root.

    Absolute paths are returned untouched, so configs may use either
    project-relative or absolute paths.
    """
    return PROJECT_ROOT / relative_path


def ensure_parent(path: Path) -> Path:
    """Create the parent directory of `path` if it does not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
