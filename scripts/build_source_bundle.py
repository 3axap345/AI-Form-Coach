"""Create a deterministic source bundle from files tracked by Git."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "dataset",
    "dist",
    "htmlcov",
    "models",
    "UI-PRMD",
    "venv",
    ".venv",
}


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_source_files() -> list[Path]:
    paths = _git_output("ls-files", "-z").split("\0")
    return [
        Path(path) for path in paths if path and not EXCLUDED_PARTS.intersection(Path(path).parts)
    ]


def main() -> Path:
    revision = _git_output("rev-parse", "--short", "HEAD")
    archive = DIST_DIR / f"ai-form-coach-{revision}.zip"
    DIST_DIR.mkdir(exist_ok=True)

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative_path in _tracked_source_files():
            source_path = PROJECT_ROOT / relative_path
            entry = zipfile.ZipInfo(relative_path.as_posix(), ARCHIVE_TIMESTAMP)
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, source_path.read_bytes())

    print(archive.relative_to(PROJECT_ROOT))
    return archive


if __name__ == "__main__":
    main()
