"""Filesystem discovery for decorator-based agent test modules."""

import hashlib
import importlib.util
from pathlib import Path

from jest_ai.api import clear_registered_tests, discover_tests

PATTERNS = ("agent_test_*.py", "test_agent_*.py")


def find_test_files(paths: list[str | Path]) -> list[Path]:
    """Find agent test files in supplied files or directories."""
    found: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            found.add(path.resolve())
        elif path.is_dir():
            for pattern in PATTERNS:
                found.update(candidate.resolve() for candidate in path.rglob(pattern))
    return sorted(found)


def discover_from_paths(paths: list[str | Path]):
    """Import agent test files and return only their fresh registrations."""
    clear_registered_tests()
    for path in find_test_files(paths):
        name = f"jest_ai_discovered_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import test file {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return discover_tests()
