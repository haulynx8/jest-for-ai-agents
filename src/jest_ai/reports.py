"""Serializable evaluation reports for local runs and CI artifacts."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jest_ai.runner import TestResult


def build_report(results: list[TestResult]) -> dict[str, Any]:
    """Build a stable, JSON-serializable summary of a test run."""
    return {
        "summary": {
            "total": len(results),
            "passed": sum(result.passed for result in results),
            "failed": sum(not result.passed for result in results),
        },
        "results": [asdict(result) for result in results],
    }


def write_json_report(results: list[TestResult], path: str | Path) -> None:
    """Write a report suitable for CI artifacts and regression comparisons."""
    Path(path).write_text(json.dumps(build_report(results), indent=2, default=str) + "\n")


def format_terminal_report(results: list[TestResult]) -> str:
    """Render a compact, readable result table for terminals and CI logs."""
    lines = [
        f"{'PASS' if result.passed else 'FAIL'} {result.name} ({result.duration:.3f}s)"
        for result in results
    ]
    for result in results:
        if result.error:
            lines.append(f"  {result.error}")
    summary = build_report(results)["summary"]
    lines.append(f"{summary['passed']}/{summary['total']} passed")
    return "\n".join(lines)


def format_ci_failures(results: list[TestResult]) -> str:
    """Render GitHub Actions-compatible error annotations for failed cases."""
    return "\n".join(
        f"::error title=Agent test failed::{result.name}: {result.error or 'Unknown failure'}"
        for result in results
        if not result.passed
    )
