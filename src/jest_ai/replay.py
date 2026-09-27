"""Save reproducible results and compare a new agent version against them."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jest_ai.reports import build_report
from jest_ai.runner import TestResult


@dataclass(frozen=True)
class ReplayDifference:
    name: str
    kind: str
    baseline: Any
    current: Any


def save_replay(results: list[TestResult], path: str | Path) -> None:
    """Persist deterministic run output as a replay baseline."""
    Path(path).write_text(json.dumps(build_report(results), indent=2, default=str) + "\n")


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load a baseline written by :func:`save_replay`."""
    return json.loads(Path(path).read_text())


def compare_replay(baseline: dict[str, Any], current: list[TestResult]) -> list[ReplayDifference]:
    """Identify pass/fail, score, and output changes by test name."""
    previous = {result["name"]: result for result in baseline["results"]}
    current_names = {result.name for result in current}
    differences: list[ReplayDifference] = []
    for result in current:
        old = previous.get(result.name)
        if old is None:
            differences.append(ReplayDifference(result.name, "new", None, result.passed))
            continue
        if old["passed"] != result.passed:
            differences.append(ReplayDifference(result.name, "passed", old["passed"], result.passed))
        if old["score"] != result.score:
            differences.append(ReplayDifference(result.name, "score", old["score"], result.score))
        if old.get("output") != result.output:
            differences.append(ReplayDifference(result.name, "output", old.get("output"), result.output))
        if old.get("tool_calls", []) != result.tool_calls:
            differences.append(ReplayDifference(result.name, "tool_calls", old.get("tool_calls", []), result.tool_calls))
        if old.get("trace", []) != result.trace:
            differences.append(ReplayDifference(result.name, "trace", old.get("trace", []), result.trace))
        current_evaluations = [evaluation.__dict__ for evaluation in result.evaluations]
        if old.get("evaluations", []) != current_evaluations:
            differences.append(ReplayDifference(result.name, "evaluations", old.get("evaluations", []), current_evaluations))
    for name, result in previous.items():
        if name not in current_names:
            differences.append(ReplayDifference(name, "removed", result, None))
    return differences
