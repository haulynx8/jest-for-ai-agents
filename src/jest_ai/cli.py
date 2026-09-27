"""Command-line workflow for discovered agent suites and replay artifacts."""

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

from jest_ai.api import run_registered_tests
from jest_ai.discovery import discover_from_paths
from jest_ai.replay import load_replay
from jest_ai.reports import (
    format_ci_failures,
    format_terminal_report,
    write_json_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="jest-ai")
    commands = parser.add_subparsers(dest="command", required=True)
    test_command = commands.add_parser("test", help="Discover and run agent tests")
    test_command.add_argument("paths", nargs="+", help="Test files or directories")
    test_command.add_argument("--agent", required=True, help="Agent target: package.module:attribute")
    test_command.add_argument("--report", default="jest-ai-report.json")
    test_command.add_argument("--concurrency", type=int, default=5)
    replay_command = commands.add_parser("replay", help="Compare a report against a saved baseline")
    replay_command.add_argument("baseline")
    replay_command.add_argument("current")
    report_command = commands.add_parser("report", help="Render a JSON report")
    report_command.add_argument("report")
    args = parser.parse_args()
    if args.command == "test":
        raise SystemExit(asyncio.run(_test(args)))
    if args.command == "replay":
        raise SystemExit(_replay(args))
    raise SystemExit(_report(args))


async def _test(args: argparse.Namespace) -> int:
    tests = discover_from_paths(args.paths)
    results = await run_registered_tests(_load_target(args.agent), tests=tests, concurrency=args.concurrency)
    write_json_report(results, args.report)
    print(format_terminal_report(results))
    if failures := format_ci_failures(results):
        print(failures)
    return 0 if all(result.passed for result in results) else 1


def _replay(args: argparse.Namespace) -> int:
    differences = compare_report_replay(load_replay(args.baseline), json.loads(Path(args.current).read_text()))
    for difference in differences:
        print(f"{difference['name']}: {difference['kind']}: {difference['baseline']!r} -> {difference['current']!r}")
    return 0 if not differences else 1


def _report(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text())
    print(f"{report['summary']['passed']}/{report['summary']['total']} passed")
    for result in report["results"]:
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['name']}")
    return 0


def _load_target(target: str) -> Any:
    module_name, separator, attribute = target.partition(":")
    if not separator or not attribute:
        raise ValueError("--agent must use package.module:attribute syntax")
    return getattr(importlib.import_module(module_name), attribute)


def compare_report_replay(baseline: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare two serialized reports for CLI use."""
    previous = {result["name"]: result for result in baseline.get("results", [])}
    latest = {result["name"]: result for result in current.get("results", [])}
    differences = []
    for name, result in latest.items():
        if name not in previous:
            differences.append({"name": name, "kind": "new", "baseline": None, "current": result})
            continue
        for key in ("passed", "score", "output", "tool_calls", "trace", "evaluations"):
            if previous[name].get(key) != result.get(key):
                differences.append(
                    {"name": name, "kind": key, "baseline": previous[name].get(key), "current": result.get(key)}
                )
    for name, result in previous.items():
        if name not in latest:
            differences.append({"name": name, "kind": "removed", "baseline": result, "current": None})
    return differences


if __name__ == "__main__":
    main()
