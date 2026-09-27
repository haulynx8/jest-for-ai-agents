import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from jest_ai.adapters import AgentResponse, normalize_response, tool_call_dicts
from jest_ai.case import AgentTest
from jest_ai.evaluators import EvaluationContext, EvaluationResult, extract_tool_calls


@dataclass
class TestResult:
    """The isolated result of executing one agent test."""

    name: str
    passed: bool
    output: Any = None
    error: str | None = None
    duration: float = 0.0
    evaluations: list[EvaluationResult] = field(default_factory=list)
    score: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


async def run_test(test: AgentTest) -> TestResult:
    """Execute one test with its own timeout and failure boundary."""
    start = time.perf_counter()

    try:
        response = await asyncio.wait_for(_invoke_agent(test), timeout=test.timeout)
        output = response.output
        expected_passed = test.expected is None or test.expected.casefold() in str(output).casefold()
        evaluations = await _run_evaluators(test, output, tool_call_dicts(response.tool_calls))
        evaluators_passed = all(result.passed for result in evaluations)
        passed = expected_passed and evaluators_passed
        error = None
        if not expected_passed:
            error = "Expected text not found"
        elif not evaluators_passed:
            error = next(result.explanation for result in evaluations if not result.passed)
        return TestResult(
            name=test.name,
            passed=passed,
            output=output,
            error=error,
            duration=time.perf_counter() - start,
            evaluations=evaluations,
            score=_score(passed, evaluations),
            trace=[{"kind": event.kind, "data": event.data} for event in response.trace],
            tool_calls=tool_call_dicts(response.tool_calls),
        )
    except TimeoutError:
        return TestResult(
            name=test.name,
            passed=False,
            error=f"Timed out after {test.timeout}s",
            duration=time.perf_counter() - start,
            score=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - agent errors become test results.
        return TestResult(
            name=test.name,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
            duration=time.perf_counter() - start,
            score=0.0,
        )


async def run_tests(tests: list[AgentTest], concurrency: int = 5) -> list[TestResult]:
    """Execute tests concurrently while limiting active agent calls."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_run(test: AgentTest) -> TestResult:
        async with semaphore:
            return await run_test(test)

    return await asyncio.gather(*(bounded_run(test) for test in tests))


async def _invoke_agent(test: AgentTest) -> AgentResponse:
    agent = test.agent
    if hasattr(agent, "run"):
        return await agent.run(test.prompt)  # type: ignore[union-attr]
    return normalize_response(await agent(test.prompt))  # type: ignore[operator]


async def _run_evaluators(
    test: AgentTest, output: Any, tool_calls: list[dict[str, Any]]
) -> list[EvaluationResult]:
    context = EvaluationContext(
        prompt=test.prompt,
        output=output,
        test_name=test.name,
        tool_calls=tool_calls or extract_tool_calls(output),
    )
    results: list[EvaluationResult] = []
    for evaluator in test.evaluators:
        try:
            results.append(await evaluator.evaluate(context))
        except Exception as exc:  # noqa: BLE001 - evaluator errors are structured failures.
            results.append(
                EvaluationResult(
                    name=type(evaluator).__name__,
                    passed=False,
                    score=0.0,
                    explanation=f"Evaluator error: {type(exc).__name__}: {exc}",
                )
            )
    return results


def _score(passed: bool, evaluations: list[EvaluationResult]) -> float:
    if evaluations:
        return sum(result.score for result in evaluations) / len(evaluations)
    return 1.0 if passed else 0.0
