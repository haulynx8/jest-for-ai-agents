"""Jest-like registration, discovery, and execution for agent tests."""

import asyncio
import importlib
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from jest_ai.adapters import AgentResponse
from jest_ai.case import Agent
from jest_ai.runner import TestResult

AgentTestFunction = Callable[[Agent], Awaitable[Any]]


@dataclass(frozen=True)
class RegisteredAgentTest:
    """A decorated agent test retained for later execution."""

    name: str
    function: AgentTestFunction
    timeout: float = 30.0
    tags: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: list[RegisteredAgentTest] = []


def test_agent(
    name: str,
    *,
    timeout: float = 30.0,
    tags: list[str] | None = None,
) -> Callable[[AgentTestFunction], AgentTestFunction]:
    """Register an async test function that receives the configured agent.

    Decorated functions are intentionally hidden from pytest collection; execute
    them through :func:`run_registered_tests` after supplying an agent adapter.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")

    def register(function: AgentTestFunction) -> AgentTestFunction:
        if not inspect.iscoroutinefunction(function):
            raise TypeError("test_agent functions must be declared with 'async def'")

        _REGISTRY.append(
            RegisteredAgentTest(
                name=name,
                function=function,
                timeout=timeout,
                tags=tuple(tags or []),
            )
        )
        function.__test__ = False
        return function

    return register


test_agent.__test__ = False


def discover_tests(module: ModuleType | str | None = None) -> list[RegisteredAgentTest]:
    """Return registered tests, importing and filtering a module when requested."""
    if isinstance(module, str):
        module = importlib.import_module(module)

    if module is None:
        return list(_REGISTRY)

    return [test for test in _REGISTRY if test.function.__module__ == module.__name__]


def clear_registered_tests() -> None:
    """Clear the registration store, primarily for isolated test environments."""
    _REGISTRY.clear()


async def run_registered_test(test: RegisteredAgentTest, agent: Agent) -> TestResult:
    """Run one decorated test with an isolated timeout and failure boundary."""
    start = time.perf_counter()
    try:
        output = await asyncio.wait_for(test.function(_agent_function(agent)), timeout=test.timeout)
        return TestResult(
            name=test.name,
            passed=True,
            output=output,
            duration=time.perf_counter() - start,
        )
    except TimeoutError:
        return TestResult(
            name=test.name,
            passed=False,
            error=f"Timed out after {test.timeout}s",
            duration=time.perf_counter() - start,
        )
    except Exception as exc:  # noqa: BLE001 - test errors are structured failures.
        return TestResult(
            name=test.name,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
            duration=time.perf_counter() - start,
        )


async def run_registered_tests(
    agent: Agent,
    *,
    tests: list[RegisteredAgentTest] | None = None,
    concurrency: int = 5,
) -> list[TestResult]:
    """Run discovered decorated tests concurrently against one agent adapter."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    selected_tests = discover_tests() if tests is None else tests
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_run(test: RegisteredAgentTest) -> TestResult:
        async with semaphore:
            return await run_registered_test(test, agent)

    return await asyncio.gather(*(bounded_run(test) for test in selected_tests))


def _agent_function(agent: Agent):
    async def invoke(prompt: str):
        if hasattr(agent, "run"):
            response: AgentResponse = await agent.run(prompt)  # type: ignore[union-attr]
            return response.output
        return await agent(prompt)  # type: ignore[operator]

    return invoke
