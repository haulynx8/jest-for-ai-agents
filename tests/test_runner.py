import asyncio

import pytest

from examples.basic_agent import support_agent
from jest_ai.case import AgentTest
from jest_ai.runner import run_test, run_tests


@pytest.mark.asyncio
async def test_agent_returns_expected_response():
    test = AgentTest(
        name="refund request",
        agent=support_agent,
        prompt="I want a refund",
        expected="initiate a refund",
    )
    result = await run_test(test)
    assert result.passed
    assert result.duration > 0


@pytest.mark.asyncio
async def test_agent_fails_on_wrong_expectation():
    test = AgentTest(
        name="incorrect expectation",
        agent=support_agent,
        prompt="Hello",
        expected="refund",
    )
    result = await run_test(test)
    assert not result.passed
    assert result.error == "Expected text not found"


@pytest.mark.asyncio
async def test_multiple_agents_run():
    tests = [
        AgentTest(
            name=f"test-{index}",
            agent=support_agent,
            prompt="I need a refund",
            expected="refund",
        )
        for index in range(10)
    ]
    results = await run_tests(tests, concurrency=3)
    assert len(results) == 10
    assert all(result.passed for result in results)


@pytest.mark.asyncio
async def test_agent_timeout_is_an_isolated_failure():
    async def slow_agent(_: str) -> str:
        await asyncio.sleep(0.1)
        return "finished"

    result = await run_test(
        AgentTest(name="timeout", agent=slow_agent, prompt="Hello", timeout=0.01)
    )

    assert not result.passed
    assert result.error == "Timed out after 0.01s"


@pytest.mark.asyncio
async def test_agent_exception_is_captured_as_a_failure():
    async def broken_agent(_: str) -> str:
        raise RuntimeError("service unavailable")

    result = await run_test(AgentTest(name="broken", agent=broken_agent, prompt="Hello"))

    assert not result.passed
    assert result.error == "RuntimeError: service unavailable"
