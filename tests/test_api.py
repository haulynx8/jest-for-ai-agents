import asyncio

import pytest

from jest_ai import (
    clear_registered_tests,
    discover_tests,
    expect,
    run_registered_tests,
    test_agent,
)


@pytest.fixture(autouse=True)
def empty_registry():
    clear_registered_tests()
    yield
    clear_registered_tests()


async def echo_agent(prompt: str) -> str:
    return f"Agent response: {prompt}"


@pytest.mark.asyncio
async def test_decorator_registers_and_executes_an_agent_test():
    @test_agent("handles refunds", tags=["support"])
    async def test_refund(agent):
        response = await agent("I want a refund")
        expect(response).to_contain("refund")

    registered = discover_tests(__name__)
    results = await run_registered_tests(echo_agent)

    assert test_refund.__test__ is False
    assert [test.name for test in registered] == ["handles refunds"]
    assert registered[0].tags == ("support",)
    assert results[0].passed


@pytest.mark.asyncio
async def test_matcher_failure_becomes_a_structured_test_failure():
    @test_agent("rejects incorrect response")
    async def test_incorrect_response(agent):
        response = await agent("Hello")
        expect(response).to_contain("refund")

    result = (await run_registered_tests(echo_agent))[0]

    assert not result.passed
    assert result.error == "AssertionError: Expected 'refund' in output 'Agent response: Hello'"


@pytest.mark.asyncio
async def test_registered_tests_honor_concurrency_and_timeouts():
    active = 0
    peak_active = 0

    @test_agent("first")
    async def first(agent):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await agent("one")
        active -= 1

    @test_agent("second")
    async def second(agent):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await agent("two")
        active -= 1

    @test_agent("times out", timeout=0.01)
    async def slow_test(agent):
        await agent("slow")

    async def delayed_agent(_: str) -> str:
        await asyncio.sleep(0.05)
        return "done"

    results = await run_registered_tests(delayed_agent, concurrency=2)

    assert peak_active == 2
    assert [result.passed for result in results] == [True, True, False]
    assert results[2].error == "Timed out after 0.01s"


def test_test_agent_requires_async_function():
    with pytest.raises(TypeError, match="async def"):

        @test_agent("sync test")
        def sync_test(_agent):
            pass
