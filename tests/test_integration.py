import json

import pytest

from jest_ai import (
    AgentFixture,
    AgentResponse,
    AgentTestSuite,
    ToolCall,
    ToolCallEvaluator,
    TraceEvent,
    compare_replay,
    format_ci_failures,
    format_terminal_report,
    load_replay,
    run_test,
    save_replay,
)


class TracedAdapter:
    async def run(self, prompt: str) -> AgentResponse:
        return AgentResponse(
            output=f"Completed: {prompt}",
            tool_calls=[ToolCall("lookup_order", {"order_id": "A-1"}, result={"found": True})],
            trace=[TraceEvent("agent_started"), TraceEvent("tool_completed", {"name": "lookup_order"})],
        )


@pytest.mark.asyncio
async def test_adapter_trace_and_tool_outcome_reach_results_and_evaluators():
    fixture = AgentFixture("orders", TracedAdapter(), default_tags=("integration",))
    case = fixture.test(
        "lookup",
        "Find A-1",
        evaluators=[ToolCallEvaluator(required={"lookup_order": {"order_id": "A-1"}})],
    )

    result = await run_test(case)

    assert result.passed
    assert case.tags == ["integration"]
    assert result.tool_calls[0]["result"] == {"found": True}
    assert result.trace[1]["kind"] == "tool_completed"


@pytest.mark.asyncio
async def test_fixture_cases_can_be_grouped_and_run_as_a_suite():
    fixture = AgentFixture("orders", TracedAdapter())
    suite = AgentTestSuite.from_tests("orders", [fixture.test("one", "first"), fixture.test("two", "second")])

    results = await suite.run(concurrency=2)

    assert [result.name for result in results] == ["one", "two"]
    assert all(result.passed for result in results)


@pytest.mark.asyncio
async def test_replay_comparison_and_ci_output_are_reproducible(tmp_path):
    fixture = AgentFixture("orders", TracedAdapter())
    result = await run_test(fixture.test("lookup", "Find A-1"))
    baseline_path = tmp_path / "baseline.json"
    save_replay([result], baseline_path)

    baseline = load_replay(baseline_path)
    changed = type(result)(**{**result.__dict__, "output": "Changed"})
    differences = compare_replay(baseline, [changed])

    assert json.loads(baseline_path.read_text())["summary"]["total"] == 1
    assert differences[0].kind == "output"
    assert "PASS lookup" in format_terminal_report([result])
    failed = type(result)(**{**result.__dict__, "passed": False, "error": "Regression"})
    assert "::error" in format_ci_failures([failed])
