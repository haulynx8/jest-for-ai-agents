import json
import os
import subprocess
import sys

import pytest

from jest_ai import (
    OpenAIResponsesAdapter,
    ToolCall,
    compare_replay,
    discover_from_paths,
    find_test_files,
    run_test,
)
from jest_ai.cli import compare_report_replay


class FakeResponses:
    async def create(self, **kwargs):
        self.request = kwargs
        call = type("Call", (), {"type": "function_call", "name": "lookup", "arguments": '{"id":"A-1"}', "call_id": "call_1"})()
        return type("Response", (), {"id": "resp_1", "output_text": "Found A-1", "output": [call]})()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


@pytest.mark.asyncio
async def test_openai_responses_adapter_normalizes_responses_api_shape_without_network():
    client = FakeClient()
    response = await OpenAIResponsesAdapter("gpt-5", client=client, instructions="Be concise").run("Find A-1")

    assert response.output == "Found A-1"
    assert response.tool_calls == [ToolCall("lookup", {"id": "A-1"}, call_id="call_1")]
    assert client.responses.request["model"] == "gpt-5"
    assert response.trace[1].data == {"name": "lookup", "call_id": "call_1"}


@pytest.mark.asyncio
async def test_openai_adapter_handles_multiple_calls_malformed_arguments_and_api_errors():
    valid = type("Call", (), {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "one"})()
    malformed = type("Call", (), {"type": "function_call", "name": "create", "arguments": "{", "call_id": "two"})()
    response = type("Response", (), {"id": "resp", "output_text": "", "output": [valid, malformed]})()
    client = FakeClient()
    client.responses.create = lambda **_: _async_value(response)

    normalized = await OpenAIResponsesAdapter("gpt-5", client=client).run("tools")

    assert [call.call_id for call in normalized.tool_calls] == ["one", "two"]
    assert normalized.tool_calls[1].arguments == {"_raw": "{"}

    async def broken_create(**_):
        raise RuntimeError("API unavailable")

    client.responses.create = broken_create
    from jest_ai import AgentTest
    result = await run_test(AgentTest("api error", OpenAIResponsesAdapter("gpt-5", client=client), "hello"))
    assert result.error == "RuntimeError: API unavailable"


def test_filesystem_discovery_imports_registered_agent_tests(tmp_path):
    test_file = tmp_path / "agent_test_refunds.py"
    test_file.write_text("from jest_ai import test_agent\n@test_agent('refund')\nasync def scenario(agent):\n    await agent('refund')\n")

    tests = discover_from_paths([tmp_path])

    assert find_test_files([tmp_path]) == [test_file]
    assert [test.name for test in tests] == ["refund"]


def test_cli_discovers_runs_reports_and_detects_a_replay_regression(tmp_path):
    (tmp_path / "sample_agent.py").write_text("async def agent(prompt):\n    return 'approved'\n")
    tests_dir = tmp_path / "agent_tests"
    tests_dir.mkdir()
    (tests_dir / "agent_test_approval.py").write_text(
        "from jest_ai import expect, test_agent\n"
        "@test_agent('approves request')\n"
        "async def approval(agent):\n"
        "    expect(await agent('approve')).to_contain('approved')\n"
    )
    report = tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "jest_ai.cli",
        "test",
        str(tests_dir),
        "--agent",
        "sample_agent:agent",
        "--report",
        str(report),
    ]
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}

    passed = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    assert passed.returncode == 0
    assert json.loads(report.read_text())["summary"]["passed"] == 1

    (tmp_path / "sample_agent.py").write_text("async def agent(prompt):\n    return 'denied'\n")
    failed_report = tmp_path / "failed-report.json"
    failed = subprocess.run(
        [*command[:-1], str(failed_report)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    replay = subprocess.run(
        [sys.executable, "-m", "jest_ai.cli", "replay", str(report), str(failed_report)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert replay.returncode == 1
    assert "passed" in replay.stdout


@pytest.mark.asyncio
async def test_replay_detects_tool_changes_and_removed_cases(tmp_path):
    async def agent(_: str):
        return {"tool_calls": [{"name": "lookup"}]}

    from jest_ai import AgentTest
    result = await run_test(AgentTest("case", agent, "hi"))
    baseline = {"results": [{**result.__dict__, "evaluations": []}]}
    changed = type(result)(**{**result.__dict__, "tool_calls": []})

    assert compare_replay(baseline, [changed])[0].kind == "tool_calls"
    assert compare_replay(baseline, [])[0].kind == "removed"
    current = {"summary": {}, "results": [{**result.__dict__, "tool_calls": []}]}
    assert compare_report_replay(baseline, current)[0]["kind"] == "tool_calls"
    assert json.dumps(current, default=str)


async def _async_value(value):
    return value
