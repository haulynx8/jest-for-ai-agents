# jest-for-ai-agents

Async-first Python testing for AI agents: isolated runs, Jest-like scenarios,
pluggable evaluators, traces, replay baselines, reports, a CLI, and CI support.

> Status: **0.1.0 alpha**. APIs may change before 1.0.0. The package is locally
> verified, including clean-environment wheel installation; live provider and
> external GitHub Actions validation remain outstanding.

## Features

| Area | Included |
| --- | --- |
| Runner | Async execution, configurable concurrency, per-test timeouts, structured failures, and durations |
| Test API | `AgentTest`, `@test_agent`, discovery, `expect(...).to_contain()`, and `.not_to_contain()` |
| Evaluators | Exact text, JSON Schema, tool policy, injection/leak regression checks, and custom semantic judges |
| Integrations | Async functions, `AgentAdapter`, fixtures/suites, and OpenAI Responses |
| Observability | Output, score, evaluator explanations, tool calls/results/errors, and ordered traces |
| DX / CI | CLI, JSON/terminal reports, GitHub annotations, replay diffs, and a workflow template |

## Requirements and installation

Python 3.11+ is required. [uv](https://docs.astral.sh/uv/) is recommended.

```bash
git clone https://github.com/haulynx/jest-for-ai-agents.git
cd jest-for-ai-agents
uv sync --dev
uv run pytest -v
```

After PyPI publication:

```bash
uv add jest-for-ai-agents
```

Install a locally built artifact instead:

```bash
uv build
uv pip install dist/jest_for_ai_agents-0.1.0-py3-none-any.whl
```

## Direct test cases

Any `async def agent(prompt: str)` function works:

```python
from jest_ai import AgentTest, run_test


async def support_agent(prompt: str) -> str:
    return "I can help you initiate a refund."


test = AgentTest(
    name="refund request",
    agent=support_agent,
    prompt="I want a refund",
    expected="initiate a refund",
)
result = await run_test(test)
assert result.passed
```

Use `run_tests(tests, concurrency=5)` for a group. Results contain `name`,
`passed`, `output`, `error`, `duration`, `score`, `evaluations`, `tool_calls`,
and `trace`. Agent timeouts, exceptions, and assertion/evaluation failures are
captured as failed results rather than aborting the entire suite.

## Jest-like scenario API

```python
from jest_ai import expect, run_registered_tests, test_agent


@test_agent("handles refunds", timeout=10, tags=["support", "regression"])
async def handles_refunds(agent):
    response = await agent("I want a refund")
    expect(response).to_contain("refund")
    expect(response).not_to_contain("cannot help")


results = await run_registered_tests(support_agent)
```

`discover_tests()` returns registered scenarios; `clear_registered_tests()` is
available for isolated programmatic environments. Decorated functions must use
`async def` and are intentionally excluded from pytest collection.

## Evaluation engine

Attach evaluators to an `AgentTest`. A test passes only when its exact-text
expectation and every evaluator pass. Every evaluation provides pass/fail,
score, explanation, and optional details.

```python
from jest_ai import AgentTest, JsonSchemaEvaluator, ToolCallEvaluator

test = AgentTest(
    name="creates an order",
    agent=my_agent,
    prompt="Create an order for item A-1",
    evaluators=[
        JsonSchemaEvaluator({"type": "object", "required": ["order_id"]}),
        ToolCallEvaluator(
            required={"create_order": {"item_id": "A-1"}},
            allowed_tools=frozenset({"create_order", "lookup_catalog"}),
        ),
    ],
)
```

Built-ins:

- `JsonSchemaEvaluator` validates JSON strings or objects against JSON Schema.
- `ToolCallEvaluator` requires calls/partial arguments and can enforce an
  allowed-tool set.
- `PromptInjectionEvaluator` checks configured refusal markers.
- `SensitiveDataLeakEvaluator` checks caller-configured sensitive strings.
- `CallableEvaluator` accepts an async custom or model-backed semantic judge;
  it keeps model provider selection in the application.

```python
from jest_ai import CallableEvaluator, EvaluationResult


async def semantic_refund_judge(context):
    passed = "refund" in str(context.output).casefold()
    return EvaluationResult("semantic_refund", passed, float(passed), "Custom judge result.")


semantic_evaluator = CallableEvaluator("semantic_refund", semantic_refund_judge)
```

### Security limitation

The security evaluators are lightweight deterministic regression checks, **not
security certification**. Refusal markers can yield false positives and
configured-string checks cannot detect every data leak. Use them alongside
threat modeling, adversarial suites, authorization controls, review, and
monitoring.

## Adapters, tool calls, and traces

Implement `AgentAdapter.run(prompt) -> AgentResponse` for another framework.
`AgentResponse` contains final `output`, `tool_calls`, ordered `trace`, and
arbitrary `metadata`. Each `ToolCall` has `name`, `arguments`, provider
`call_id`, `result`, and actual `error`; a call ID is never stored as an error.

`CallableAgentAdapter` adapts ordinary async functions. Raw dictionaries with
`output` or `tool_calls` are normalized too, so tool activity reaches evaluator
contexts and reports.

### OpenAI Responses adapter

```python
from jest_ai import OpenAIResponsesAdapter

agent = OpenAIResponsesAdapter(model="gpt-5")
```

The official async OpenAI SDK adapter normalizes `output_text`, records function
calls, and preserves provider call IDs. Set `OPENAI_API_KEY` before live calls.
It has mock coverage for multiple calls, malformed arguments, and API errors;
no live API run has occurred here because no key is configured.

## Fixtures and suites

```python
from jest_ai import AgentFixture, AgentTestSuite

fixture = AgentFixture("support", support_adapter, default_tags=("support",))
suite = AgentTestSuite.from_tests(
    "support regression",
    [
        fixture.test("refund", "I need a refund", expected="refund"),
        fixture.test("greeting", "Hello", expected="help"),
    ],
)
results = await suite.run(concurrency=2)
```

## CLI, discovery, and reports

The CLI finds `agent_test_*.py` and `test_agent_*.py` files recursively.
Agent targets use `package.module:attribute` syntax.

```bash
jest-ai test agent_tests --agent my_project.agents:support_agent --report report.json
jest-ai report report.json
jest-ai replay baseline.json report.json
```

`jest-ai test` writes JSON, prints readable results, emits GitHub Actions error
annotations for failures, and exits 1 when a test fails. `jest-ai replay` exits
1 when it finds a difference. Programmatic report helpers are `build_report`,
`format_terminal_report`, `format_ci_failures`, and `write_json_report`.

## Replay regression detection

```python
from jest_ai import compare_replay, load_replay, save_replay

save_replay(results, "baseline.json")
differences = compare_replay(load_replay("baseline.json"), new_results)
```

Both library and CLI comparisons identify new/removed tests plus changes to
pass state, score, output, tool calls, traces, and evaluator results.

## CI setup

Copy [`.github/workflows/agent-tests.yml`](.github/workflows/agent-tests.yml)
to the consuming repository, replace `agent_tests` and the agent target, and
configure provider credentials as repository secrets. The workflow runs tests,
invokes the CLI, and uses `if: always()` to upload JSON reports after failures.

The template has been inspected locally, but not yet executed in an external
GitHub Actions repository.

## Runnable example

The complete example is in
[`examples/support_workflow`](examples/support_workflow/README.md):

```bash
uv run jest-ai test examples/support_workflow \
  --agent examples.support_workflow.agent:support_agent \
  --report baseline.json
uv run jest-ai replay baseline.json baseline.json
```

The built wheel has been installed in a fresh temporary environment and used to
discover and run this example without importing the development checkout.

## Validation and remaining release work

Current local validation:

- 22 automated tests passing
- Ruff checks passing
- source distribution and wheel builds passing
- CLI help and subprocess-based discovery/execution/report/replay tests passing
- clean-environment wheel installation and runnable-example execution passing
- mock-only OpenAI Responses adapter coverage passing

Before presenting this as release-ready, still:

- run a controlled live OpenAI Responses integration with an explicitly
  supplied test API key
- run the workflow in an external GitHub repository and verify artifact upload
- publish to PyPI only after those external checks

## License and releases

Licensed under the [MIT License](LICENSE). See [CHANGELOG.md](CHANGELOG.md) for
release notes and the alpha compatibility policy.
