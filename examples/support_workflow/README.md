# Support workflow example

Run this example from the repository root:

```bash
uv run jest-ai test examples/support_workflow --agent examples.support_workflow.agent:support_agent --report baseline.json
uv run jest-ai replay baseline.json baseline.json
```

The test is automatically discovered, produces a JSON report, and can be used
as the initial replay baseline. Replace the async function with an
`OpenAIResponsesAdapter` or another `AgentAdapter` in a real application.
