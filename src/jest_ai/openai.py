"""Concrete adapter for the OpenAI Responses API."""

import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from jest_ai.adapters import AgentResponse, ToolCall, TraceEvent


@dataclass
class OpenAIResponsesAdapter:
    """Run a prompt through Responses and retain function-call traces."""

    model: str
    client: Any = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None

    async def run(self, prompt: str) -> AgentResponse:
        client = self.client or AsyncOpenAI()
        response = await client.responses.create(
            model=self.model,
            input=prompt,
            instructions=self.instructions,
            tools=self.tools or [],
        )
        tool_calls = [_function_call(item) for item in getattr(response, "output", [])]
        calls = [call for call in tool_calls if call is not None]
        return AgentResponse(
            output=getattr(response, "output_text", ""),
            tool_calls=calls,
            trace=[
                TraceEvent("response_created", {"id": getattr(response, "id", None)}),
                *[
                    TraceEvent("tool_called", {"name": call.name, "call_id": call.call_id})
                    for call in calls
                ],
            ],
            metadata={"model": self.model, "response_id": getattr(response, "id", None)},
        )


def _function_call(item: Any) -> ToolCall | None:
    if getattr(item, "type", None) != "function_call":
        return None
    arguments = getattr(item, "arguments", "{}")
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except (TypeError, json.JSONDecodeError):
        parsed = {"_raw": arguments}
    return ToolCall(
        name=getattr(item, "name", "unknown"),
        arguments=parsed,
        call_id=getattr(item, "call_id", None),
    )
