"""Standard adapters and execution traces for agents from any framework."""

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation, including its observed outcome."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    result: Any = None
    error: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    """An ordered event captured by an adapter during an agent invocation."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResponse:
    """Normalized agent output plus tool calls and an optional execution trace."""

    output: Any
    tool_calls: list[ToolCall] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(Protocol):
    """Framework-neutral agent contract used by the runner."""

    async def run(self, prompt: str) -> AgentResponse: ...


@dataclass(frozen=True)
class CallableAgentAdapter:
    """Adapt the MVP's `async def agent(prompt)` contract to an AgentAdapter."""

    agent: Callable[[str], Awaitable[Any]]
    name: str = "callable_agent"

    async def run(self, prompt: str) -> AgentResponse:
        result = await self.agent(prompt)
        return normalize_response(result)


def normalize_response(value: Any) -> AgentResponse:
    """Normalize common adapter/framework response shapes without provider imports."""
    if isinstance(value, AgentResponse):
        return value
    if isinstance(value, dict) and ("output" in value or "tool_calls" in value):
        return AgentResponse(
            output=value.get("output", value.get("text", value)),
            tool_calls=[_tool_call(call) for call in value.get("tool_calls", [])],
            trace=[_trace_event(event) for event in value.get("trace", [])],
            metadata=dict(value.get("metadata", {})),
        )
    return AgentResponse(output=value)


def tool_call_dicts(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Convert trace records to stable mappings used by evaluators and reports."""
    return [asdict(tool_call) for tool_call in tool_calls]


def _tool_call(value: ToolCall | dict[str, Any]) -> ToolCall:
    return value if isinstance(value, ToolCall) else ToolCall(**value)


def _trace_event(value: TraceEvent | dict[str, Any]) -> TraceEvent:
    return value if isinstance(value, TraceEvent) else TraceEvent(**value)
