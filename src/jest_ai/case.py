from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jest_ai.evaluators import Evaluator

Agent = Callable[[str], Awaitable[Any]] | Any


@dataclass
class AgentTest:
    """A single interaction and its expected outcome."""

    name: str
    agent: Agent
    prompt: str
    expected: str | None = None
    timeout: float = 30.0
    tags: list[str] = field(default_factory=list)
    evaluators: list["Evaluator"] = field(default_factory=list)
