"""Reusable agent fixtures and named suites."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from jest_ai.adapters import AgentAdapter
from jest_ai.case import AgentTest
from jest_ai.runner import TestResult, run_tests


@dataclass(frozen=True)
class AgentFixture:
    """A named adapter and defaults shared by a group of agent tests."""

    name: str
    adapter: AgentAdapter
    default_tags: tuple[str, ...] = field(default_factory=tuple)

    def test(self, name: str, prompt: str, **kwargs: object) -> AgentTest:
        """Create a case that uses this fixture's adapter and tags."""
        tags = [*self.default_tags, *kwargs.pop("tags", [])]  # type: ignore[arg-type]
        return AgentTest(name=name, agent=self.adapter, prompt=prompt, tags=tags, **kwargs)


@dataclass(frozen=True)
class AgentTestSuite:
    """A named, reusable collection of related cases."""

    name: str
    tests: tuple[AgentTest, ...]

    @classmethod
    def from_tests(cls, name: str, tests: Iterable[AgentTest]) -> "AgentTestSuite":
        return cls(name=name, tests=tuple(tests))

    async def run(self, concurrency: int = 5) -> list[TestResult]:
        return await run_tests(list(self.tests), concurrency=concurrency)
