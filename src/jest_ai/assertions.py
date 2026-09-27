from typing import Any


class Expectation:
    """Chainable assertions for values returned by an agent."""

    def __init__(self, actual: Any) -> None:
        self.actual = actual

    def to_contain(self, expected: str) -> None:
        """Assert that a case-insensitive string representation contains text."""
        expect_contains(str(self.actual), expected)

    def not_to_contain(self, unexpected: str) -> None:
        """Assert that a case-insensitive string representation excludes text."""
        expect_not_contains(str(self.actual), unexpected)


def expect(output: Any) -> Expectation:
    """Start a Jest-like assertion chain for an agent response."""
    return Expectation(output)


def expect_contains(output: str, expected: str) -> None:
    assert expected.casefold() in output.casefold(), (
        f"Expected {expected!r} in output {output!r}"
    )


def expect_not_contains(output: str, unexpected: str) -> None:
    assert unexpected.casefold() not in output.casefold(), (
        f"Unexpected text {unexpected!r} found in output"
    )


def expect_tool_called(tool_calls: list[dict], tool_name: str) -> None:
    called = [call.get("name") for call in tool_calls]
    assert tool_name in called, (
        f"Expected tool {tool_name!r} to be called; actual calls: {called}"
    )
