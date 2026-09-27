"""Async-first testing primitives for AI agents."""

from jest_ai.adapters import (
    AgentAdapter,
    AgentResponse,
    CallableAgentAdapter,
    ToolCall,
    TraceEvent,
)
from jest_ai.api import (
    RegisteredAgentTest,
    clear_registered_tests,
    discover_tests,
    run_registered_test,
    run_registered_tests,
    test_agent,
)
from jest_ai.assertions import expect
from jest_ai.case import AgentTest
from jest_ai.discovery import discover_from_paths, find_test_files
from jest_ai.evaluators import (
    CallableEvaluator,
    EvaluationContext,
    EvaluationResult,
    JsonSchemaEvaluator,
    PromptInjectionEvaluator,
    SensitiveDataLeakEvaluator,
    ToolCallEvaluator,
)
from jest_ai.openai import OpenAIResponsesAdapter
from jest_ai.replay import ReplayDifference, compare_replay, load_replay, save_replay
from jest_ai.reports import (
    build_report,
    format_ci_failures,
    format_terminal_report,
    write_json_report,
)
from jest_ai.runner import TestResult, run_test, run_tests
from jest_ai.suites import AgentFixture, AgentTestSuite

__all__ = [
    "AgentAdapter",
    "AgentFixture",
    "AgentResponse",
    "AgentTest",
    "AgentTestSuite",
    "CallableAgentAdapter",
    "CallableEvaluator",
    "EvaluationContext",
    "EvaluationResult",
    "JsonSchemaEvaluator",
    "OpenAIResponsesAdapter",
    "PromptInjectionEvaluator",
    "RegisteredAgentTest",
    "ReplayDifference",
    "SensitiveDataLeakEvaluator",
    "TestResult",
    "ToolCall",
    "ToolCallEvaluator",
    "TraceEvent",
    "build_report",
    "clear_registered_tests",
    "compare_replay",
    "discover_from_paths",
    "discover_tests",
    "expect",
    "find_test_files",
    "format_ci_failures",
    "format_terminal_report",
    "load_replay",
    "run_registered_test",
    "run_registered_tests",
    "run_test",
    "run_tests",
    "save_replay",
    "test_agent",
    "write_json_report",
]
