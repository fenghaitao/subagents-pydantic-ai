"""Tests for the auto-retry layer (`subagents_pydantic_ai.retry`).

Covers transient-error classification, backoff computation, the
`RetryConfig` resolution, the `run_with_retry` driver (both the legacy
fast path and the `iter()`-based resume-with-history path), and the
`_run_async` integration that surfaces retries on the task handle.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ProcessEventStream
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_graph import End

from subagents_pydantic_ai import (
    InMemoryMessageBus,
    RetryConfig,
    TaskManager,
    TaskStatus,
    compute_backoff_delay,
    is_transient_error,
    run_with_retry,
)
from subagents_pydantic_ai.toolset import _run_async
from subagents_pydantic_ai.types import AgentMessage, MessageType, SubAgentConfig

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeResult:
    """Stand-in for `AgentRunResult`."""

    def __init__(self, output: str, usage: Any = None) -> None:
        self.output = output
        self._usage = usage

    @property
    def usage(self) -> Any:
        # pydantic-ai 2.0: `AgentRunResult.usage` is a property, not a method.
        return self._usage


class _ScriptedRun:
    """Stand-in for `AgentRun` driven the way `Agent.run` drives it.

    `_drive_run` (non-streaming path) advances via `run.next(node)` so the
    node hooks fire, exactly like `Agent.run`. We model that here instead of
    the bare `async for` protocol: a success step pre-sets `result` (the
    drive loop breaks before calling `next`), a failing step raises its
    `iter_raise` exception from `next`.
    """

    def __init__(self, step: dict[str, Any]) -> None:
        self._step = step
        self.result = step.get("result")
        self._raised = False
        # Non-`End` sentinel so the drive loop body runs at least once.
        self.next_node: Any = object()

    async def next(self, node: Any) -> Any:
        if "iter_raise" in self._step and not self._raised:
            self._raised = True
            raise self._step["iter_raise"]
        return End(self.result)

    def all_messages(self) -> list[Any]:
        messages: list[Any] = self._step.get("messages", [])
        return messages


class _ScriptedCM:
    def __init__(self, step: dict[str, Any]) -> None:
        self._step = step

    async def __aenter__(self) -> _ScriptedRun:
        if "aenter_raise" in self._step:
            raise self._step["aenter_raise"]
        return _ScriptedRun(self._step)

    async def __aexit__(self, *exc: object) -> bool:
        return False


class ScriptedAgent:
    """Agent fake driven by a list of per-attempt step dicts.

    Each step may contain `result` (success), `iter_raise` (raise
    while iterating, with optional `messages`), `aenter_raise` (raise
    before yielding a run), or `run_raise` (legacy `run()` path).
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        self._steps = list(steps)
        self.iter_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []

    def iter(self, prompt: Any, *, message_history: Any = None, **kwargs: Any) -> _ScriptedCM:
        self.iter_calls.append({"prompt": prompt, "message_history": message_history, **kwargs})
        return _ScriptedCM(self._steps.pop(0))

    async def run(self, prompt: Any, **kwargs: Any) -> Any:
        self.run_calls.append({"prompt": prompt, "kwargs": kwargs})
        step = self._steps.pop(0)
        if "run_raise" in step:
            raise step["run_raise"]
        return step["result"]


class FakeDeps:
    """Minimal deps object `_run_async` can attach state to."""


async def _no_sleep(_: float) -> None:
    return None


# --------------------------------------------------------------------------- #
# is_transient_error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ModelHTTPError(503, "m"), True),
        (ModelHTTPError(429, "m"), True),
        (ModelHTTPError(500, "m"), True),
        (ModelHTTPError(400, "m"), False),
        (ModelHTTPError(401, "m"), False),
        (ModelHTTPError(404, "m"), False),
        (ModelAPIError("m", "connection reset"), True),
        (ValueError("nope"), False),
        (asyncio.CancelledError(), False),
    ],
)
async def test_is_transient_error(exc: BaseException, expected: bool) -> None:
    assert is_transient_error(exc) is expected


# --------------------------------------------------------------------------- #
# RetryConfig
# --------------------------------------------------------------------------- #


async def test_retry_config_from_config_defaults() -> None:
    cfg = RetryConfig.from_config(SubAgentConfig(name="a", description="d", instructions="i"))
    assert cfg.max_retries == 3
    assert cfg.initial_delay == 1.0
    assert cfg.max_delay == 30.0
    assert cfg.backoff_multiplier == 2.0
    assert cfg.jitter is True
    assert cfg.retry_on is None


async def test_retry_config_from_config_custom() -> None:
    cfg = RetryConfig.from_config(
        SubAgentConfig(
            name="a",
            description="d",
            instructions="i",
            max_retries=5,
            retry_initial_delay=0.5,
            retry_max_delay=10.0,
            retry_backoff_multiplier=3.0,
            retry_jitter=False,
        )
    )
    assert cfg.max_retries == 5
    assert cfg.initial_delay == 0.5
    assert cfg.max_delay == 10.0
    assert cfg.backoff_multiplier == 3.0
    assert cfg.jitter is False


async def test_retry_config_should_retry_default() -> None:
    cfg = RetryConfig(max_retries=1)
    assert cfg.should_retry(ModelHTTPError(503, "m")) is True
    assert cfg.should_retry(ValueError("x")) is False


async def test_retry_config_should_retry_custom_predicate() -> None:
    cfg = RetryConfig(max_retries=1, retry_on=lambda exc: isinstance(exc, ValueError))
    assert cfg.should_retry(ValueError("x")) is True
    assert cfg.should_retry(ModelHTTPError(503, "m")) is False


# --------------------------------------------------------------------------- #
# compute_backoff_delay
# --------------------------------------------------------------------------- #


async def test_compute_backoff_delay_no_jitter() -> None:
    cfg = RetryConfig(initial_delay=1.0, max_delay=10.0, backoff_multiplier=2.0, jitter=False)
    assert compute_backoff_delay(1, cfg) == 1.0
    assert compute_backoff_delay(2, cfg) == 2.0
    assert compute_backoff_delay(3, cfg) == 4.0
    # Capped at max_delay.
    assert compute_backoff_delay(10, cfg) == 10.0


async def test_compute_backoff_delay_with_jitter() -> None:
    cfg = RetryConfig(initial_delay=4.0, max_delay=100.0, backoff_multiplier=2.0, jitter=True)
    seen: list[tuple[float, float]] = []

    def fake_rng(a: float, b: float) -> float:
        seen.append((a, b))
        return b / 2

    delay = compute_backoff_delay(2, cfg, rng=fake_rng)
    assert delay == 4.0  # (4 * 2^1) / 2
    assert seen == [(0.0, 8.0)]


# --------------------------------------------------------------------------- #
# run_with_retry — fast path (max_retries == 0)
# --------------------------------------------------------------------------- #


async def test_fast_path_uses_agent_run() -> None:
    agent = ScriptedAgent([{"result": FakeResult("ok")}])
    result = await run_with_retry(
        agent, "go", run_kwargs={"deps": 1}, retry=RetryConfig(max_retries=0)
    )
    assert result.output == "ok"
    assert agent.run_calls and not agent.iter_calls


async def test_fast_path_propagates_error() -> None:
    agent = ScriptedAgent([{"run_raise": ModelHTTPError(503, "m")}])
    with pytest.raises(ModelHTTPError):
        await run_with_retry(agent, "go", run_kwargs={}, retry=RetryConfig(max_retries=0))


# --------------------------------------------------------------------------- #
# run_with_retry — retry path
# --------------------------------------------------------------------------- #


async def test_retry_success_first_try_no_sleep() -> None:
    agent = ScriptedAgent([{"result": FakeResult("done")}])
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)

    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={"deps": 1},
        retry=RetryConfig(max_retries=2, jitter=False),
        sleep=sleep,
    )
    assert result.output == "done"
    assert agent.iter_calls and not agent.run_calls
    assert slept == []


async def test_retry_then_success_resumes_with_history() -> None:
    history = [{"role": "assistant", "content": "partial"}]
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m"), "messages": history},
            {"result": FakeResult("recovered")},
        ]
    )
    events: list[tuple[int, str, float]] = []

    def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        events.append((attempt, type(exc).__name__, delay))

    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={"deps": 1},
        retry=RetryConfig(max_retries=3, initial_delay=0.01, jitter=False),
        on_retry=on_retry,
        sleep=_no_sleep,
    )
    assert result.output == "recovered"
    # Second attempt resumed: prompt dropped, history replayed.
    assert agent.iter_calls[0]["prompt"] == "go"
    assert agent.iter_calls[0]["message_history"] is None
    assert agent.iter_calls[1]["prompt"] is None
    assert agent.iter_calls[1]["message_history"] == history
    assert events == [(1, "ModelHTTPError", 0.01)]


async def test_retry_exhausted_raises_last_error() -> None:
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m")},
            {"iter_raise": ModelHTTPError(502, "m")},
        ]
    )
    with pytest.raises(ModelHTTPError) as exc_info:
        await run_with_retry(
            agent,
            "go",
            run_kwargs={},
            retry=RetryConfig(max_retries=1, jitter=False),
            sleep=_no_sleep,
        )
    assert exc_info.value.status_code == 502


async def test_non_transient_not_retried() -> None:
    agent = ScriptedAgent([{"iter_raise": ValueError("logic bug")}])
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)

    with pytest.raises(ValueError, match="logic bug"):
        await run_with_retry(
            agent,
            "go",
            run_kwargs={},
            retry=RetryConfig(max_retries=5, jitter=False),
            sleep=sleep,
        )
    assert slept == []


async def test_retry_aenter_failure_run_is_none() -> None:
    """A failure before a run is yielded skips history capture."""
    agent = ScriptedAgent(
        [
            {"aenter_raise": ModelHTTPError(503, "m")},
            {"result": FakeResult("after-aenter-fail")},
        ]
    )
    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={},
        retry=RetryConfig(max_retries=2, jitter=False),
        sleep=_no_sleep,
    )
    assert result.output == "after-aenter-fail"
    # No history captured -> prompt is preserved on the retry.
    assert agent.iter_calls[1]["prompt"] == "go"
    assert agent.iter_calls[1]["message_history"] is None


async def test_retry_empty_messages_keeps_prompt() -> None:
    """An empty accumulated history does not drop the prompt."""
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m"), "messages": []},
            {"result": FakeResult("ok")},
        ]
    )
    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={},
        retry=RetryConfig(max_retries=2, jitter=False),
        sleep=_no_sleep,
    )
    assert result.output == "ok"
    assert agent.iter_calls[1]["prompt"] == "go"
    assert agent.iter_calls[1]["message_history"] is None


async def test_retry_on_retry_async_callback_awaited() -> None:
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m")},
            {"result": FakeResult("ok")},
        ]
    )
    awaited: list[int] = []

    async def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
        awaited.append(attempt)

    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={},
        retry=RetryConfig(max_retries=2, jitter=False),
        on_retry=on_retry,
        sleep=_no_sleep,
    )
    assert result.output == "ok"
    assert awaited == [1]


async def test_retry_without_on_retry_callback() -> None:
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m")},
            {"result": FakeResult("ok")},
        ]
    )
    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={},
        retry=RetryConfig(max_retries=2, jitter=False),
        sleep=_no_sleep,
    )
    assert result.output == "ok"


async def test_retry_pops_caller_message_history() -> None:
    """A caller-supplied message_history seeds the first attempt."""
    seed = [{"role": "user", "content": "seed"}]
    agent = ScriptedAgent([{"result": FakeResult("ok")}])
    await run_with_retry(
        agent,
        "go",
        run_kwargs={"message_history": seed, "deps": 1},
        retry=RetryConfig(max_retries=1, jitter=False),
        sleep=_no_sleep,
    )
    assert agent.iter_calls[0]["message_history"] == seed


# --------------------------------------------------------------------------- #
# run_with_retry - cooperative (soft) cancellation
# --------------------------------------------------------------------------- #


async def test_cancel_check_stops_run_cooperatively() -> None:
    """A cancel_check returning True stops the drive loop with CancelledError."""
    # result=None so the drive loop body runs and reaches the cancel check
    # before completing.
    agent = ScriptedAgent([{}])

    with pytest.raises(asyncio.CancelledError):
        await run_with_retry(
            agent,
            "go",
            run_kwargs={},
            retry=RetryConfig(max_retries=2, jitter=False),
            sleep=_no_sleep,
            cancel_check=lambda: True,
        )
    # CancelledError is a BaseException, so the retry loop never retried it.
    assert len(agent.iter_calls) == 1


async def test_cancel_check_false_does_not_stop() -> None:
    """A cancel_check that stays False lets the run complete normally."""
    agent = ScriptedAgent([{"result": FakeResult("done")}])

    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={},
        retry=RetryConfig(max_retries=2, jitter=False),
        sleep=_no_sleep,
        cancel_check=lambda: False,
    )
    assert result.output == "done"


# --------------------------------------------------------------------------- #
# _run_async integration — handle reflects retries
# --------------------------------------------------------------------------- #


async def test_run_async_retries_then_completes() -> None:
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m"), "messages": [{"x": 1}]},
            {"result": FakeResult("finished", usage=None)},
        ]
    )
    config = SubAgentConfig(
        name="t",
        description="d",
        instructions="i",
        max_retries=2,
        retry_initial_delay=0.0,
        retry_jitter=False,
    )
    bus = InMemoryMessageBus()
    tm = TaskManager(message_bus=bus)

    await _run_async(
        agent=agent,
        config=config,
        description="do it",
        deps=FakeDeps(),
        task_id="task-r",
        task_manager=tm,
        message_bus=bus,
    )
    await asyncio.sleep(0.05)

    handle = tm.get_handle("task-r")
    assert handle is not None
    assert handle.status == TaskStatus.COMPLETED
    assert handle.result == "finished"
    assert handle.retry_count == 1
    # Transient error message cleared after eventual success.
    assert handle.error is None


async def test_run_async_retries_exhausted_fails() -> None:
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m")},
            {"iter_raise": ModelHTTPError(503, "m")},
        ]
    )
    config = SubAgentConfig(
        name="t",
        description="d",
        instructions="i",
        max_retries=1,
        retry_initial_delay=0.0,
        retry_jitter=False,
    )
    bus = InMemoryMessageBus()
    tm = TaskManager(message_bus=bus)

    await _run_async(
        agent=agent,
        config=config,
        description="do it",
        deps=FakeDeps(),
        task_id="task-f",
        task_manager=tm,
        message_bus=bus,
    )
    await asyncio.sleep(0.05)

    handle = tm.get_handle("task-f")
    assert handle is not None
    assert handle.status == TaskStatus.FAILED
    assert handle.retry_count == 1
    assert "503" in str(handle.error)


async def test_run_async_soft_cancel_marks_handle_cancelled() -> None:
    """soft_cancel is consumed by the run loop and surfaces on the handle.

    Regression guard for #117: the cancel event set by soft_cancel must be
    polled by the running subagent so the task stops cooperatively and the
    handle reflects CANCELLED.
    """

    config = SubAgentConfig(name="t", description="d", instructions="i")
    bus = InMemoryMessageBus()
    tm = TaskManager(message_bus=bus)
    task_id = "task-soft"

    class _BlockingRun:
        """A run whose node loop blocks until the cancel event is set."""

        def __init__(self) -> None:
            self.result = None
            self.next_node: Any = object()

        async def next(self, node: Any) -> Any:
            # Advance one node, then yield control so the drive loop re-enters
            # its top-of-loop cancel_check. Returning a non-End node keeps the
            # loop going until cancel_check observes the soft cancel.
            await asyncio.sleep(0.005)
            return object()

        def all_messages(self) -> list[Any]:
            return []

    class _BlockingCM:
        async def __aenter__(self) -> _BlockingRun:
            return _BlockingRun()

        async def __aexit__(self, *exc: object) -> bool:
            return False

    class _BlockingAgent:
        def iter(self, prompt: Any, *, message_history: Any = None, **kwargs: Any) -> _BlockingCM:
            return _BlockingCM()

    agent = _BlockingAgent()

    await _run_async(
        agent=agent,
        config=config,
        description="do it",
        deps=FakeDeps(),
        task_id=task_id,
        task_manager=tm,
        message_bus=bus,
    )
    # Let the task reach its first node boundary / block.
    await asyncio.sleep(0.01)

    result = await tm.soft_cancel(task_id)
    assert result is True

    # Give the run loop a chance to observe the cancel and tear down.
    for _ in range(50):
        handle = tm.get_handle(task_id)
        assert handle is not None
        if handle.status == TaskStatus.CANCELLED:
            break
        await asyncio.sleep(0.01)

    handle = tm.get_handle(task_id)
    assert handle is not None
    assert handle.status == TaskStatus.CANCELLED
    assert handle.error == "Task was cancelled"


# --------------------------------------------------------------------------- #
# run_kwargs (e.g. usage_limits) survive every retry attempt
# --------------------------------------------------------------------------- #


async def test_run_kwargs_forwarded_on_every_attempt() -> None:
    """Caller run_kwargs (deps, usage_limits, ...) reach the agent on each try.

    Regression guard for the retry x usage-limits interaction: limits must
    not be dropped when a transient failure triggers a resume via iter().
    """
    sentinel_limits = object()
    agent = ScriptedAgent(
        [
            {"iter_raise": ModelHTTPError(503, "m"), "messages": [{"x": 1}]},
            {"result": FakeResult("recovered")},
        ]
    )

    result = await run_with_retry(
        agent,
        "go",
        run_kwargs={"deps": 1, "usage_limits": sentinel_limits},
        retry=RetryConfig(max_retries=2, jitter=False),
        sleep=_no_sleep,
    )

    assert result.output == "recovered"
    assert len(agent.iter_calls) == 2
    assert agent.iter_calls[0]["usage_limits"] is sentinel_limits
    assert agent.iter_calls[1]["usage_limits"] is sentinel_limits


# --------------------------------------------------------------------------- #
# Event streaming on the iter()-based retry path
#
# Regression guard: the retry path drives the run via agent.iter(). A bare
# `async for _ in run` uses AgentRun.__anext__, which skips node hooks and
# never streams — so a configured event_stream_handler (or a
# wrap_run_event_stream capability) would silently never fire, dropping
# tool-call/reasoning events that consumers stream to their platform.
# run_with_retry must drive the run the way agent.run() does instead.
# --------------------------------------------------------------------------- #


async def test_event_stream_capability_fires_on_retry_path() -> None:
    """A `ProcessEventStream` capability receives events on the retry path."""

    events: list[Any] = []

    async def handler(_ctx: Any, stream: Any) -> None:
        async for event in stream:
            events.append(event)

    agent = Agent(TestModel(), capabilities=[ProcessEventStream(handler)])

    result = await run_with_retry(
        agent,
        "hello",
        run_kwargs={},
        retry=RetryConfig(max_retries=3, jitter=False),
        sleep=_no_sleep,
    )

    # The run completed AND the capability saw streamed events — proving the
    # iter()-driven retry path streams just like agent.run().
    assert result.output
    assert events, "event stream capability never fired on the retry path"


async def test_event_stream_handler_override_fires_on_retry_path() -> None:
    """An explicit `event_stream_handler` override drives streaming."""

    events: list[Any] = []

    async def handler(_ctx: Any, stream: Any) -> None:
        async for event in stream:
            events.append(event)

    # Agent has no handler of its own — the override is the only source.
    agent = Agent(TestModel())

    result = await run_with_retry(
        agent,
        "hello",
        run_kwargs={},
        retry=RetryConfig(max_retries=3, jitter=False),
        sleep=_no_sleep,
        event_stream_handler=handler,
    )

    assert result.output
    assert events, "override event_stream_handler never fired"


async def test_fast_path_forwards_event_stream_handler_override() -> None:
    """`max_retries == 0` forwards an explicit handler to `agent.run()`."""

    events: list[Any] = []

    async def handler(_ctx: Any, stream: Any) -> None:
        async for event in stream:
            events.append(event)

    agent = Agent(TestModel())

    result = await run_with_retry(
        agent,
        "hello",
        run_kwargs={},
        retry=RetryConfig(max_retries=0),
        event_stream_handler=handler,
    )

    assert result.output
    assert events, "fast path did not forward the event_stream_handler override"


async def test_retry_path_without_streaming_consumer_completes() -> None:
    """No handler and no streaming capability → cheap bare drive still runs."""

    agent = Agent(TestModel())

    result = await run_with_retry(
        agent,
        "hello",
        run_kwargs={},
        retry=RetryConfig(max_retries=3, jitter=False),
        sleep=_no_sleep,
    )

    assert result.output


async def test_streaming_drive_breaks_on_wrap_run_short_circuit() -> None:
    """A `wrap_run` short-circuit publishes the result before any node runs.

    `agent.iter()` stores the short-circuit result as `run.result` before
    yielding, so the streaming driver must detect it and stop instead of
    stepping the (already resolved) graph.
    """

    short_circuit = AgentRunResult("short-circuited")

    class _ShortCircuitRun(AbstractCapability):
        async def wrap_run(self, ctx: Any, *, handler: Any) -> Any:
            # Never call handler() → the run is skipped, result used directly.
            return short_circuit

    events: list[Any] = []

    async def handler(_ctx: Any, stream: Any) -> None:  # pragma: no cover - never reached
        async for event in stream:
            events.append(event)

    agent = Agent(TestModel(), capabilities=[_ShortCircuitRun()])

    result = await run_with_retry(
        agent,
        "hello",
        run_kwargs={},
        retry=RetryConfig(max_retries=2, jitter=False),
        sleep=_no_sleep,
        event_stream_handler=handler,
    )

    assert result is short_circuit
    assert events == []


def _make_tool_then_text_model(captured: dict[str, Any]) -> Any:
    """FunctionModel: call tool `dig` on the first turn, then emit final text.

    Records, on the second model request, every `UserPromptPart` content the
    model can see — so a test can assert injected steering arrived.
    """

    def model_fn(messages: list[Any], info: AgentInfo) -> Any:
        n_responses = sum(1 for m in messages if isinstance(m, ModelResponse))
        if n_responses == 0:
            return ModelResponse(parts=[ToolCallPart(tool_name="dig", args={})])
        captured["texts"] = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        return ModelResponse(parts=[TextPart(content="done")])

    return FunctionModel(model_fn)


async def test_inject_messages_folded_into_next_model_request() -> None:
    """Pending steering is appended to the subagent's next model request."""
    captured: dict[str, Any] = {}
    agent = Agent(_make_tool_then_text_model(captured))

    @agent.tool_plain
    def dig() -> str:
        return "dug"

    calls = {"n": 0}

    async def inject() -> list[str]:
        calls["n"] += 1
        # Nothing before the first request; steering arrives before the second.
        return ["narrow to packages/sparta/"] if calls["n"] == 2 else []

    result = await run_with_retry(
        agent,
        "search the repo",
        run_kwargs={},
        retry=RetryConfig(max_retries=3, jitter=False),
        inject_messages=inject,
    )

    assert result.output == "done"
    assert "narrow to packages/sparta/" in captured["texts"]
    # Polled once per model-request node (before request 1 and request 2),
    # never around the intervening tool-call node.
    assert calls["n"] == 2


async def test_run_async_steering_message_reaches_subagent() -> None:
    """End-to-end: a steering message sent mid-run is seen on the next request."""
    captured: dict[str, Any] = {}
    parked = asyncio.Event()
    release = asyncio.Event()

    def model_fn(messages: list[Any], info: AgentInfo) -> Any:
        n_responses = sum(1 for m in messages if isinstance(m, ModelResponse))
        if n_responses == 0:
            return ModelResponse(parts=[ToolCallPart(tool_name="hold", args={})])
        captured["texts"] = [
            p.content
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    async def hold() -> str:
        # Park the subagent between request 1 and request 2 so the test can
        # deliver a steering message at a deterministic point.
        parked.set()
        await release.wait()
        return "held"

    config = SubAgentConfig(name="t", description="d", instructions="i")
    bus = InMemoryMessageBus()
    tm = TaskManager(message_bus=bus)

    await _run_async(
        agent=agent,
        config=config,
        description="go",
        deps=FakeDeps(),
        task_id="steer-1",
        task_manager=tm,
        message_bus=bus,
    )
    bg_task = tm.tasks["steer-1"]

    await asyncio.wait_for(parked.wait(), timeout=2.0)
    await bus.send(
        AgentMessage(
            type=MessageType.TASK_UPDATE,
            sender="parent",
            receiver="subagent-steer-1",
            payload={"message": "STEER NOW"},
            task_id="steer-1",
        )
    )
    release.set()
    await asyncio.wait_for(bg_task, timeout=2.0)

    assert "STEER NOW" in captured["texts"]
    handle = tm.get_handle("steer-1")
    assert handle is not None
    assert handle.status == TaskStatus.COMPLETED
