"""Tests for the auto-retry layer (``subagents_pydantic_ai.retry``).

Covers transient-error classification, backoff computation, the
``RetryConfig`` resolution, the ``run_with_retry`` driver (both the legacy
fast path and the ``iter()``-based resume-with-history path), and the
``_run_async`` integration that surfaces retries on the task handle.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

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
from subagents_pydantic_ai.types import SubAgentConfig

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeResult:
    """Stand-in for ``AgentRunResult``."""

    def __init__(self, output: str, usage: Any = None) -> None:
        self.output = output
        self._usage = usage

    def usage(self) -> Any:
        return self._usage


class _ScriptedRun:
    """Async-iterable stand-in for ``AgentRun``."""

    def __init__(self, step: dict[str, Any]) -> None:
        self._step = step
        self.result = step.get("result")
        self._raised = False

    def __aiter__(self) -> _ScriptedRun:
        return self

    async def __anext__(self) -> Any:
        if "iter_raise" in self._step and not self._raised:
            self._raised = True
            raise self._step["iter_raise"]
        raise StopAsyncIteration

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

    Each step may contain ``result`` (success), ``iter_raise`` (raise
    while iterating, with optional ``messages``), ``aenter_raise`` (raise
    before yielding a run), or ``run_raise`` (legacy ``run()`` path).
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        self._steps = list(steps)
        self.iter_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []

    def iter(self, prompt: Any, *, message_history: Any = None, **kwargs: Any) -> _ScriptedCM:
        self.iter_calls.append({"prompt": prompt, "message_history": message_history})
        return _ScriptedCM(self._steps.pop(0))

    async def run(self, prompt: Any, **kwargs: Any) -> Any:
        self.run_calls.append({"prompt": prompt, "kwargs": kwargs})
        step = self._steps.pop(0)
        if "run_raise" in step:
            raise step["run_raise"]
        return step["result"]


class FakeDeps:
    """Minimal deps object ``_run_async`` can attach state to."""


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
