"""Auto-retry for subagent runs over flaky networking.

Wraps a subagent's execution so transient failures (gateway 5xx, rate
limits, connection drops from proxies such as LiteLLM) are retried with
exponential backoff. Each retry resumes with the full accumulated message
history, so partial progress (model turns, tool calls) is not lost.

The retry path deliberately uses :meth:`Agent.iter` rather than
``capture_run_messages()`` to recover the failed run's messages, because
nested ``capture_run_messages`` contexts do not work
(https://github.com/pydantic/pydantic-ai/issues/1568) and subagents
always run nested inside a parent agent's run. ``Agent.iter`` exposes the
accumulated history directly, sidestepping that limitation entirely.

When retrying is disabled (``max_retries == 0``, the default), the legacy
``agent.run()`` call path is used unchanged, so behaviour only differs
when retrying is explicitly opted in.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from subagents_pydantic_ai.types import SubAgentConfig

# HTTP status codes that indicate a transient, retry-worthy failure:
# request timeout, conflict, too-early, rate limit, and 5xx server/gateway
# errors (502/503/504 are typical of an overloaded LiteLLM proxy).
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

RetryPredicate = Callable[[BaseException], bool]
"""Predicate deciding whether an exception is a transient, retryable error."""

OnRetryCallback = Callable[[int, BaseException, float], "Awaitable[None] | None"]
"""Callback invoked before each retry sleep with ``(attempt, exc, delay)``."""


def is_transient_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like a transient networking failure.

    Treated as transient (worth retrying):

    - ``ModelHTTPError`` with a 408/409/425/429/5xx status code — gateway
      hiccups, rate limits or upstream overload, typical with proxies
      such as LiteLLM.
    - ``ModelAPIError`` that is *not* an HTTP error — connection resets,
      read timeouts and other transport-level problems surfaced by the
      model client.

    Everything else (auth/4xx, ``UnexpectedModelBehavior``,
    ``UsageLimitExceeded``, ``UserError``, validation errors, task
    cancellation, ...) is treated as non-transient and is not retried.
    """
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _TRANSIENT_STATUS_CODES
    # A bare ModelAPIError (no HTTP status) is a transport/connection
    # error from the model client — safe to retry.
    return isinstance(exc, ModelAPIError)


@dataclass(frozen=True)
class RetryConfig:
    """Resolved retry policy for a subagent run.

    Attributes:
        max_retries: Number of *additional* attempts after the first
            failure. ``0`` disables retrying entirely (the default, so
            existing behaviour is unchanged unless opted in).
        initial_delay: Seconds to wait before the first retry.
        max_delay: Upper bound for the backoff delay, in seconds.
        backoff_multiplier: The delay is multiplied by this each attempt.
        jitter: When ``True``, the delay is randomised in
            ``[0, computed_delay]`` (full jitter) to avoid a thundering
            herd across many concurrent subagents.
        retry_on: Predicate deciding whether an exception is transient.
            ``None`` uses :func:`is_transient_error`.
    """

    max_retries: int = 0
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on: RetryPredicate | None = None

    @classmethod
    def from_config(cls, config: SubAgentConfig) -> RetryConfig:
        """Build a :class:`RetryConfig` from a :class:`SubAgentConfig`.

        Missing keys fall back to the dataclass defaults, so a config
        without any ``retry_*`` keys yields a disabled policy.
        """
        return cls(
            max_retries=config.get("max_retries", 0),
            initial_delay=config.get("retry_initial_delay", 1.0),
            max_delay=config.get("retry_max_delay", 30.0),
            backoff_multiplier=config.get("retry_backoff_multiplier", 2.0),
            jitter=config.get("retry_jitter", True),
            retry_on=config.get("retry_on"),
        )

    def should_retry(self, exc: BaseException) -> bool:
        """Return whether *exc* is retryable under this policy."""
        predicate = self.retry_on or is_transient_error
        return predicate(exc)


def compute_backoff_delay(
    attempt: int,
    cfg: RetryConfig,
    rng: Callable[[float, float], float] = random.uniform,
) -> float:
    """Compute the delay (seconds) before retry *attempt* (1-based).

    Exponential backoff (``initial_delay * multiplier ** (attempt - 1)``)
    capped at ``cfg.max_delay``. With ``cfg.jitter`` the result is
    randomised in ``[0, delay]`` (full jitter). ``rng`` is injectable for
    deterministic tests.
    """
    base = cfg.initial_delay * (cfg.backoff_multiplier ** (attempt - 1))
    delay = min(base, cfg.max_delay)
    if cfg.jitter:
        delay = rng(0.0, delay)
    return delay


async def run_with_retry(
    agent: Any,
    user_prompt: str | None,
    *,
    run_kwargs: dict[str, Any],
    retry: RetryConfig,
    on_retry: OnRetryCallback | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Run *agent* with auto-retry on transient errors.

    When ``retry.max_retries <= 0`` this is exactly ``agent.run(...)`` —
    the legacy path, unchanged. Otherwise the agent is driven via
    ``agent.iter()`` so that, on a transient failure, the accumulated
    message history from the failed attempt is replayed via
    ``message_history`` on the next attempt and the subagent resumes
    instead of restarting from scratch.

    Args:
        agent: The pydantic-ai ``Agent`` to run.
        user_prompt: Initial prompt. After a retry that captured history
            it is set to ``None`` because the prompt is already replayed
            inside ``message_history``.
        run_kwargs: Extra kwargs forwarded to ``agent.run``/``agent.iter``
            (``deps``, ``toolsets``, ...). A caller-supplied
            ``message_history`` is honoured as the starting history.
        retry: The resolved retry policy.
        on_retry: Optional callback invoked before each retry sleep with
            ``(attempt, exc, delay)``. May be sync or async.
        sleep: Async sleep function, injectable for tests.

    Returns:
        The ``AgentRunResult`` of the first successful attempt.

    Raises:
        The last exception when retries are exhausted or the error is not
        transient. ``asyncio.CancelledError`` is a ``BaseException`` and
        is never caught here, so cooperative/hard task cancellation
        propagates unchanged.
    """
    if retry.max_retries <= 0:
        # Fast path: preserve exact legacy behaviour (no iter, no capture).
        return await agent.run(user_prompt, **run_kwargs)

    message_history = run_kwargs.pop("message_history", None)
    prompt = user_prompt
    attempt = 0
    while True:
        run = None
        try:
            async with agent.iter(prompt, message_history=message_history, **run_kwargs) as run:
                async for _ in run:
                    pass
            return run.result
        except Exception as exc:
            if attempt >= retry.max_retries or not retry.should_retry(exc):
                raise
            attempt += 1
            # Resume from wherever the failed attempt got to. ``run`` is
            # None only if ``agent.iter()`` failed before yielding.
            if run is not None:
                accumulated = run.all_messages()
                if accumulated:
                    message_history = accumulated
                    prompt = None
            delay = compute_backoff_delay(attempt, retry)
            if on_retry is not None:
                maybe_coro = on_retry(attempt, exc, delay)
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            await sleep(delay)
