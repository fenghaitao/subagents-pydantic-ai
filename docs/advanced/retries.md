# Auto-Retry

Subagent runs are automatically retried on transient networking failures. Model
gateways and proxies (e.g. LiteLLM) occasionally return `502`/`503`/`504`, hit a
`429` rate limit, or drop a connection. Rather than failing the whole delegation,
the toolset retries the subagent with exponential backoff.

Crucially, each retry **resumes with the full accumulated message history** from
the failed attempt, so partial progress (completed model turns and tool calls) is
not thrown away &mdash; the subagent continues instead of restarting from scratch.

## Defaults

Retrying is **on by default**: a subagent gets `3` extra attempts after the first
failure, with exponential backoff and jitter. You opt out by setting
`max_retries=0`, which restores the legacy `agent.run()` path.

## Per-Subagent Configuration

Retry behaviour is configured through `retry_*` fields on
[`SubAgentConfig`][subagents_pydantic_ai.types.SubAgentConfig]. Any field you omit
falls back to the default policy.

```python
from subagents_pydantic_ai import SubAgentConfig

SubAgentConfig(
    name="researcher",
    description="Researches topics",
    instructions="You are a research assistant.",
    max_retries=5,                 # extra attempts after the first failure (default 3)
    retry_initial_delay=1.0,       # seconds before the first retry (default 1.0)
    retry_max_delay=30.0,          # cap on the backoff delay (default 30.0)
    retry_backoff_multiplier=2.0,  # delay growth factor per attempt (default 2.0)
    retry_jitter=True,             # randomise delay in [0, delay] (default True)
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Extra attempts after the first failure. `0` disables retrying |
| `retry_initial_delay` | `float` | `1.0` | Seconds to wait before the first retry |
| `retry_max_delay` | `float` | `30.0` | Upper bound on the backoff delay |
| `retry_backoff_multiplier` | `float` | `2.0` | Delay multiplier applied each attempt |
| `retry_jitter` | `bool` | `True` | Randomise the delay in `[0, delay]` (full jitter) to avoid a thundering herd across concurrent subagents |
| `retry_on` | `Callable[[BaseException], bool]` | built-in classifier | Custom predicate deciding whether an exception is transient |

These fields are resolved into a
[`RetryConfig`][subagents_pydantic_ai.retry.RetryConfig] via
`RetryConfig.from_config(config)`.

## What Counts as Transient

By default, [`is_transient_error`][subagents_pydantic_ai.retry.is_transient_error]
decides whether a failure is worth retrying:

- A `ModelHTTPError` with status `408`, `409`, `425`, `429`, `500`, `502`, `503`,
  `504`, or `529` &mdash; gateway hiccups, rate limits, or upstream overload.
- A bare `ModelAPIError` (no HTTP status) &mdash; connection resets, read
  timeouts, and other transport-level failures from the model client.

Everything else &mdash; auth/4xx errors, `UnexpectedModelBehavior`,
`UsageLimitExceeded`, `UserError`, validation errors, and task cancellation
(`asyncio.CancelledError`) &mdash; is treated as **non-transient** and is not
retried.

### Custom Classification

Provide your own predicate via `retry_on` to override the default classifier:

```python
from pydantic_ai.exceptions import ModelHTTPError

def only_rate_limits(exc: BaseException) -> bool:
    return isinstance(exc, ModelHTTPError) and exc.status_code == 429

SubAgentConfig(
    name="researcher",
    description="Researches topics",
    instructions="You are a research assistant.",
    retry_on=only_rate_limits,
)
```

## Backoff Delay

[`compute_backoff_delay`][subagents_pydantic_ai.retry.compute_backoff_delay]
computes the wait before each retry (1-based attempt):

```
base  = initial_delay * (backoff_multiplier ** (attempt - 1))
delay = min(base, max_delay)
# with jitter: delay = random.uniform(0.0, delay)  (full jitter)
```

With the defaults this yields roughly `1s`, `2s`, `4s`, ... capped at `30s`,
each randomised down to `[0, delay]` when jitter is enabled.

## Observing Retries

While a task is waiting between attempts, its status is
[`TaskStatus.RETRYING`][subagents_pydantic_ai.types.TaskStatus]. The number of
retries performed for a task is tracked on
[`TaskHandle.retry_count`][subagents_pydantic_ai.types.TaskHandle].

```python
handle = task_manager.get_handle(task_id)
if handle.status == TaskStatus.RETRYING:
    print(f"Retrying (attempt {handle.retry_count})")
```

## Under the Hood

[`run_with_retry`][subagents_pydantic_ai.retry.run_with_retry] drives the retry
loop. When `max_retries > 0`, it runs the agent via `Agent.iter()` so that, on a
transient failure, the accumulated history from the failed attempt is replayed via
`message_history` on the next attempt. When `max_retries <= 0`, it falls through to
a plain `agent.run()` &mdash; the legacy path, unchanged. Event streaming
(`event_stream_handler` and `wrap_run_event_stream` capabilities) keeps working
across retries, and cooperative (soft) cancellation is honoured at node boundaries
on the retry-driven path.

See the [Prompts &amp; Retry API](../api/prompts.md#retry) for full signatures.

## Next Steps

- [Execution Modes](execution-modes.md) - Sync vs async delegation
- [Cancellation](cancellation.md) - Stopping running tasks
