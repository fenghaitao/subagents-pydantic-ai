# Prompts &amp; Retry API

This page documents the exported prompt builders, the static prompt/tool
description constants, and the auto-retry helpers.

## Prompt Builders

### get_subagent_system_prompt

::: subagents_pydantic_ai.get_subagent_system_prompt
    options:
      show_root_heading: true
      show_source: true

### get_task_instructions_prompt

::: subagents_pydantic_ai.get_task_instructions_prompt
    options:
      show_root_heading: true
      show_source: true

## Prompt &amp; Description Constants

These string constants are exported for inspection and overriding. The
`*_DESCRIPTION` constants are the default model-facing tool descriptions used by
[`create_subagent_toolset`][subagents_pydantic_ai.toolset.create_subagent_toolset]
(override them per-tool via its `descriptions` argument).

::: subagents_pydantic_ai.SUBAGENT_SYSTEM_PROMPT
    options:
      show_root_heading: true

::: subagents_pydantic_ai.DUAL_MODE_SYSTEM_PROMPT
    options:
      show_root_heading: true

::: subagents_pydantic_ai.DEFAULT_GENERAL_PURPOSE_DESCRIPTION
    options:
      show_root_heading: true

::: subagents_pydantic_ai.TASK_TOOL_DESCRIPTION
    options:
      show_root_heading: true

::: subagents_pydantic_ai.CHECK_TASK_DESCRIPTION
    options:
      show_root_heading: true

::: subagents_pydantic_ai.ANSWER_SUBAGENT_DESCRIPTION
    options:
      show_root_heading: true

::: subagents_pydantic_ai.LIST_ACTIVE_TASKS_DESCRIPTION
    options:
      show_root_heading: true

::: subagents_pydantic_ai.WAIT_TASKS_DESCRIPTION
    options:
      show_root_heading: true

::: subagents_pydantic_ai.SOFT_CANCEL_TASK_DESCRIPTION
    options:
      show_root_heading: true

::: subagents_pydantic_ai.HARD_CANCEL_TASK_DESCRIPTION
    options:
      show_root_heading: true

## Retry

See [Auto-Retry](../advanced/retries.md) for a conceptual overview.

### RetryConfig

::: subagents_pydantic_ai.RetryConfig
    options:
      show_root_heading: true
      show_source: true

### run_with_retry

::: subagents_pydantic_ai.run_with_retry
    options:
      show_root_heading: true
      show_source: true

### is_transient_error

::: subagents_pydantic_ai.is_transient_error
    options:
      show_root_heading: true
      show_source: true

### compute_backoff_delay

::: subagents_pydantic_ai.compute_backoff_delay
    options:
      show_root_heading: true
      show_source: true
