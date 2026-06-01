# SubAgentCapability

`SubAgentCapability` is the recommended way to add subagent delegation to a Pydantic AI agent.
It's a [pydantic-ai capability](https://ai.pydantic.dev/capabilities/) that bundles
delegation tools and instructions into a single plug-and-play unit.

## Why Capability over Toolset?

| Feature | SubAgentCapability | create_subagent_toolset |
|---------|:-:|:-:|
| Tools registered automatically | Yes | Yes |
| Dynamic system prompt (lists subagents) | Yes | Manual wiring |
| AgentSpec YAML support | Yes | No |
| Single import | Yes | Need toolset + prompt function |
| `task_manager` access | Property | `getattr(toolset, "task_manager")` |

## Basic Usage

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[SubAgentCapability(
        subagents=[
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
            ),
        ],
    )],
)
```

## Configuration

```python
SubAgentCapability(
    subagents=[...],                    # Subagent configurations
    default_model="openai:gpt-4.1",    # Default model for subagents
    include_general_purpose=True,       # Include GP subagent (default: True)
    max_nesting_depth=0,                # Allow nested subagents (0 = no nesting)
    toolsets_factory=my_factory,        # Custom toolsets for subagents
    registry=my_registry,              # Dynamic agent registry
    descriptions={                      # Override tool descriptions
        "task": "Delegate work to a specialist",
    },
    usage_limits=UsageLimits(           # Static limits, or a factory (see below)
        request_limit=10,
    ),
)
```

### Usage limits

`usage_limits` caps token/request usage for delegated subagent runs. Pass a
`UsageLimits` (from pydantic-ai) instance to reuse the same limits
for every task, or a
[`UsageLimitsFactory`][subagents_pydantic_ai.types.UsageLimitsFactory] —
`(ctx, config) -> UsageLimits | None` — called once per delegated task with the
parent run context and the selected subagent config. A factory may return `None`
to run that task without explicit limits. Limits are enforced on every retry
attempt as well.

```python
from pydantic_ai import RunContext, UsageLimits
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

def limits_for(ctx: RunContext, config: SubAgentConfig) -> UsageLimits | None:
    # Give the researcher a larger budget than other subagents.
    if config["name"] == "researcher":
        return UsageLimits(request_limit=20)
    return UsageLimits(request_limit=5)

cap = SubAgentCapability(subagents=[...], usage_limits=limits_for)
```

## How It Works

When you pass `SubAgentCapability` to an agent, pydantic-ai calls:

1. **`get_toolset()`** — returns the `FunctionToolset` containing delegation tools
   (`task`, `check_task`, `answer_subagent`, `list_active_tasks`, `wait_tasks`,
   `soft_cancel_task`, `hard_cancel_task`)

2. **`get_instructions()`** — returns a callable that generates the system prompt
   listing available subagents with their descriptions (via
   [`get_subagent_system_prompt`][subagents_pydantic_ai.prompts.get_subagent_system_prompt])

!!! warning "Sync-mode questions are not supported via the capability"
    `SubAgentCapability` builds its toolset without an `ask_user` callback (see
    [`capability.py`][subagents_pydantic_ai.capability.SubAgentCapability]). A
    subagent that calls `ask_parent` in **sync** mode therefore gets a
    configuration error. To support sync-mode questions
    (`can_ask_questions=True`), build the toolset directly with
    [`create_subagent_toolset`][subagents_pydantic_ai.toolset.create_subagent_toolset]
    and pass `ask_user=...`. In **async** mode the parent answers via
    `answer_subagent`, which works with the capability. See
    [Parent-Child Questions](../advanced/questions.md).

## Observability

Access the task manager for monitoring background tasks:

```python
cap = SubAgentCapability(subagents=[...])
agent = Agent("openai:gpt-4.1", capabilities=[cap])

# After agent runs, check active tasks
task_mgr = cap.task_manager
if task_mgr:
    active = task_mgr.list_active_tasks()
```

## Composing with Other Capabilities

```python
from pydantic_ai import Agent
from pydantic_ai_todo import TodoCapability
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        TodoCapability(enable_subtasks=True),
        SubAgentCapability(subagents=[...]),
    ],
)
```

## Custom Agents via SubAgentConfig

`SubAgentCapability` supports using custom agent instances through the `agent` and
`agent_factory` fields on `SubAgentConfig`. When the capability compiles subagents
internally, it follows the same resolution priority as `_compile_subagent`:
`agent` > `agent_factory` > default `Agent()`.

```python
from pydantic_ai import Agent
from pydantic_deep import create_deep_agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[SubAgentCapability(
        subagents=[
            # Pre-built agent
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
                agent=create_deep_agent(model="openai:gpt-4.1"),
            ),
            # Agent factory
            SubAgentConfig(
                name="coder",
                description="Writes code",
                instructions="You write Python code.",
                agent_factory=lambda cfg: create_deep_agent(
                    model=cfg.get("model", "openai:gpt-4.1"),
                    system_prompt=cfg["instructions"],
                ),
            ),
            # Default: plain Agent is created automatically
            SubAgentConfig(
                name="writer",
                description="Writes content",
                instructions="You write clear documentation.",
            ),
        ],
    )],
)
```

See [SubAgentConfig](types.md#subagentconfig) for full details on these fields.

## AgentSpec (YAML)

```yaml
model: openai:gpt-4.1
capabilities:
  - SubAgentCapability:
      subagents:
        - name: researcher
          description: Researches topics
          instructions: You are a research assistant.
      include_general_purpose: true
```
