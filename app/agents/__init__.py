"""The specialised agents and a small shared helper.

Every agent is just a function:

    def run(state: AgentState, ctx: Context) -> AgentState

It reads fields it needs from `state`, optionally calls the LLM via
`ctx.llm`, writes its own fields back into `state`, and returns it. The
graph (app/graph.py) chains these together. Keeping agents as plain
functions (rather than classes) makes the data flow easy to follow.
"""

from __future__ import annotations

from string import Template


def render(template_text: str, **fields: object) -> str:
    """Fill a `$placeholder` prompt template with the given fields.

    Uses `safe_substitute` so a missing field leaves the placeholder
    intact instead of raising — handy while iterating on prompts.
    """
    return Template(template_text).safe_substitute(**fields)
