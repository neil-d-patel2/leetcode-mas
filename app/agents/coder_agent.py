"""Coder agent: emit a PARTIAL scaffold, never a full solution.

This is the agent most likely to "cheat" by writing the whole answer, so
its prompt is strict and the critic double-checks its output. We extract
the fenced code block so later agents can work with clean code.
"""

from __future__ import annotations

from app.agents import render
from app.state import AgentState, Context
from app.tools.parse_code import count_todos, extract_code_block


def run(state: AgentState, ctx: Context) -> AgentState:
    prompt = render(
        ctx.prompt("coder_prompt"),
        plan=state.plan,
        patterns=", ".join(state.patterns) if state.patterns else "(see plan)",
    )
    response = ctx.llm.generate(prompt, tag="coder")

    state.scaffold_code = extract_code_block(response)
    state.log(
        "coder",
        f"produced scaffold ({count_todos(state.scaffold_code)} TODO marker(s))",
    )
    return state
