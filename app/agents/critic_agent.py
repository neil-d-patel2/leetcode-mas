"""Critic agent: guard reasoning quality AND the 'no full solutions' rule.

It inspects the scaffold and decides whether it gives too much away. The
graph uses `state.over_complete` to decide whether to loop back.
"""

from __future__ import annotations

from app.agents import render
from app.state import AgentState, Context


def run(state: AgentState, ctx: Context) -> AgentState:
    prompt = render(
        ctx.prompt("critic_prompt"),
        plan=state.plan,
        scaffold_code=state.scaffold_code,
    )
    response = ctx.llm.generate(prompt, tag="critic")

    state.critique = response
    state.over_complete = _detect_over_complete(response)
    state.log(
        "critic",
        f"critique recorded (over_complete={state.over_complete})",
    )
    return state


def _detect_over_complete(text: str) -> bool:
    """Look for the critic's literal YES/NO verdict line.

    Falls back to a simple heuristic if the verdict isn't found.
    """
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped in {"yes", "overcomplete: yes", "over complete: yes"}:
            return True
        if stripped in {"no", "overcomplete: no", "over complete: no"}:
            return False
    # Heuristic fallback: a scaffold with no TODOs is suspiciously complete.
    return "todo" not in text.lower()
