"""Problem agent: understand the problem deeply, without solving it.

First stage of the pipeline. It produces a plain-language understanding
and a list of constraints that later agents build on.
"""

from __future__ import annotations

from app.agents import render
from app.state import AgentState, Context


def run(state: AgentState, ctx: Context) -> AgentState:
    prompt = render(ctx.prompt("problem_prompt"), problem_text=state.problem_text)
    response = ctx.llm.generate(prompt, tag="problem")

    # For the scaffold we store the whole response as the understanding.
    # TODO (later): parse the "Understanding" / "Constraints" sections
    # apart instead of duplicating the text into both fields.
    state.problem_understanding = response
    state.constraints = response
    state.log("problem", "produced problem understanding + constraints")
    return state
