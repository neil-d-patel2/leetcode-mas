"""Debugger agent: when reasoning is weak, ask guiding questions.

Triggered by the graph when the critic flags over-completeness or when
checks fail. It does NOT fix the code — it nudges the learner with
invariants and questions, then the graph may loop back for another pass.
"""

from __future__ import annotations

from app.agents import render
from app.state import AgentState, Context


def _format_results(state: AgentState) -> str:
    if not state.test_results:
        return "(no test results)"
    return "\n".join(
        f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.detail}"
        for r in state.test_results
    )


def run(state: AgentState, ctx: Context) -> AgentState:
    prompt = render(
        ctx.prompt("debugger_prompt"),
        plan=state.plan,
        critique=state.critique,
        test_results=_format_results(state),
    )
    response = ctx.llm.generate(prompt, tag="debugger")

    state.debug_notes = response
    state.loop_count += 1
    state.log("debugger", f"produced debug notes (loop {state.loop_count})")
    return state
