"""End-to-end: the whole pipeline runs and fills in the state."""

from app.graph import build_graph
from app.state import AgentState


def test_pipeline_runs_end_to_end(ctx, config):
    state = AgentState(
        problem_text="Given nums and target, return indices summing to target.",
        problem_id="two-sum",
    )
    state = build_graph(config).run(state, ctx)

    # Each stage should have contributed something.
    assert state.problem_understanding
    assert state.plan
    assert state.scaffold_code
    assert state.critique
    assert any("complete" in h for h in state.history) or state.history


def test_scaffold_is_not_a_full_solution(ctx, config):
    """Core product guarantee: output stays scaffolded, not solved."""
    state = AgentState(problem_text="anything", problem_id="two-sum")
    state = build_graph(config).run(state, ctx)
    assert "TODO" in state.scaffold_code.upper()
