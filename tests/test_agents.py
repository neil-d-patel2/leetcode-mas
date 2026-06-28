"""Per-agent unit tests against the stub LLM."""

from app.agents import problem_agent, planner_agent, coder_agent, critic_agent
from app.state import AgentState


def test_problem_agent_fills_understanding(ctx):
    state = AgentState(problem_text="Return the sum of two numbers.")
    state = problem_agent.run(state, ctx)
    assert state.problem_understanding


def test_planner_retrieves_similar(ctx):
    state = AgentState(problem_text="Find two numbers that add to a target.")
    state = problem_agent.run(state, ctx)
    state = planner_agent.run(state, ctx)
    # The bundled leetcode/ samples should yield at least one analogy.
    assert len(state.similar_problems) >= 1


def test_coder_produces_scaffold_with_todos(ctx):
    state = AgentState(problem_text="x", plan="use a hash map")
    state = coder_agent.run(state, ctx)
    assert state.scaffold_code
    assert "TODO" in state.scaffold_code.upper()


def test_critic_sets_over_complete_flag(ctx):
    state = AgentState(scaffold_code="def solve(): return 42")  # no TODOs
    state = critic_agent.run(state, ctx)
    assert isinstance(state.over_complete, bool)
