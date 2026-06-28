"""The orchestration graph — how the agents are wired together.

This is a small, from-scratch version of the idea behind LangGraph:
  - A graph is a set of named NODES (each node runs one agent).
  - EDGES say which node runs next.
  - Some edges are CONDITIONAL: a router function looks at the current
    state and decides where to go (this is how we build loops).
  - Running the graph means: start at the entry node, run it, ask its
    edge where to go next, repeat until we reach END.

We build it by hand (instead of importing langgraph) so every moving part
is visible and dependency-free. The mental model transfers directly to
LangGraph if you later swap this out.

Pipeline shape:

    problem -> planner -> coder -> critic -> test -> (router)

The router after `test` decides:
  - if everything looks good                -> END
  - else, and we still have loops left      -> debugger -> coder (retry)
  - else (out of loops)                     -> END
"""

from __future__ import annotations

from typing import Callable

from app.agents import (
    coder_agent,
    critic_agent,
    debugger_agent,
    planner_agent,
    problem_agent,
    test_agent,
)
from app.config import Config
from app.state import AgentState, Context

# Sentinel meaning "stop here".
END = "__end__"

# A node takes the state + context and returns the (mutated) state.
NodeFn = Callable[[AgentState, Context], AgentState]
# A router takes the state and returns the name of the next node (or END).
RouterFn = Callable[[AgentState], str]


class Graph:
    """A minimal node/edge graph runner over a shared AgentState."""

    def __init__(self) -> None:
        self.nodes: dict[str, NodeFn] = {}
        self.routers: dict[str, RouterFn] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def set_entry(self, name: str) -> None:
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        """A plain edge: after `src`, always go to `dst`."""
        self.routers[src] = lambda _state, _dst=dst: _dst

    def add_conditional_edge(self, src: str, router: RouterFn) -> None:
        """A conditional edge: `router(state)` chooses the next node."""
        self.routers[src] = router

    def run(self, state: AgentState, ctx: Context, max_steps: int = 25) -> AgentState:
        """Execute nodes following edges until END (or a step cap)."""
        if self.entry is None:
            raise ValueError("Graph has no entry node. Call set_entry().")

        current = self.entry
        for _ in range(max_steps):
            if current == END:
                break
            if current not in self.nodes:
                raise KeyError(f"Unknown node '{current}'")
            state = self.nodes[current](state, ctx)
            router = self.routers.get(current)
            current = router(state) if router else END
        else:
            state.log("graph", f"hit max_steps={max_steps}; stopping")
        state.log("graph", "pipeline complete")
        return state


def _make_after_test(config: Config) -> RouterFn:
    """Build the router that runs after the test agent.

    Loop back through the debugger if the critic flagged the scaffold as
    over-complete OR a real check failed — but only while loops remain.
    Defined as a closure so the router can see `config.max_loops` while
    still matching the `RouterFn` signature (state -> next node).
    """

    def _after_test(state: AgentState) -> str:
        # 'is_scaffold' failing just means it's not skeletal enough; the
        # critic already covers over-completeness, so ignore it here and
        # only treat genuine case failures as "failed".
        failed = any(
            not r.passed and r.name not in {"is_scaffold"}
            for r in state.test_results
        )
        needs_work = state.over_complete or failed
        if needs_work and state.loop_count < config.max_loops:
            return "debugger"
        return END

    return _after_test


def build_graph(config: Config) -> Graph:
    """Wire the six agents into the pipeline described at the top."""
    g = Graph()

    g.add_node("problem", problem_agent.run)
    g.add_node("planner", planner_agent.run)
    g.add_node("coder", coder_agent.run)
    g.add_node("critic", critic_agent.run)
    # test_agent.run takes an extra problem_id; read it off the state.
    g.add_node("test", lambda s, c: test_agent.run(s, c, s.problem_id or None))
    g.add_node("debugger", debugger_agent.run)

    g.set_entry("problem")
    g.add_edge("problem", "planner")
    g.add_edge("planner", "coder")
    g.add_edge("coder", "critic")
    g.add_edge("critic", "test")
    g.add_conditional_edge("test", _make_after_test(config))
    # After the debugger nudges the learner, retry from the coder.
    g.add_edge("debugger", "coder")

    return g
