"""The shared state that flows through the agent pipeline.

This is the heart of a LangGraph-style system: instead of passing many
arguments between functions, every agent receives one `AgentState`,
reads what it needs, fills in its own fields, and returns it. The graph
runner threads this single object from agent to agent.

`Context` is the *runtime* bag (the LLM client, config, retriever). It is
kept separate from `AgentState` because it does not change as the
problem is processed — it's the "environment" the agents run inside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# These imports are only needed for type hints. Importing them lazily
# (under TYPE_CHECKING) avoids circular imports at runtime, since those
# modules may eventually import from here.
if TYPE_CHECKING:  # pragma: no cover
    from app.config import Config
    from app.memory.vector_store import VectorStore
    from app.models.ollama_client import OllamaClient


@dataclass
class SimilarProblem:
    """One retrieved past solution used as an analogy (not to copy)."""

    name: str
    path: str
    score: float
    tags: list[str] = field(default_factory=list)
    snippet: str = ""


@dataclass
class TestResult:
    """Outcome of running a sample case against scaffolded code."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class AgentState:
    """Everything the agents produce while working on one problem.

    Fields roughly follow the pipeline order. Each agent owns a slice:
      problem_agent  -> problem_understanding, constraints
      planner_agent  -> patterns, similar_problems, plan
      coder_agent    -> scaffold_code
      critic_agent   -> critique, over_complete
      test_agent     -> test_results
      debugger_agent -> debug_notes
    """

    # Input
    problem_text: str = ""
    # Optional id used to locate sample cases in data/test_cases/<id>.json
    problem_id: str = ""

    # Problem agent output
    problem_understanding: str = ""
    constraints: str = ""

    # Planner agent output
    patterns: list[str] = field(default_factory=list)
    similar_problems: list[SimilarProblem] = field(default_factory=list)
    plan: str = ""

    # Coder agent output (intentionally PARTIAL — a scaffold, not a solution)
    scaffold_code: str = ""

    # Critic agent output
    critique: str = ""
    # True if the critic thinks the scaffold reveals too much of the answer.
    over_complete: bool = False

    # Test agent output
    test_results: list[TestResult] = field(default_factory=list)

    # Debugger agent output
    debug_notes: str = ""

    # Bookkeeping
    loop_count: int = 0
    # A human-readable trace of which agent did what, for learning/debugging.
    history: list[str] = field(default_factory=list)

    def log(self, who: str, message: str) -> None:
        """Append a line to the trace and echo it so runs are observable."""
        entry = f"[{who}] {message}"
        self.history.append(entry)
        print(entry)


@dataclass
class Context:
    """Runtime dependencies shared by all agents (the 'environment')."""

    config: "Config"
    llm: "OllamaClient"
    retriever: "VectorStore"
    # The raw text of each loaded prompt template, keyed by name.
    prompts: dict[str, str] = field(default_factory=dict)

    def prompt(self, name: str) -> str:
        """Return a loaded prompt template, with a clear error if missing."""
        if name not in self.prompts:
            raise KeyError(
                f"Prompt '{name}' not loaded. Available: {sorted(self.prompts)}"
            )
        return self.prompts[name]
