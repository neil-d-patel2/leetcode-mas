"""Planner agent: identify patterns and retrieve analogous past problems.

This agent is where retrieval meets reasoning:
  1. Query the vector store with the problem understanding to find the
     learner's most similar past solutions.
  2. Ask the LLM to name candidate patterns and lay out guiding steps
     (hints), using those analogies as context.
"""

from __future__ import annotations

from app.agents import render
from app.state import AgentState, Context, SimilarProblem


def _retrieve_similar(state: AgentState, ctx: Context) -> list[SimilarProblem]:
    query = state.problem_understanding or state.problem_text
    hits = ctx.retriever.query(query, top_k=ctx.config.retrieval_top_k)
    similar = [
        SimilarProblem(
            name=payload.get("name", "unknown"),
            path=payload.get("path", ""),
            score=score,
            tags=payload.get("tags", []),
            snippet=payload.get("snippet", ""),
        )
        for score, payload in hits
    ]
    return similar


def _format_similar(similar: list[SimilarProblem]) -> str:
    if not similar:
        return "(no similar past problems found yet)"
    lines = []
    for s in similar:
        tags = ", ".join(s.tags) if s.tags else "-"
        lines.append(f"- {s.name} (score {s.score:.2f}; tags: {tags})")
    return "\n".join(lines)


def run(state: AgentState, ctx: Context) -> AgentState:
    state.similar_problems = _retrieve_similar(state, ctx)
    state.log(
        "planner",
        f"retrieved {len(state.similar_problems)} similar past problem(s)",
    )

    prompt = render(
        ctx.prompt("planner_prompt"),
        problem_understanding=state.problem_understanding,
        constraints=state.constraints,
        similar_problems=_format_similar(state.similar_problems),
    )
    response = ctx.llm.generate(prompt, tag="planner")

    state.plan = response
    # TODO (later): parse a real list of pattern names out of the
    # response. For now we keep the whole plan and leave patterns empty
    # unless the LLM happens to mention known ones.
    state.patterns = _guess_patterns(response)
    state.log("planner", f"patterns: {state.patterns or '[none parsed yet]'}")
    return state


# A starter list of well-known patterns to look for in the plan text.
_KNOWN_PATTERNS = [
    "sliding window",
    "two pointer",
    "binary search",
    "hashing",
    "dynamic programming",
    "dp",
    "bfs",
    "dfs",
    "backtracking",
    "greedy",
    "heap",
    "union find",
    "stack",
    "recursion",
]


def _guess_patterns(text: str) -> list[str]:
    low = text.lower()
    return [p for p in _KNOWN_PATTERNS if p in low]
