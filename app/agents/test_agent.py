"""Test agent: validate the scaffold against sample cases.

Remember the scaffold is intentionally incomplete, so we lead with
static checks (does it parse? is it still a scaffold?) and only run
sample cases if the learner has provided them and filled the code in.
Sample cases for a problem live in data/test_cases/<id>.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.state import AgentState, Context
from app.tools.run_tests import run_sample_cases, static_check


def _load_cases(ctx: Context, problem_id: str | None) -> dict:
    """Load {func_name, cases} for a problem id, or empty if none."""
    if not problem_id:
        return {}
    path = Path(ctx.config.data_dir) / "test_cases" / f"{problem_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run(state: AgentState, ctx: Context, problem_id: str | None = None) -> AgentState:
    results = static_check(state.scaffold_code)

    spec = _load_cases(ctx, problem_id)
    if spec.get("cases"):
        results += run_sample_cases(
            state.scaffold_code,
            spec.get("func_name", "solve"),
            spec["cases"],
        )

    state.test_results = results
    passed = sum(1 for r in results if r.passed)
    state.log("test", f"{passed}/{len(results)} check(s) passed")
    return state
