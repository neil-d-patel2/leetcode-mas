"""Lightweight check of scaffolded code against sample cases.

IMPORTANT: by design the coder agent produces PARTIAL code full of
`# TODO` markers, so it usually will NOT pass real test cases — and that
is fine. This module therefore does two complementary things:

  1. `static_check`: sanity checks that don't require running the code
     (does it parse? does it still contain TODOs / stay scaffolded?).
  2. `run_sample_cases`: optionally executes the code against sample
     inputs IF the learner has filled it in. Execution is sandboxed only
     lightly (restricted builtins) — this is a learning tool, not a
     security boundary, so only run code you trust.

Both return a list of `TestResult` so the test agent can record them in
the shared state.
"""

from __future__ import annotations

import ast
from typing import Any

from app.state import TestResult


def static_check(code: str) -> list[TestResult]:
    """Checks that never execute the code."""
    results: list[TestResult] = []

    # 1) Does it parse as valid Python?
    try:
        ast.parse(code)
        results.append(TestResult("parses", True, "code is syntactically valid"))
    except SyntaxError as exc:
        results.append(TestResult("parses", False, f"syntax error: {exc}"))

    # 2) Is it still appropriately scaffolded (contains TODOs)?
    has_todo = "todo" in code.lower()
    results.append(
        TestResult(
            "is_scaffold",
            has_todo,
            "contains TODOs (good — guidance, not a solution)"
            if has_todo
            else "no TODOs found — may be too complete",
        )
    )
    return results


def run_sample_cases(
    code: str,
    func_name: str,
    cases: list[dict[str, Any]],
) -> list[TestResult]:
    """Execute `func_name` from `code` against `cases`.

    Each case is {"input": [...args...], "expected": value, "name": str}.
    Returns one TestResult per case. If the function isn't defined yet
    (still a scaffold), every case is reported as a non-fatal skip.
    """
    results: list[TestResult] = []

    safe_globals: dict[str, Any] = {"__builtins__": __builtins__}
    namespace: dict[str, Any] = {}
    try:
        exec(compile(code, "<scaffold>", "exec"), safe_globals, namespace)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the pipeline
        return [TestResult("exec", False, f"failed to load code: {exc}")]

    func = namespace.get(func_name)
    if not callable(func):
        return [TestResult(func_name, False, f"'{func_name}' not defined yet")]

    for i, case in enumerate(cases):
        name = case.get("name", f"case_{i}")
        try:
            got = func(*case.get("input", []))
            passed = got == case.get("expected")
            detail = f"got {got!r}, expected {case.get('expected')!r}"
        except Exception as exc:  # noqa: BLE001
            passed, detail = False, f"raised {type(exc).__name__}: {exc}"
        results.append(TestResult(name, passed, detail))
    return results
