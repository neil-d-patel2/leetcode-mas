"""Tests for the code-execution helpers in app/tools/run_tests.py."""

from app.tools.run_tests import run_sample_cases, static_check


def test_static_check_flags_valid_scaffold():
    code = "def solve(nums):\n    # TODO: implement\n    pass\n"
    results = {r.name: r for r in static_check(code)}
    assert results["parses"].passed
    assert results["is_scaffold"].passed  # has a TODO


def test_static_check_detects_syntax_error():
    results = {r.name: r for r in static_check("def solve(:\n  pass")}
    assert not results["parses"].passed


def test_run_sample_cases_passes_for_correct_impl():
    code = (
        "def solve(nums, target):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        if target - x in seen:\n"
        "            return [seen[target - x], i]\n"
        "        seen[x] = i\n"
    )
    cases = [{"name": "basic", "input": [[2, 7, 11, 15], 9], "expected": [0, 1]}]
    results = run_sample_cases(code, "solve", cases)
    assert results[0].passed


def test_run_sample_cases_reports_missing_function():
    results = run_sample_cases("x = 1", "solve", [{"input": [], "expected": 1}])
    assert not results[0].passed
