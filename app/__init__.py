"""LeetCode Multi-Agent System (MAS).

A spaced-repetition + intuition-building tool for LeetCode problems.

Design goal: this system intentionally does NOT produce full optimal
solutions. It guides the user toward *thinking* about a problem by:
  - understanding the problem,
  - identifying the relevant algorithmic patterns,
  - retrieving analogous problems the user has already solved,
  - and offering scaffolded (partial) code plus next reasoning steps.

The package is organised as a small, LangGraph-style pipeline of
specialised "agents" that all read from and write to one shared state
object. Read `app/graph.py` to see how the agents are wired together.
"""
