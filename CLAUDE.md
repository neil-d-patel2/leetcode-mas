# Project Memory

## Overview
LeetCode Multi-Agent System (MAS): a spaced-repetition + intuition-building
tutor for LeetCode problems. It deliberately does NOT produce full optimal
solutions — it guides toward *thinking* via problem understanding, pattern
recognition, retrieval of the user's own past solved problems, and partial
scaffolds (`# TODO`-driven) instead of answers. This is a personal learning
project; clarity of the reasoning flow matters more than code polish.

## Architecture
- One shared `AgentState` (data) + `Context` (runtime deps: config, llm,
  retriever, prompts) flow through the pipeline. Both in `app/state.py`.
- Agents are plain functions `run(state, ctx) -> state` in `app/agents/`:
  problem -> planner -> coder -> critic -> test -> (router) -> debugger.
- `app/graph.py` is a hand-written LangGraph-style runner (nodes, static +
  conditional edges, `END` sentinel, step cap). Built via `build_graph(config)`.
  The post-`test` router (`_make_after_test`) loops back through debugger ->
  coder while work is needed and `loop_count < config.max_loops`.
- Retrieval: `app/memory/embeddings.py` (HashingEmbedder default = dep-free
  bag-of-words; OllamaEmbedder for real) + `vector_store.py` (in-memory
  cosine). Corpus built from the `leetcode/` folder by
  `app/tools/leetcode_utils.py` (`scan_solutions`).
- LLM: `app/models/ollama_client.py` — real Ollama HTTP API or an offline
  stub (per-agent canned text). Stub is the DEFAULT so everything runs with
  zero third-party deps.
- Prompts: `app/prompts/*.txt` use `string.Template` `$placeholders` (chosen
  so code examples with `{}` don't clash). Rendered via `app.agents.render`.
- Entry point: `app/main.py` (`python -m app.main`).

## Development Commands
- Run pipeline (stub LLM): `python -m app.main --problem-id two-sum`
- Inline problem: `python -m app.main --problem "..."`
- Tests: `python -m pytest -q` (requires `pip install -r requirements.txt`).

## Build Process
None. Pure Python, no build step. Stub mode needs only the standard library.

## Deployment Process
N/A (local CLI tool).

## Coding Standards
- `from __future__ import annotations` at top of modules; PEP 604 hints.
- Agents stay pure-ish: read state, call `ctx.llm`, write state, `state.log()`.
- Heavy deps (`requests`) imported lazily so stub mode never needs them.
- Type-only imports under `TYPE_CHECKING` to avoid import cycles.

## Known Issues
- Stub LLM output carries a `[STUB:...]` prefix, so the generated scaffold
  isn't valid Python — the `parses` static check fails in stub mode. Expected;
  real mode produces clean code.
- problem_agent currently stores the whole LLM response in BOTH
  `problem_understanding` and `constraints` (TODO: parse sections apart).
- planner derives `patterns` by keyword-matching the plan text (`_guess_patterns`).

## Lessons Learned
- The 6-agent pipeline runs end-to-end and exercises the debugger loop when
  sample cases fail (verified with two-sum). 10 unit/integration tests pass.

## Decisions
- Wrote a from-scratch graph runner instead of depending on `langgraph` — the
  mental model is identical but every moving part is visible (learning goal).
- In-memory vector store + hashing embedder chosen for zero-dependency runs;
  swap to FAISS/Chroma + OllamaEmbedder behind the same interface later.

## Notes
- Config via env / `.env` (`Config.load()`); copy `.env.example` to `.env`.
  Set `USE_STUB_LLM=false` + run `ollama serve` for real models.
- Point at real solutions via `LEETCODE_DIR` or by filling `leetcode/`.
  Tag files with a `# tags: ...` line; else tags inferred from folder names.
- Spec mentions `experiments/` and `data/solved_cache.json` for spaced
  repetition — cache file exists as a stub; repetition scheduling not built yet.
