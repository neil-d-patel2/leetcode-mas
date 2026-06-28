# leetcode-mas

A multi-agent system that helps you **build intuition** for LeetCode
problems using notes from your own past solved problems — **without
handing you the solution**.

It is a spaced-repetition + pattern-recognition tutor, not a solution
generator. Given a problem it: understands it, names the likely
algorithmic patterns, retrieves analogous problems you've already solved,
explains the intuition, and offers a **partial scaffold** (full of
guiding `# TODO`s) rather than working code.

## Quick start

```bash
# Runs immediately with a stub LLM — no Ollama needed.
python -m app.main --problem-id two-sum

# Run the tests.
pip install -r requirements.txt
python -m pytest -q
```

To use a real local model, install [Ollama](https://ollama.com), pull a
model, copy `.env.example` to `.env`, and set `USE_STUB_LLM=false`.

## How it works

One shared `AgentState` flows through a small graph of specialised agents:

```
problem -> planner -> coder -> critic -> test --(needs work?)--> debugger
   |          |          |        |        |                        |
understand  patterns + scaffold  guards   sample                 guiding
 problem    retrieval  (TODOs)  reasoning checks                 questions
                                & over-                             |
                                completeness                   loops back
                                                               to coder
```

- **State** — `app/state.py` (`AgentState` = data, `Context` = runtime deps).
- **Agents** — `app/agents/*.py`, each a `run(state, ctx) -> state` function.
- **Orchestration** — `app/graph.py`, a from-scratch LangGraph-style runner.
- **Retrieval** — `app/memory/` (embeddings + vector store) over your
  `leetcode/` folder, indexed by `app/tools/leetcode_utils.py`.
- **Prompts** — `app/prompts/*.txt`, using `$placeholder` templating.
- **LLM** — `app/models/ollama_client.py` (real Ollama or offline stub).

See `CLAUDE.md` for architecture notes and conventions.

## Pointing it at your solutions

Drop your solved solutions into `leetcode/` (any of `.py .java .cpp .js`
…), or set `LEETCODE_DIR` in `.env`. Add a `# tags: sliding-window, dp`
line near the top of a file to tag it; otherwise tags are inferred from
the folder names. These are used as analogies only — never copied.
