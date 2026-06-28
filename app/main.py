"""Entry point: build the runtime, load the corpus, run one problem.

Run it with:
    python -m app.main
    python -m app.main --problem-id two-sum
    python -m app.main --problem "Given an array, return ..."

By default it uses the stub LLM (no Ollama required) so you can watch the
whole pipeline execute immediately. Flip USE_STUB_LLM=false in `.env`
once Ollama is running to get real output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import Config
from app.graph import build_graph
from app.memory.embeddings import build_embedder
from app.memory.vector_store import VectorStore
from app.models.ollama_client import OllamaClient
from app.state import AgentState, Context
from app.tools.leetcode_utils import scan_solutions


def load_prompts(config: Config) -> dict[str, str]:
    """Read every *.txt prompt template into a name -> text dict."""
    prompts: dict[str, str] = {}
    for path in Path(config.prompts_dir).glob("*.txt"):
        prompts[path.stem] = path.read_text(encoding="utf-8")
    return prompts


def build_retriever(config: Config) -> VectorStore:
    """Load the user's past solutions and index them for retrieval."""
    store = VectorStore(embedder=build_embedder(config))
    docs = scan_solutions(config.leetcode_dir)
    if docs:
        store.add_documents(
            texts=[d.embed_text() for d in docs],
            payloads=[d.to_payload() for d in docs],
        )
    print(f"[setup] indexed {len(store)} past solution(s) from {config.leetcode_dir}")
    return store


def build_context(config: Config) -> Context:
    """Assemble the shared runtime dependencies for the agents."""
    return Context(
        config=config,
        llm=OllamaClient(config),
        retriever=build_retriever(config),
        prompts=load_prompts(config),
    )


def load_problem(config: Config, problem_id: str | None, problem_text: str | None):
    """Resolve the problem to work on, from data/problems.json or CLI text."""
    if problem_text:
        return problem_id or "", problem_text

    problems_path = Path(config.data_dir) / "problems.json"
    problems = json.loads(problems_path.read_text()) if problems_path.exists() else {}

    if problem_id and problem_id in problems:
        return problem_id, problems[problem_id]["statement"]
    if problems:
        # Default to the first problem in the file.
        first_id = next(iter(problems))
        return first_id, problems[first_id]["statement"]

    raise SystemExit(
        "No problem given. Add one to data/problems.json or pass --problem."
    )


def print_summary(state: AgentState) -> None:
    print("\n" + "=" * 60)
    print("RESULT (intuition guidance — intentionally not a full solution)")
    print("=" * 60)
    print("\n## Understanding\n" + state.problem_understanding)
    print("\n## Patterns\n" + (", ".join(state.patterns) or "(none parsed)"))
    print("\n## Similar past problems")
    for s in state.similar_problems:
        print(f"  - {s.name} (score {s.score:.2f})")
    print("\n## Plan / hints\n" + state.plan)
    print("\n## Scaffold\n" + state.scaffold_code)
    print("\n## Critique\n" + state.critique)
    if state.debug_notes:
        print("\n## Debug notes\n" + state.debug_notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="LeetCode Multi-Agent tutor")
    parser.add_argument("--problem-id", help="id from data/problems.json")
    parser.add_argument("--problem", help="raw problem statement text")
    args = parser.parse_args()

    config = Config.load()
    print(f"[setup] stub LLM: {config.use_stub_llm} | model: {config.model}")

    ctx = build_context(config)
    problem_id, problem_text = load_problem(config, args.problem_id, args.problem)

    state = AgentState(problem_text=problem_text, problem_id=problem_id)
    state.log("main", f"working on problem '{problem_id or '(inline)'}'")

    graph = build_graph(config)
    state = graph.run(state, ctx)

    print_summary(state)


if __name__ == "__main__":
    main()
