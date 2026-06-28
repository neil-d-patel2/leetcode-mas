"""Central configuration for the multi-agent system.

Everything that might change between machines or runs lives here so the
rest of the code never has to guess at paths, model names, or URLs.

Values are read from environment variables (optionally loaded from a
`.env` file) with sensible defaults, so the project runs out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Project root = the directory that contains the `app/` package.
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load key=value pairs from a `.env` file into os.environ.

    We do this by hand (instead of requiring python-dotenv) so the
    project has zero mandatory dependencies for a first run. Lines that
    are blank or start with `#` are ignored. Existing environment
    variables are never overwritten.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    """Runtime configuration, populated from the environment."""

    # --- Ollama / LLM settings ---------------------------------------
    # Base URL of the local Ollama server.
    ollama_host: str = "http://localhost:11434"
    # Which model to use for generation. Pull it with `ollama pull <name>`.
    model: str = "llama3"
    # Model used to produce embeddings for retrieval (see app/memory).
    embed_model: str = "nomic-embed-text"
    # If True, never call Ollama; agents return canned "stub" text.
    # This lets the whole pipeline run before you've set up Ollama.
    use_stub_llm: bool = True
    # Sampling temperature for generation.
    temperature: float = 0.4

    # --- Retrieval settings ------------------------------------------
    # Folder containing the user's past solved LeetCode solutions.
    # This is the "pattern library" the system retrieves from.
    leetcode_dir: Path = field(default_factory=lambda: ROOT / "leetcode")
    # How many similar past problems to retrieve per query.
    retrieval_top_k: int = 3

    # --- Data / cache locations --------------------------------------
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    prompts_dir: Path = field(default_factory=lambda: ROOT / "app" / "prompts")

    # --- Orchestration settings --------------------------------------
    # Safety cap on how many times the critic/debugger loop may run.
    max_loops: int = 2

    @classmethod
    def load(cls) -> "Config":
        """Build a Config from environment variables (and .env)."""
        _load_dotenv()
        return cls(
            ollama_host=os.environ.get("OLLAMA_HOST", cls.ollama_host),
            model=os.environ.get("LLM_MODEL", cls.model),
            embed_model=os.environ.get("EMBED_MODEL", cls.embed_model),
            use_stub_llm=_env_bool("USE_STUB_LLM", cls.use_stub_llm),
            temperature=float(os.environ.get("LLM_TEMPERATURE", cls.temperature)),
            leetcode_dir=Path(os.environ.get("LEETCODE_DIR", ROOT / "leetcode")),
            retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", cls.retrieval_top_k)),
            max_loops=int(os.environ.get("MAX_LOOPS", cls.max_loops)),
        )
