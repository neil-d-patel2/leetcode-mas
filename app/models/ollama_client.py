"""A thin client for a local Ollama LLM server.

Two modes:
  - Real mode: POSTs to Ollama's HTTP API at `config.ollama_host`.
  - Stub mode: returns canned, role-appropriate text without any network
    call. This is the default (see Config.use_stub_llm) so the entire
    pipeline runs before you've installed Ollama or pulled a model.

Switch to real mode by setting USE_STUB_LLM=false in your `.env` (and
making sure `ollama serve` is running with the model pulled).

We import `requests` lazily inside the method so that stub mode works
even if `requests` isn't installed yet.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Config


class OllamaClient:
    def __init__(self, config: "Config") -> None:
        self.config = config

    def generate(self, prompt: str, system: str | None = None, tag: str = "llm") -> str:
        """Return the model's completion for `prompt`.

        Args:
            prompt: The fully-rendered user prompt.
            system: Optional system instruction.
            tag: A short label (usually the agent name) used to pick
                 sensible stub output and to make logs readable.
        """
        if self.config.use_stub_llm:
            return self._stub(prompt, system, tag)
        return self._call_ollama(prompt, system)

    # ------------------------------------------------------------------
    # Real Ollama call
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str, system: str | None) -> str:
        import requests  # imported lazily; only needed in real mode

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        if system:
            payload["system"] = system

        url = f"{self.config.ollama_host}/api/generate"
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    # ------------------------------------------------------------------
    # Stub mode
    # ------------------------------------------------------------------
    def _stub(self, prompt: str, system: str | None, tag: str) -> str:
        """Canned output so the pipeline is runnable end-to-end.

        Each agent passes its own `tag`, so we can return text that looks
        like what that agent would produce. This is purely a placeholder
        to teach the data flow — replace by setting USE_STUB_LLM=false.
        """
        canned = {
            "problem": (
                "This problem asks you to process an input collection and "
                "return an aggregate result. Restate it in your own words and "
                "note what the output represents."
            ),
            "planner": (
                "Likely patterns: hashing for O(1) lookups; two-pointer or "
                "sliding window if the structure is a sequence. Start by asking "
                "what state you must track as you scan the input."
            ),
            "coder": (
                "def solve(nums):\n"
                "    # TODO: choose the data structure that gives fast lookups\n"
                "    seen = {}\n"
                "    for i, x in enumerate(nums):\n"
                "        # TODO: what do you check here before moving on?\n"
                "        pass\n"
                "    # TODO: return the result\n"
            ),
            "critic": (
                "The scaffold leaves the core decision to the learner (good). "
                "Make sure edge cases like empty input and duplicates are "
                "considered before filling in the TODOs."
            ),
            "debugger": (
                "If a case fails, re-check the loop invariant: what must be "
                "true about `seen` at each step? Trace one small input by hand."
            ),
        }
        body = canned.get(tag, "[stub] (no canned text for this agent)")
        return f"[STUB:{tag}] {body}"
