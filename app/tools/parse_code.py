"""Pull clean code out of an LLM's free-form text response.

LLMs usually wrap code in Markdown fences (```python ... ```). The coder
agent needs just the code (e.g. to count TODOs or hand to the test
agent), so this extracts the first fenced block, falling back to the raw
text if no fence is present.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Return the first fenced code block, or the stripped text if none."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def count_todos(code: str) -> int:
    """How many `TODO` markers remain — a proxy for 'how scaffolded'."""
    return len(re.findall(r"#\s*TODO", code, flags=re.IGNORECASE))
