"""Read the user's past solved LeetCode solutions into structured records.

This is the bridge between a folder of solution files on disk and the
vector store. We walk the folder, read each solution, and pull out light
metadata (a display name and any tags) so retrieval results are useful.

We intentionally extract PATTERNS and structure, not full answers — the
snippet we keep is short and meant only as an analogy/reminder.

Tag convention (optional): put a line like
    # tags: sliding-window, two-pointer
near the top of a solution and it will be picked up automatically.
Otherwise tags are inferred from the folder names in the path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# File extensions we treat as solution files.
CODE_EXTENSIONS = {".py", ".java", ".cpp", ".c", ".js", ".ts", ".go", ".rs"}

_TAGS_RE = re.compile(r"tags?\s*[:=]\s*(.+)", re.IGNORECASE)


@dataclass
class SolutionDoc:
    """One past solution loaded from disk."""

    name: str
    path: str
    text: str
    tags: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        """Metadata stored next to the vector (kept small on purpose)."""
        return {
            "name": self.name,
            "path": self.path,
            "tags": self.tags,
            "snippet": self.text[:400],
        }

    def embed_text(self) -> str:
        """The text we embed for similarity (name + tags + body)."""
        return f"{self.name}\n{' '.join(self.tags)}\n{self.text}"


def _extract_tags(text: str, path: Path, base: Path) -> list[str]:
    # Prefer an explicit `tags:` line if present.
    for line in text.splitlines()[:15]:
        match = _TAGS_RE.search(line)
        if match:
            return [t.strip() for t in re.split(r"[,;]", match.group(1)) if t.strip()]
    # Otherwise infer from folder names between base and the file.
    rel_parts = path.relative_to(base).parts[:-1]
    return [p.replace("_", "-").replace(" ", "-").lower() for p in rel_parts]


def _display_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def scan_solutions(leetcode_dir: Path) -> list[SolutionDoc]:
    """Recursively load all solution files under `leetcode_dir`.

    Returns an empty list (instead of raising) if the folder is missing,
    so the system still runs before you've pointed it at your solutions.
    """
    base = Path(leetcode_dir)
    if not base.exists():
        return []

    docs: list[SolutionDoc] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        docs.append(
            SolutionDoc(
                name=_display_name(path),
                path=str(path),
                text=text,
                tags=_extract_tags(text, path, base),
            )
        )
    return docs
