"""Shared API-key loader for all providers.

Lookup order for a named key (e.g. ``MASSIVEKEY``):
    1. Environment variable with that name.
    2. ``<repo-root>/key.md`` entry of the form:  NAME="value"

``key.md`` is gitignored. Raising rather than returning ``""`` means an
unauthenticated request never reaches the wire.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
KEY_FILE: Path = REPO_ROOT / "key.md"


def load_key(name: str) -> str:
    """Return the key named ``name`` from env or ``key.md``; raise if missing."""
    env = os.environ.get(name)
    if env:
        return env.strip()

    if KEY_FILE.is_file():
        text = KEY_FILE.read_text(encoding="utf-8")
        match = re.search(rf"{re.escape(name)}\s*=\s*\"([^\"]+)\"", text)
        if match:
            return match.group(1).strip()

    raise RuntimeError(
        f"{name} not found — set the env var or add to key.md: "
        f'{name}="<your token>"'
    )
