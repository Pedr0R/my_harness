from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
import stat
from typing import Any, Callable

from workspace import is_sensitive_path


IGNORED_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache", ".sessions"}
)


@dataclass(frozen=True)
class ToolSpec:
    """Contrato único que une a função executável (handler) ao seu JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    checks_path: bool = False
    default_path: str | None = None

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _truncate(s: str, n: int = 25_000) -> str:
    """Trunca strings longas para não sobrecarregar o contexto."""
    return s if len(s) <= n else s[:n] + "\n[truncado]"


def _is_link_or_junction(path: str) -> bool:
    if os.name == "nt":
        try:
            return bool(os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except OSError:
            return False

    return os.path.islink(path)


def _resolve_safe_walk_entry(
    entry_str: str,
    entry_path: pathlib.Path,
    search_root: pathlib.Path,
) -> pathlib.Path | None:
    resolved = entry_path

    if _is_link_or_junction(entry_str):
        try:
            resolved = entry_path.resolve()
        except OSError:
            return None

    if is_sensitive_path(resolved, search_root) is not None:
        return None

    return resolved


def _filter_walk_dirs(
    dirpath: str,
    current_dir: pathlib.Path,
    dirnames: list[str],
    search_root: pathlib.Path,
) -> list[str]:
    kept: list[str] = []

    for d in sorted(dirnames):
        if d in IGNORED_DIRS:
            continue

        dir_str = os.path.join(dirpath, d)
        resolved = _resolve_safe_walk_entry(dir_str, current_dir / d, search_root)

        if resolved is None or resolved.name in IGNORED_DIRS:
            continue

        kept.append(d)

    return kept
