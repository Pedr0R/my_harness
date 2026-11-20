from __future__ import annotations

import pathlib
from typing import Any, Callable

from guardrails.base import BaseGuardrail, GuardrailResult


FILE_TOOLS: frozenset[str] = frozenset({"read_file", "write_file", "search_replace"})

SENSITIVE_DIRS: frozenset[str] = frozenset({".git", ".ssh", ".gnupg", ".aws"})

SENSITIVE_KEY_FILES: frozenset[str] = frozenset(
    {"id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
)

SENSITIVE_KEY_SUFFIXES: frozenset[str] = frozenset({".pem", ".key"})


def _check_sensitive_dir(rel_parts: tuple[str, ...], _filename: str, _suffix: str) -> str | None:
    if any(part in SENSITIVE_DIRS for part in rel_parts):
        return "Blocked by path sandbox (sensitive path)."
    return None


def _check_env_file(_rel_parts: tuple[str, ...], filename: str, _suffix: str) -> str | None:
    if filename == ".env" or filename.startswith(".env."):
        return "Blocked by path sandbox (sensitive path)."
    return None


def _check_private_key(_rel_parts: tuple[str, ...], filename: str, suffix: str) -> str | None:
    if filename in SENSITIVE_KEY_FILES or suffix in SENSITIVE_KEY_SUFFIXES:
        return "Blocked by path sandbox (sensitive path)."
    return None


SENSITIVE_PATH_CHECKS: tuple[Callable[[tuple[str, ...], str, str], str | None], ...] = (
    _check_sensitive_dir,
    _check_env_file,
    _check_private_key,
)


class PathSandboxingGuardrail(BaseGuardrail):
    name = "path_sandboxing"

    def __init__(
        self,
        root_dir: str | pathlib.Path | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.root_dir: pathlib.Path | None = (
            pathlib.Path(root_dir).resolve() if root_dir is not None else None
        )

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> GuardrailResult:
        if not self.enabled or tool_name not in FILE_TOOLS:
            return GuardrailResult(allowed=True)

        raw_path = args.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return GuardrailResult(
                allowed=False,
                reason="Blocked by path sandbox (invalid path).",
            )

        root = self.root_dir if self.root_dir is not None else pathlib.Path.cwd().resolve()
        try:
            resolved = (root / pathlib.Path(raw_path)).resolve()
        except OSError:
            return GuardrailResult(
                allowed=False,
                reason="Blocked by path sandbox (invalid path).",
            )

        if not resolved.is_relative_to(root):
            return GuardrailResult(
                allowed=False,
                reason="Blocked by path sandbox (outside workspace).",
            )

        rel = resolved.relative_to(root)
        lower_parts = tuple(part.lower() for part in rel.parts)
        lower_name = resolved.name.lower()
        lower_suffix = resolved.suffix.lower()

        for check in SENSITIVE_PATH_CHECKS:
            reason = check(lower_parts, lower_name, lower_suffix)
            if reason is not None:
                return GuardrailResult(allowed=False, reason=reason)

        return GuardrailResult(allowed=True)
