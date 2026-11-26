from __future__ import annotations

import pathlib
from typing import Any

from guardrails.base import BaseGuardrail, GuardrailResult
from tools import DEFAULT_PATH_TOOLS, FILE_TOOLS
from workspace import (
    SENSITIVE_DIRS,
    SENSITIVE_KEY_FILES,
    SENSITIVE_KEY_SUFFIXES,
    SENSITIVE_PATH_CHECKS,
    is_sensitive_path,
)

__all__ = [
    "DEFAULT_PATH_TOOLS",
    "FILE_TOOLS",
    "PathSandboxingGuardrail",
    "SENSITIVE_DIRS",
    "SENSITIVE_KEY_FILES",
    "SENSITIVE_KEY_SUFFIXES",
    "SENSITIVE_PATH_CHECKS",
    "is_sensitive_path",
]


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

    def clone(
        self,
        *,
        root_dir: str | pathlib.Path | None = None,
    ) -> PathSandboxingGuardrail:
        effective_root = root_dir if root_dir is not None else self.root_dir
        return PathSandboxingGuardrail(root_dir=effective_root, enabled=self.enabled)

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> GuardrailResult:
        if not self.enabled or tool_name not in FILE_TOOLS:
            return GuardrailResult(allowed=True)

        if "path" not in args and tool_name in DEFAULT_PATH_TOOLS:
            raw_path: Any = "."
        else:
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

        reason = is_sensitive_path(resolved, root)
        if reason is not None:
            return GuardrailResult(allowed=False, reason=reason)

        return GuardrailResult(allowed=True)
