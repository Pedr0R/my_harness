from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    modified_args: dict[str, Any] | None = None


class BaseGuardrail:
    name: str = "base"

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> GuardrailResult:
        return GuardrailResult(allowed=True)

    def sanitize_output(self, tool_name: str, output: str) -> str:
        return output

    def reset(self) -> None:
        return None
