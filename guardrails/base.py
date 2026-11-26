from __future__ import annotations

import copy
from dataclasses import dataclass
import pathlib
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

    def clone(self, *, root_dir: str | pathlib.Path | None = None) -> BaseGuardrail:
        del root_dir
        cloned = copy.deepcopy(self)
        cloned.reset()
        return cloned


def _format_blocked_reason(reason: str | None) -> str:
    if not reason:
        return "Blocked by safety rules."
    if reason.lower().startswith("blocked"):
        return reason
    return f"Blocked by guardrail: {reason}"


def run_guardrail_checks(
    guardrails: list[BaseGuardrail],
    tool_name: str,
    args: Any,
) -> tuple[bool, str | None, Any]:
    effective_args = dict(args) if isinstance(args, dict) else args
    for guardrail in guardrails:
        if not getattr(guardrail, "enabled", True):
            continue
        check = guardrail.check_tool_call(tool_name, effective_args)
        if check.modified_args is not None:
            effective_args = check.modified_args
        if not check.allowed:
            return False, _format_blocked_reason(check.reason), effective_args
    return True, None, effective_args


def sanitize_with_guardrails(
    guardrails: list[BaseGuardrail],
    tool_name: str,
    output: str,
) -> str:
    sanitized = output
    for guardrail in guardrails:
        if not getattr(guardrail, "enabled", True):
            continue
        sanitized = guardrail.sanitize_output(tool_name, sanitized)
    return sanitized


def reset_guardrails(guardrails: list[BaseGuardrail]) -> None:
    for guardrail in guardrails:
        if hasattr(guardrail, "reset"):
            guardrail.reset()


def clone_guardrails(
    guardrails: list[BaseGuardrail],
    *,
    root_dir: str | pathlib.Path | None = None,
) -> list[BaseGuardrail]:
    cloned: list[BaseGuardrail] = []
    for guardrail in guardrails:
        if hasattr(guardrail, "clone"):
            cloned.append(guardrail.clone(root_dir=root_dir))
        else:
            item = copy.deepcopy(guardrail)
            if hasattr(item, "reset"):
                item.reset()
            cloned.append(item)
    return cloned
