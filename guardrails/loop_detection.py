from __future__ import annotations

import json
from typing import Any

from guardrails.base import BaseGuardrail, GuardrailResult


class LoopDetectionGuardrail(BaseGuardrail):
    name = "loop_detection"

    def __init__(
        self,
        max_repeats: int = 3,
        block_on_loop: bool = True,
        enabled: bool = True,
    ) -> None:
        if max_repeats < 1:
            raise ValueError("max_repeats deve ser maior ou igual a 1.")
        super().__init__(enabled=enabled)
        self.max_repeats = max_repeats
        self.block_on_loop = block_on_loop
        self._last_signature: str | None = None
        self._repeat_count: int = 0
        self._warn_pending: bool = False

    def _signature(self, tool_name: str, args: dict[str, Any]) -> str:
        return json.dumps(
            {"tool": tool_name, "args": args},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(allowed=True)

        self._warn_pending = False
        signature = self._signature(tool_name, args)
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1

        if self._repeat_count >= self.max_repeats:
            if self.block_on_loop:
                return GuardrailResult(
                    allowed=False,
                    reason="Blocked by loop detection. Change strategy.",
                )
            self._warn_pending = True

        return GuardrailResult(allowed=True)

    def sanitize_output(self, tool_name: str, output: str) -> str:
        if not self.enabled or not self._warn_pending:
            return output

        self._warn_pending = False
        return f"{output}\n[Guardrail warning] Repeated tool loop detected. Change strategy."

    def reset(self) -> None:
        self._last_signature = None
        self._repeat_count = 0
        self._warn_pending = False
