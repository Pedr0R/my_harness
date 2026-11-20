from __future__ import annotations

import re
from typing import Any

from guardrails.base import BaseGuardrail, GuardrailResult


COMMAND_SAFETY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "rm_rf_dangerous_target",
        re.compile(
            r"(?i)\brm\s+"
            r"(?:-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*"
            r"|(?:-[a-z]*r[a-z]*|--recursive)\s+(?:-[a-z]*f[a-z]*|--force)"
            r"|(?:-[a-z]*f[a-z]*|--force)\s+(?:-[a-z]*r[a-z]*|--recursive))\s+"
            r"(?:--no-preserve-root\s+)?"
            r"(?:/(?:\*|\s|$|[;&|])|~(?:/|\s|$|[;&|])|\$HOME(?:/|\s|$|[;&|])"
            r"|\*(?:\s|$|[;&|])|[a-z]:[\\/]*(?:\s|$|[;&|]))"
        ),
    ),
    (
        "win_rmdir_dangerous_target",
        re.compile(
            r"(?i)\b(?:rmdir|rd)\b"
            r"(?:[^;&|\n]*\s/s\b[^;&|\n]*(?:[a-z]:[\\/]*(?:\s|$|[;&|])|\s\\(?:\s|$|[;&|])|\s\*(?:\s|$|[;&|])|%USERPROFILE%)"
            r"|\s+(?:[a-z]:[\\/]*|\\|\*|%USERPROFILE%)\s+[^;&|\n]*/s\b)"
        ),
    ),
    (
        "ps_remove_item_dangerous_target",
        re.compile(
            r"(?i)\bRemove-Item\b(?=[^;&|\n]*-Recurse\b)(?=[^;&|\n]*-Force\b)"
            r"[^;&|\n]*(?:(?:\s|['\"])(?:[a-z]:[\\/]*|/|\\|~|\*|\$HOME)(?:['\"]|\s|$|[;&|]))"
        ),
    ),
    (
        "disk_mkfs",
        re.compile(r"(?i)(?:^|[;&|]\s*)(?:sudo\s+)?mkfs(?:\.[a-z0-9]+)?\b"),
    ),
    (
        "disk_dd_raw_device",
        re.compile(r"(?i)\bdd\b[^;&|\n]*\bof=/dev/"),
    ),
    (
        "disk_win_format",
        re.compile(r"(?i)(?:^|[;&|]\s*)format(?:\.com)?\s+[a-z]:"),
    ),
    (
        "disk_win_diskpart",
        re.compile(r"(?i)(?:^|[;&|]\s*)diskpart(?:\.exe)?\b"),
    ),
    (
        "pipe_to_shell_unix",
        re.compile(
            r"(?i)\b(?:curl|wget)\b[^|\n]*\|\s*(?:sudo\s+)?"
            r"(?:sh|bash|zsh|dash|ksh|pwsh|powershell|iex|Invoke-Expression)\b"
        ),
    ),
    (
        "pipe_to_shell_powershell",
        re.compile(
            r"(?i)\b(?:iwr|Invoke-WebRequest|irm|Invoke-RestMethod)\b[^|\n]*\|\s*"
            r"(?:iex|Invoke-Expression|pwsh|powershell)\b"
        ),
    ),
    (
        "git_reset_hard",
        re.compile(r"(?i)\bgit\s+reset\b[^;&|\n]*--hard\b"),
    ),
    (
        "git_clean_fd",
        re.compile(
            r"(?i)\bgit\s+clean\b[^;&|\n]*"
            r"(?:-[a-z]*f[a-z]*d|-[a-z]*d[a-z]*f|-f\s+-d|-d\s+-f)"
        ),
    ),
    (
        "git_push_force",
        re.compile(r"(?i)\bgit\s+push\b[^;&|\n]*(?:\s|^)(?:--force|-f)(?:\s|$|[;&|])"),
    ),
    (
        "system_power_control",
        re.compile(
            r"(?i)(?:^|[;&|]\s*)(?:sudo\s+)?"
            r"(?:shutdown|reboot|poweroff|halt)(?:\.exe)?(?:\s|$|[;&|])"
        ),
    ),
    (
        "fork_bomb",
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
    ),
    (
        "sensitive_file_shell_read",
        re.compile(
            r"(?i)\b(?:cat|type|more|less|head|tail|Get-Content|gc)\b[^;&|\n]*"
            r"(?:(?:^|[\s'\"/\\])\.env(?:\.[A-Za-z0-9._-]+)?(?:$|[\s'\"&|;])"
            r"|\.ssh[/\\]id_[A-Za-z0-9._-]+"
            r"|/etc/shadow\b)"
        ),
    ),
)


class CommandSafetyGuardrail(BaseGuardrail):
    name = "command_safety"

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> GuardrailResult:
        if not self.enabled or tool_name != "bash":
            return GuardrailResult(allowed=True)

        cmd = args.get("cmd")
        if not isinstance(cmd, str):
            return GuardrailResult(
                allowed=False,
                reason="Blocked by safety rules.",
            )

        for _rule_id, pattern in COMMAND_SAFETY_RULES:
            if pattern.search(cmd):
                return GuardrailResult(
                    allowed=False,
                    reason="Blocked by safety rules.",
                )

        return GuardrailResult(allowed=True)
