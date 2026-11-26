from __future__ import annotations

import pathlib
import re

from guardrails.base import BaseGuardrail


SECRET_VALUE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
    ),
    (
        "google_api_key",
        re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    ),
    (
        "sk_secret_key",
        re.compile(r"sk-(?:proj-|ant-|live-|test-)?[0-9A-Za-z\-_]{16,}"),
    ),
    (
        "github_classic_token",
        re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[0-9A-Za-z]{20,}"),
    ),
    (
        "github_fine_grained_token",
        re.compile(r"github_pat_[0-9A-Za-z_]{20,}"),
    ),
    (
        "xai_api_key",
        re.compile(r"xai-[0-9A-Za-z\-_]{20,}"),
    ),
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
)

SENSITIVE_ENV_ASSIGNMENT_RE = re.compile(
    r"(\b(?:GEMINI|OPENAI|GROK|DEEPSEEK|ANTHROPIC|AWS|GITHUB|API|AUTH|ACCESS|SECRET|PRIVATE)"
    r"_[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*[:=]\s*)(\S+)",
    flags=re.IGNORECASE,
)


class SecretsRedactionGuardrail(BaseGuardrail):
    name = "secrets_redaction"

    def __init__(self, replacement: str = "[REDACTED]", enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.replacement = replacement

    def sanitize_output(self, tool_name: str, output: str) -> str:
        if not self.enabled or not output:
            return output

        redacted = output
        for _rule_id, pattern in SECRET_VALUE_RULES:
            redacted = pattern.sub(lambda _match: self.replacement, redacted)

        redacted = SENSITIVE_ENV_ASSIGNMENT_RE.sub(
            lambda match: f"{match.group(1)}{self.replacement}",
            redacted,
        )
        return redacted

    def clone(
        self,
        *,
        root_dir: str | pathlib.Path | None = None,
    ) -> SecretsRedactionGuardrail:
        del root_dir
        return SecretsRedactionGuardrail(
            replacement=self.replacement,
            enabled=self.enabled,
        )
