from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib

from guardrails.base import BaseGuardrail
from guardrails.command_safety import CommandSafetyGuardrail
from guardrails.loop_detection import LoopDetectionGuardrail
from guardrails.path_sandbox import PathSandboxingGuardrail
from guardrails.secrets_redaction import SecretsRedactionGuardrail


_FALSY_ENV_VALUES = frozenset({"0", "false", "no", "off", "n"})
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "y"})


def _parse_bool_env(env_name: str, default: bool = True) -> bool:
    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in _FALSY_ENV_VALUES:
        return False
    if normalized in _TRUTHY_ENV_VALUES:
        return True
    raise ValueError(
        f"Valor booleano inválido em {env_name}={raw!r}. Use true/false ou 1/0."
    )


@dataclass
class GuardrailsConfig:
    path_sandboxing: bool = True
    command_safety: bool = True
    secrets_redaction: bool = True
    loop_detection: bool = True
    root_dir: pathlib.Path | None = None
    loop_max_repeats: int = 3

    @classmethod
    def from_env(cls) -> GuardrailsConfig:
        return cls(
            path_sandboxing=_parse_bool_env("GUARDRAIL_PATH_SANDBOXING", default=True),
            command_safety=_parse_bool_env("GUARDRAIL_COMMAND_SAFETY", default=True),
            secrets_redaction=_parse_bool_env("GUARDRAIL_SECRETS_REDACTION", default=True),
            loop_detection=_parse_bool_env("GUARDRAIL_LOOP_DETECTION", default=True),
        )


def build_guardrails(config: GuardrailsConfig | None = None) -> list[BaseGuardrail]:
    cfg = config if config is not None else GuardrailsConfig()
    guardrails: list[BaseGuardrail] = []
    if cfg.path_sandboxing:
        guardrails.append(PathSandboxingGuardrail(root_dir=cfg.root_dir))
    if cfg.command_safety:
        guardrails.append(CommandSafetyGuardrail())
    if cfg.secrets_redaction:
        guardrails.append(SecretsRedactionGuardrail())
    if cfg.loop_detection:
        guardrails.append(LoopDetectionGuardrail(max_repeats=cfg.loop_max_repeats))
    return guardrails
