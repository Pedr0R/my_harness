from __future__ import annotations

from dataclasses import dataclass, field
import json
import pathlib
from typing import Any, Callable

from guardrails import (
    BaseGuardrail,
    reset_guardrails,
    run_guardrail_checks,
    sanitize_with_guardrails,
)
from providers import ModelProvider, get_provider
from session import load_session_file, save_session_file
from tools import TOOLS, TOOL_SCHEMAS, _truncate
from workspace import CheckpointResult, WorkspaceCheckpoint


__all__ = [
    "Agent",
    "AgentLimitError",
    "discover_context",
]


_MAX_CHECKPOINTS = 20


class AgentLimitError(Exception):
    """Exceção levantada quando o agente atinge o limite configurado de turnos ou custo."""


def _is_limit_stop(exc: BaseException) -> bool:
    """Identifica paradas esperadas por limite de turnos ou custo."""
    return (
        isinstance(exc, (AgentLimitError, StopIteration))
        or isinstance(getattr(exc, "__cause__", None), StopIteration)
    )


def discover_context(cwd: pathlib.Path | None = None) -> str:
    """Varre os diretórios da raiz até a pasta atual em busca de arquivos AGENTS.md."""
    target_cwd = cwd if cwd is not None else pathlib.Path.cwd()
    parts = []
    for p in reversed([target_cwd, *target_cwd.parents]):
        md = p / "AGENTS.md"
        if md.exists():
            parts.append(f"<ctx path='{md}'>\n{md.read_text(encoding='utf-8', errors='replace')}\n</ctx>")
    return "\n".join(parts)


@dataclass
class Agent:
    model: ModelProvider = field(default_factory=get_provider)
    name: str = "Agent"
    instructions: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    n_turns: int = 0
    cost: float = 0.0
    max_turns: int = 50
    max_cost: float = 5.00
    compact_at_tokens: int = 120_000
    guardrails: list[BaseGuardrail] = field(default_factory=list)
    tools: dict[str, Callable[..., Any]] = field(default_factory=lambda: dict(TOOLS))
    tool_schemas: list[dict[str, Any]] = field(default_factory=lambda: list(TOOL_SCHEMAS))
    checkpointing: bool = False
    workspace_dir: str | pathlib.Path | None = None
    _checkpoints: list[WorkspaceCheckpoint] = field(default_factory=list, repr=False)
    last_rollback_message: str | None = field(default=None, repr=False)

    def _record_checkpoint(self, checkpoint: WorkspaceCheckpoint) -> None:
        checkpoint.commit()
        self._checkpoints.append(checkpoint)
        if len(self._checkpoints) > _MAX_CHECKPOINTS:
            del self._checkpoints[:-_MAX_CHECKPOINTS]

    def undo_last_turn(self) -> CheckpointResult:
        """Desfaz as alterações de arquivos do último turno concluído usando o checkpoint mais recente."""
        if not self._checkpoints:
            return CheckpointResult(
                ok=False,
                active=False,
                message="Nenhum checkpoint disponível para desfazer.",
            )
        checkpoint = self._checkpoints.pop()
        result = checkpoint.rollback()
        if result.ok:
            self.last_rollback_message = result.message
        return result

    def _token_estimate(self) -> int:
        """Estima o volume de tokens acumulado no histórico de mensagens."""
        return sum(len(json.dumps(m, default=str)) for m in self.messages)

    async def _check_limits(self) -> None:
        """Garante que os limites de turnos e custo financeiro não sejam ultrapassados."""
        if self.n_turns >= self.max_turns:
            raise AgentLimitError(f"max_turns atingido ({self.max_turns})")
        if self.cost >= self.max_cost:
            raise AgentLimitError(f"max_cost atingido (${self.max_cost:.2f})")

    async def _maybe_compact(self) -> None:
        """Compacta o contexto quando o histórico atinge o limiar configurado de tokens."""
        if self._token_estimate() < self.compact_at_tokens:
            return

        summary = await self.model.complete(
            self.messages
            + [
                {
                    "role": "user",
                    "content": "Resuma a conversa. Preserve decisões tomadas e questões não resolvidas.",
                }
            ],
            tools=[],
        )
        content = summary.get("content", "") if hasattr(summary, "get") else getattr(summary, "content", "")
        keep = max(4, int(len(self.messages) * 0.30))
        self.messages = [
            self.messages[0],
            {"role": "assistant", "content": content},
            *self.messages[-keep:],
        ]

    def reset_session(self) -> None:
        """Limpa a memória de mensagens da sessão atual."""
        self.messages.clear()
        self.n_turns = 0
        self._checkpoints.clear()
        self.last_rollback_message = None
        reset_guardrails(self.guardrails)

    def get_session_memory(self) -> dict[str, Any]:
        """Retorna a memória da sessão atual em formato de dicionário."""
        return {
            "agent_name": self.name,
            "turns": self.n_turns,
            "estimated_cost": self.cost,
            "messages": list(self.messages),
        }

    def save_to_file(
        self,
        path: str | pathlib.Path,
        metadata: dict[str, Any] | None = None,
    ) -> pathlib.Path:
        """Salva atomicamente o estado da sessão e das mensagens do agente em disco (JSON)."""
        return save_session_file(
            path,
            n_turns=self.n_turns,
            cost=self.cost,
            messages=self.messages,
            metadata=metadata,
        )

    def load_from_file(self, path: str | pathlib.Path) -> dict[str, Any]:
        """Carrega o estado salvo de um arquivo JSON e restaura messages, n_turns e cost."""
        data, messages, n_turns, cost = load_session_file(path)
        self.messages = messages
        self.n_turns = n_turns
        self.cost = cost
        self._checkpoints.clear()
        self.last_rollback_message = None
        reset_guardrails(self.guardrails)
        return data

    def _execute_tool_call(self, call_name: str, call_args: Any) -> str:
        allowed, blocked_message, effective_args = run_guardrail_checks(
            self.guardrails, call_name, call_args
        )
        if not allowed:
            raw_output: Any = blocked_message or "Blocked by safety rules."
        else:
            try:
                fn = self.tools.get(call_name)
                if not fn:
                    raw_output = f"ERRO: Ferramenta desconhecida '{call_name}'"
                else:
                    raw_output = fn(**effective_args)
            except Exception as exc:
                raw_output = f"ERRO: {type(exc).__name__}: {exc}"

        sanitized = sanitize_with_guardrails(self.guardrails, call_name, str(raw_output))
        return _truncate(sanitized)

    async def run(self, task: str) -> str:
        """Executa um turno de raciocínio e ação do agente para atender à tarefa."""
        self.last_rollback_message = None

        if not self.messages:
            default_instructions = (
                f"Você é um agente de engenharia de software conciso e prestativo.\n\n{discover_context()}"
            )
            system_content = self.instructions or default_instructions
            self.messages.append({"role": "system", "content": system_content})

        self.messages.append({"role": "user", "content": task})

        checkpoint: WorkspaceCheckpoint | None = None
        if self.checkpointing:
            checkpoint = WorkspaceCheckpoint(root_dir=self.workspace_dir)
            checkpoint.snapshot()

        try:
            while True:
                await self._check_limits()
                await self._maybe_compact()
                resp = await self.model.complete(self.messages, tools=self.tool_schemas)
                self.n_turns += 1
                self.cost += resp.get("cost", 0.0)

                msg_entry = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
                self.messages.append(msg_entry)

                tool_calls = resp.get("tool_calls") or []
                if not tool_calls:
                    if checkpoint is not None and checkpoint.is_active:
                        self._record_checkpoint(checkpoint)
                    return resp.get("content", "")

                for call in tool_calls:
                    call_name = call["name"] if isinstance(call, dict) else call.name
                    call_args = call["args"] if isinstance(call, dict) else call.args
                    call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", f"call_{call_name}")

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call_name,
                            "content": self._execute_tool_call(call_name, call_args),
                        }
                    )
        except BaseException as exc:
            if checkpoint is not None and checkpoint.is_active:
                if _is_limit_stop(exc):
                    self._record_checkpoint(checkpoint)
                else:
                    result = checkpoint.rollback()
                    self.last_rollback_message = result.message
            raise
