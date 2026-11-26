from __future__ import annotations

from collections.abc import AsyncIterator
import contextlib
from dataclasses import dataclass, field
from enum import Enum
import inspect
import json
import pathlib
from typing import Any, Awaitable, Callable

from guardrails import (
    BaseGuardrail,
    reset_guardrails,
    run_guardrail_checks,
    sanitize_with_guardrails,
)
from providers import ModelProvider, ModelResponse, get_provider, stream_or_complete
from session import load_session_file, save_session_file
from tools import APPROVAL_REQUIRED_TOOLS, TOOLS, TOOL_SCHEMAS, _truncate
from workspace import CheckpointResult, WorkspaceCheckpoint


__all__ = [
    "Agent",
    "AgentEvent",
    "AgentLimitError",
    "ApprovalAction",
    "ApprovalCallback",
    "ApprovalDecision",
    "ApprovalResult",
    "EventType",
    "discover_context",
]


_MAX_CHECKPOINTS = 20


class AgentLimitError(Exception):
    """Exceção levantada quando o agente atinge o limite configurado de turnos ou custo."""


class EventType(str, Enum):
    """Tipos de eventos emitidos pelo loop assíncrono do agente durante streaming."""

    TOKEN = "token"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    FINAL = "final"


@dataclass(frozen=True)
class AgentEvent:
    """Evento imutável emitido por Agent.run_stream para consumo pela CLI ou UI."""

    type: EventType
    text: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_output: str | None = None
    tool_call_id: str | None = None


class ApprovalAction(str, Enum):
    """Ações possíveis em uma decisão de aprovação humana (HitL)."""

    APPROVE = "approve"
    APPROVE_ALL = "approve_all"
    REJECT = "reject"


@dataclass(frozen=True)
class ApprovalDecision:
    """Resultado estruturado da verificação de aprovação para uma chamada de ferramenta."""

    action: ApprovalAction
    reason: str | None = None

    @property
    def approved(self) -> bool:
        return self.action in (ApprovalAction.APPROVE, ApprovalAction.APPROVE_ALL)


ApprovalResult = ApprovalDecision | ApprovalAction | bool
ApprovalCallback = Callable[[str, dict[str, Any]], ApprovalResult | Awaitable[ApprovalResult]]


def _normalize_approval_decision(raw: ApprovalResult) -> ApprovalDecision:
    """Normaliza retornos de callback de aprovação para ApprovalDecision."""
    if isinstance(raw, ApprovalDecision):
        return raw
    if isinstance(raw, ApprovalAction):
        return ApprovalDecision(action=raw)
    if isinstance(raw, bool):
        return ApprovalDecision(action=ApprovalAction.APPROVE if raw else ApprovalAction.REJECT)
    raise TypeError(f"Retorno inválido do callback de aprovação: {type(raw).__name__}.")


def _format_rejection_message(tool_name: str, reason: str | None = None) -> str:
    """Formata feedback estruturado enviado ao modelo quando o usuário recusa uma ferramenta."""
    cleaned_reason = reason.strip() if reason else ""
    if cleaned_reason:
        return (
            f"Ação recusada pelo usuário: a execução da ferramenta '{tool_name}' não foi aprovada. "
            f"Motivo: {cleaned_reason}. Ajuste seu plano ou pergunte ao usuário como prosseguir."
        )
    return (
        f"Ação recusada pelo usuário: a execução da ferramenta '{tool_name}' não foi aprovada. "
        "Ajuste seu plano ou pergunte ao usuário como prosseguir."
    )


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
    auto_approve: bool = True
    require_approval: set[str] = field(default_factory=lambda: set(APPROVAL_REQUIRED_TOOLS))
    approval_callback: ApprovalCallback | None = None
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

    async def _request_approval(
        self,
        call_name: str,
        effective_args: dict[str, Any],
    ) -> ApprovalDecision:
        """Solicita aprovação humana para ferramentas configuradas em require_approval."""
        if self.auto_approve or call_name not in self.require_approval:
            return ApprovalDecision(ApprovalAction.APPROVE)
        if self.approval_callback is None:
            return ApprovalDecision(
                ApprovalAction.REJECT,
                reason="nenhum callback de aprovação configurado",
            )

        raw = self.approval_callback(call_name, dict(effective_args))
        if inspect.isawaitable(raw):
            raw = await raw

        decision = _normalize_approval_decision(raw)
        if decision.action is ApprovalAction.APPROVE_ALL:
            self.auto_approve = True
        return decision

    async def _execute_tool_call(self, call_name: str, call_args: Any) -> str:
        allowed, blocked_message, effective_args = run_guardrail_checks(
            self.guardrails, call_name, call_args
        )
        if not allowed:
            raw_output: Any = blocked_message or "Blocked by safety rules."
        else:
            fn = self.tools.get(call_name)
            if not fn:
                raw_output = f"ERRO: Ferramenta desconhecida '{call_name}'"
            else:
                approval_args = dict(effective_args) if isinstance(effective_args, dict) else {}
                decision = await self._request_approval(call_name, approval_args)
                if not decision.approved:
                    raw_output = _format_rejection_message(call_name, decision.reason)
                else:
                    try:
                        raw_output = fn(**effective_args)
                        if inspect.isawaitable(raw_output):
                            raw_output = await raw_output
                    except Exception as exc:
                        raw_output = f"ERRO: {type(exc).__name__}: {exc}"

        sanitized = sanitize_with_guardrails(self.guardrails, call_name, str(raw_output))
        return _truncate(sanitized)

    async def run_stream(self, task: str) -> AsyncIterator[AgentEvent]:
        """Executa o loop de raciocínio e ação emitindo eventos tipados em tempo real."""
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

                resp: Any = None
                fallback_text: list[str] = []
                async for chunk in stream_or_complete(self.model, self.messages, self.tool_schemas):
                    if chunk.text_delta:
                        fallback_text.append(chunk.text_delta)
                        yield AgentEvent(type=EventType.TOKEN, text=chunk.text_delta)
                    if chunk.response is not None:
                        resp = chunk.response

                if resp is None:
                    resp = ModelResponse(content="".join(fallback_text))

                self.n_turns += 1
                self.cost += resp.get("cost", 0.0)

                msg_entry = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
                self.messages.append(msg_entry)

                tool_calls = resp.get("tool_calls") or []
                if not tool_calls:
                    if checkpoint is not None and checkpoint.is_active:
                        self._record_checkpoint(checkpoint)
                    yield AgentEvent(type=EventType.FINAL, text=resp.get("content", ""))
                    return

                for call in tool_calls:
                    call_name = call["name"] if isinstance(call, dict) else call.name
                    call_args = call["args"] if isinstance(call, dict) else call.args
                    call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", f"call_{call_name}")
                    event_args = dict(call_args) if isinstance(call_args, dict) else {}

                    yield AgentEvent(
                        type=EventType.TOOL_START,
                        tool_name=call_name,
                        tool_args=event_args,
                        tool_call_id=call_id,
                    )
                    tool_output = await self._execute_tool_call(call_name, call_args)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call_name,
                            "content": tool_output,
                        }
                    )
                    yield AgentEvent(
                        type=EventType.TOOL_END,
                        tool_name=call_name,
                        tool_output=tool_output,
                        tool_call_id=call_id,
                    )
        except BaseException as exc:
            if checkpoint is not None and checkpoint.is_active:
                if _is_limit_stop(exc):
                    self._record_checkpoint(checkpoint)
                else:
                    result = checkpoint.rollback()
                    self.last_rollback_message = result.message
            raise

    async def run(self, task: str) -> str:
        """Executa um turno completo de raciocínio e ação retornando o texto final."""
        final_text = ""
        async with contextlib.aclosing(self.run_stream(task)) as events:
            async for event in events:
                if event.type is EventType.FINAL:
                    final_text = event.text
        return final_text
