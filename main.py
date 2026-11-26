import argparse
import asyncio
import contextlib
from datetime import datetime
import pathlib
import re
import sys
from typing import Any, TextIO

from agent import Agent, AgentEvent, AgentLimitError, ApprovalAction, ApprovalDecision, EventType
from guardrails import GuardrailsConfig, build_guardrails
from providers import get_provider
from session import (
    DEFAULT_SESSIONS_DIR,
    SESSION_ID_RE,
    generate_session_id,
    list_saved_sessions,
    resolve_session_path,
    resolve_session_target,
    restore_session_data as _restore_session_data,
    validate_session_id,
)
from tools import TOOL_SCHEMAS, build_tools


__all__ = [
    "DEFAULT_SESSIONS_DIR",
    "GUARDRAIL_CHOICES",
    "SESSION_ID_RE",
    "TerminalStreamRenderer",
    "build_guardrails_config",
    "format_approval_prompt",
    "generate_session_id",
    "list_saved_sessions",
    "parse_approval_response",
    "parse_args",
    "resolve_session_path",
    "resolve_session_target",
    "terminal_chat",
    "validate_session_id",
]

GUARDRAIL_CHOICES = (
    "path_sandboxing",
    "command_safety",
    "secrets_redaction",
    "loop_detection",
    "all",
)

EXIT_COMMANDS = frozenset({"exit", "sair", "quit", "q"})

_APPROVAL_RESPONSE_MAP: dict[str, ApprovalAction] = {
    "": ApprovalAction.REJECT,
    "n": ApprovalAction.REJECT,
    "nao": ApprovalAction.REJECT,
    "não": ApprovalAction.REJECT,
    "no": ApprovalAction.REJECT,
    "s": ApprovalAction.APPROVE,
    "sim": ApprovalAction.APPROVE,
    "y": ApprovalAction.APPROVE,
    "yes": ApprovalAction.APPROVE,
    "a": ApprovalAction.APPROVE_ALL,
    "all": ApprovalAction.APPROVE_ALL,
    "sempre": ApprovalAction.APPROVE_ALL,
    "todos": ApprovalAction.APPROVE_ALL,
}

_AUTO_APPROVE_VALUES: dict[str, bool] = {
    "on": True,
    "true": True,
    "s": True,
    "sim": True,
    "1": True,
    "off": False,
    "false": False,
    "n": False,
    "nao": False,
    "não": False,
    "0": False,
}


def _format_preview(value: Any, max_preview: int = 120) -> str:
    cleaned = str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    if len(cleaned) <= max_preview:
        return cleaned
    if max_preview <= 3:
        return cleaned[:max_preview] + "..."
    return cleaned[: max_preview - 3] + "..."


def _format_tool_event_args(tool_name: str, args: dict[str, Any], max_preview: int = 80) -> str:
    if tool_name == "bash":
        return _format_preview(args.get("cmd", ""), max_preview)
    if tool_name in ("write_file", "search_replace", "read_file"):
        return _format_preview(args.get("path", ""), max_preview)
    pairs = ", ".join(f"{k}={_format_preview(v, max_preview)}" for k, v in args.items())
    return _format_preview(pairs, max_preview)


class TerminalStreamRenderer:
    """Renderiza eventos de Agent.run_stream em tempo real no terminal."""

    def __init__(self, stream: TextIO | None = None, max_preview: int = 80) -> None:
        self._stream: TextIO = stream if stream is not None else sys.stdout
        self._max_preview = max_preview
        self._in_text_block = False
        self._has_rendered_prefix = False
        self.final_text = ""

    def _write(self, text: str) -> None:
        self._stream.write(text)
        self._stream.flush()

    def handle(self, event: AgentEvent) -> None:
        if event.type is EventType.TOKEN:
            if not event.text:
                return
            if not self._in_text_block:
                prefix = "\nAgente: " if not self._has_rendered_prefix else "\n"
                self._write(prefix)
                self._has_rendered_prefix = True
                self._in_text_block = True
            self._write(event.text)
            return

        if event.type is EventType.TOOL_START:
            if self._in_text_block:
                self._write("\n")
                self._in_text_block = False
            tool_name = event.tool_name or "ferramenta"
            summary = _format_tool_event_args(
                tool_name,
                event.tool_args or {},
                max_preview=self._max_preview,
            )
            detail = f" ({summary})" if summary else ""
            self._write(f"\n[Executando ferramenta]: {tool_name}{detail}\n")
            return

        if event.type is EventType.TOOL_END:
            tool_name = event.tool_name or "ferramenta"
            raw_lines = (event.tool_output or "").splitlines()
            first_line = raw_lines[0] if raw_lines else ""
            preview = _format_preview(first_line, self._max_preview) if first_line else "ok"
            self._write(f"[Ferramenta concluída]: {tool_name} -> {preview}\n")
            return

        if event.type is EventType.FINAL:
            self.final_text = event.text
            if self._in_text_block:
                self._write("\n")
                self._in_text_block = False
            elif not self._has_rendered_prefix and event.text:
                self._write(f"\nAgente: {event.text}\n")
                self._has_rendered_prefix = True

    def finish_on_interrupt(self) -> None:
        """Fecha linha aberta se o streaming for interrompido antes do evento FINAL."""
        if self._in_text_block:
            self._write("\n")
            self._in_text_block = False


def format_approval_prompt(
    tool_name: str,
    args: dict[str, Any],
    max_preview: int = 120,
) -> str:
    """Formata o resumo da chamada de ferramenta para confirmação interativa no terminal."""
    if tool_name == "bash":
        preview = _format_preview(args.get("cmd", ""), max_preview)
        return f"Executar [bash: '{preview}']? (s/n/a): "
    if tool_name == "write_file":
        path_preview = _format_preview(args.get("path", ""), max_preview)
        raw_content = args.get("content", "")
        content_len = len(raw_content) if isinstance(raw_content, str) else len(str(raw_content))
        return f"Executar [write_file: '{path_preview}' ({content_len} chars)]? (s/n/a): "
    if tool_name == "search_replace":
        path_preview = _format_preview(args.get("path", ""), max_preview)
        return f"Executar [search_replace: '{path_preview}']? (s/n/a): "

    pairs = ", ".join(f"{k}={_format_preview(v, max_preview)}" for k, v in args.items())
    preview = _format_preview(pairs, max_preview)
    return f"Executar [{tool_name}: {preview}]? (s/n/a): "


def parse_approval_response(raw: str) -> ApprovalDecision | None:
    """Interpreta a resposta do usuário ('s'/'n'/'a') em ApprovalDecision ou None se inválida."""
    action = _APPROVAL_RESPONSE_MAP.get(raw.strip().lower())
    if action is None:
        return None
    return ApprovalDecision(action=action)


async def _prompt_tool_approval(tool_name: str, args: dict[str, Any]) -> ApprovalDecision:
    """Solicita confirmação humana no terminal sem bloquear o event loop."""
    prompt = format_approval_prompt(tool_name, args)
    while True:
        try:
            raw = await asyncio.to_thread(input, f"\n[Aprovação] {prompt}")
        except EOFError:
            print("\n[Aprovação]: entrada encerrada (EOF); ação recusada.")
            return ApprovalDecision(ApprovalAction.REJECT)

        decision = parse_approval_response(raw)
        if decision is not None:
            return decision
        print("Resposta inválida. Digite 's' (sim), 'n' (não) ou 'a' (aprovar todas na sessão).")


def _format_provider_error(exc: Exception) -> str:
    """Formata erros de API/provedor em uma mensagem concisa para o terminal."""
    text = str(exc)
    retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?s)", text, flags=re.IGNORECASE)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        hint = (
            f" Aguarde ~{retry_match.group(1)} e tente novamente."
            if retry_match
            else " Aguarde alguns segundos e tente novamente."
        )
        return f"[Limite de requisições da API (429)]: cota temporariamente excedida.{hint}"
    first_line = text.splitlines()[0].strip() if text else type(exc).__name__
    return f"[Erro ao consultar o provedor]: {type(exc).__name__}: {first_line}"


def build_guardrails_config(disabled_guardrails: list[str] | None = None) -> GuardrailsConfig:
    config = GuardrailsConfig.from_env()
    disabled = set(disabled_guardrails or [])
    if "all" in disabled:
        config.path_sandboxing = False
        config.command_safety = False
        config.secrets_redaction = False
        config.loop_detection = False
    else:
        for name in disabled:
            if hasattr(config, name):
                setattr(config, name, False)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat interativo no terminal com SWE-Agent.")
    parser.add_argument(
        "-s",
        "--session",
        dest="session_id",
        default=None,
        help="ID da sessão existente em .sessions/ para retomar a conversa.",
    )
    parser.add_argument(
        "--sessions-dir",
        default=DEFAULT_SESSIONS_DIR,
        help=f"Diretório de armazenamento das sessões (padrão: {DEFAULT_SESSIONS_DIR}).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provedor de modelo (ex.: gemini, grok, openai, deepseek, ollama).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Nome do modelo a ser utilizado pelo provedor.",
    )
    parser.add_argument(
        "--disable-guardrail",
        action="append",
        choices=GUARDRAIL_CHOICES,
        default=[],
        help="Desabilita um guardrail específico (pode ser repetido; use 'all' para desabilitar todos).",
    )
    parser.add_argument(
        "--no-checkpoint",
        dest="checkpointing",
        action="store_false",
        default=True,
        help="Desabilita checkpoint automático de workspace e rollback em caso de falha.",
    )
    parser.add_argument(
        "-y",
        "--auto-approve",
        dest="auto_approve",
        action="store_true",
        default=False,
        help="Aprova automaticamente chamadas de ferramentas sem pedir confirmação interativa.",
    )
    parser.add_argument(
        "--no-stream",
        dest="streaming",
        action="store_false",
        default=True,
        help="Desabilita streaming em tempo real e exibe a resposta apenas ao concluir o turno.",
    )
    return parser.parse_args(argv)


def _print_banner(resumed: bool, session_data: dict[str, Any], agent: Agent) -> None:
    print("=" * 60)
    print("🤖 Chat Interativo com SWE-Agent")
    print("Comandos: digite 'exit' ou 'sair' para encerrar a conversa.")
    print("          digite 'memory' para visualizar a memória da sessão.")
    print("          digite '/load <session_id|last>' para carregar uma sessão salva.")
    print("          digite '/sessions' para listar as sessões salvas.")
    print("          digite '/undo' para desfazer as alterações do último turno.")
    print("          digite '/auto-approve [on|off]' para alternar aprovação automática.")
    if agent.auto_approve:
        print("Modo auto-approve ativo: ferramentas serão aprovadas automaticamente.")
    if resumed:
        interactions = len(session_data["history"]) // 2
        print(
            f"Sessão retomada: {session_data['session_id']} "
            f"({interactions} interações, {agent.n_turns} turnos internos)"
        )
    print("=" * 60 + "\n")


def _print_saved_sessions(sessions_path: pathlib.Path) -> None:
    saved = list_saved_sessions(sessions_path)
    print("\n[Sessões Salvas]:")
    if not saved:
        print(f"  Nenhuma sessão encontrada em {sessions_path}.")
    else:
        for entry in saved:
            if entry["valid"]:
                print(
                    f"  - {entry['session_id']} | "
                    f"interações: {entry['interactions']} | "
                    f"turnos: {entry['n_turns']} | "
                    f"atualizado em: {entry['updated_at']}"
                )
            else:
                print(f"  - {entry['session_id']} [arquivo inválido/corrompido]")
    print("-" * 60 + "\n")


def _print_session_memory(session_data: dict[str, Any], agent: Agent) -> None:
    print("\n[Memória da Sessão Atual]:")
    print(f"Sessão: {session_data['session_id']}")
    print(f"Total de turnos internos: {agent.n_turns}")
    print(f"Custo estimado: ${agent.cost:.4f}")
    print(f"Total de mensagens gravadas: {len(session_data['history'])}")
    for item in session_data["history"]:
        print(f"  - [{item['timestamp']}] {item['speaker'].upper()}: {item['text'][:80]}...")
    print("-" * 60 + "\n")


def _handle_undo_command(agent: Agent) -> None:
    result = agent.undo_last_turn()
    if result.ok:
        print(f"\n[Rollback]: {result.message}")
    else:
        print(f"\n[Rollback indisponível]: {result.message}")
    print("-" * 60 + "\n")


def _handle_auto_approve_command(raw_arg: str, agent: Agent) -> None:
    arg = raw_arg.strip().lower()
    if not arg:
        agent.auto_approve = not agent.auto_approve
    elif arg in _AUTO_APPROVE_VALUES:
        agent.auto_approve = _AUTO_APPROVE_VALUES[arg]
    else:
        print("\n[Uso]: /auto-approve [on|off]\n")
        return

    status = "ATIVADO (aprovação automática)" if agent.auto_approve else "DESATIVADO (confirmação manual)"
    print(f"\n[Auto-approve]: {status}")
    print("-" * 60 + "\n")


def _handle_load_command(
    raw_target: str,
    sessions_path: pathlib.Path,
    agent: Agent,
) -> tuple[dict[str, Any], pathlib.Path] | None:
    if not raw_target:
        print("\n[Uso]: /load <session_id|last> (digite '/sessions' para listar)\n")
        return None
    try:
        validated_id, target_file = resolve_session_target(sessions_path, raw_target)
        loaded = agent.load_from_file(target_file)
        restored_data = _restore_session_data(loaded, fallback_id=validated_id)
        interactions = len(restored_data["history"]) // 2
        print(
            f"\n[Sessão carregada]: {restored_data['session_id']} "
            f"(interações: {interactions}, turnos internos: {agent.n_turns}, "
            f"custo: ${agent.cost:.4f})"
        )
        print("-" * 60 + "\n")
        return restored_data, target_file
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[Erro ao carregar sessão]: {exc}\n")
        return None


async def terminal_chat(
    session_id: str | None = None,
    sessions_dir: str | pathlib.Path = DEFAULT_SESSIONS_DIR,
    disabled_guardrails: list[str] | None = None,
    checkpointing: bool = True,
    provider_name: str | None = None,
    model_name: str | None = None,
    auto_approve: bool = False,
    streaming: bool = True,
) -> dict[str, Any]:
    provider_kwargs: dict[str, Any] = {"model": model_name} if model_name else {}
    provider = get_provider(provider_name, **provider_kwargs)
    guardrails_config = build_guardrails_config(disabled_guardrails)
    agent = Agent(
        name="SWE-Agent",
        model=provider,
        instructions=(
            "Você é um assistente de engenharia de software conciso e prestativo. "
            "Mantenha o contexto das mensagens anteriores."
        ),
        guardrails=build_guardrails(guardrails_config),
        tools=build_tools(),
        tool_schemas=TOOL_SCHEMAS,
        checkpointing=checkpointing,
        auto_approve=auto_approve,
        approval_callback=_prompt_tool_approval,
    )

    sessions_path = pathlib.Path(sessions_dir)
    resumed = False

    if session_id is not None:
        try:
            validated_id, current_file = resolve_session_target(sessions_path, session_id)
            loaded = agent.load_from_file(current_file)
            session_data = _restore_session_data(loaded, fallback_id=validated_id)
            resumed = True
        except (FileNotFoundError, ValueError) as exc:
            print(f"Erro ao carregar sessão '{session_id}': {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    else:
        new_id = generate_session_id(sessions_path)
        now_iso = datetime.now().isoformat()
        _, current_file = resolve_session_path(sessions_path, new_id)
        session_data = {
            "session_id": new_id,
            "started_at": now_iso,
            "updated_at": now_iso,
            "history": [],
        }

    persisted_file: pathlib.Path | None = current_file if resumed else None

    def persist_session() -> pathlib.Path:
        nonlocal persisted_file
        session_data["updated_at"] = datetime.now().isoformat()
        persisted_file = agent.save_to_file(
            current_file,
            metadata={
                "session_id": session_data["session_id"],
                "started_at": session_data["started_at"],
                "updated_at": session_data["updated_at"],
                "history": session_data["history"],
            },
        )
        return persisted_file

    _print_banner(resumed, session_data, agent)

    while True:
        try:
            user_input = await asyncio.to_thread(input, "Você: ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando sessão via interrupção...")
            break

        if not user_input:
            continue

        lower_cmd = user_input.lower()
        if lower_cmd in EXIT_COMMANDS:
            print("\nEncerrando conversa...")
            break
        if lower_cmd == "/sessions":
            _print_saved_sessions(sessions_path)
            continue
        if lower_cmd == "/load" or lower_cmd.startswith("/load "):
            loaded_result = _handle_load_command(user_input[5:].strip(), sessions_path, agent)
            if loaded_result is not None:
                session_data, current_file = loaded_result
                persisted_file = current_file
            continue
        if lower_cmd in ("/undo", "/rollback"):
            _handle_undo_command(agent)
            continue
        if lower_cmd == "/auto-approve" or lower_cmd.startswith("/auto-approve "):
            _handle_auto_approve_command(user_input[len("/auto-approve") :].strip(), agent)
            continue
        if lower_cmd == "memory":
            _print_session_memory(session_data, agent)
            continue

        prev_msg_count = len(agent.messages)
        prev_turns = agent.n_turns
        session_data["history"].append(
            {
                "speaker": "user",
                "text": user_input,
                "timestamp": datetime.now().isoformat(),
            }
        )

        print("\n[Agente pensando...]")
        renderer = TerminalStreamRenderer() if streaming else None
        try:
            if renderer is not None:
                async with contextlib.aclosing(agent.run_stream(user_input)) as events:
                    async for event in events:
                        renderer.handle(event)
                response = renderer.final_text
            else:
                response = await agent.run(user_input)
        except (AgentLimitError, StopIteration) as exc:
            if renderer is not None:
                renderer.finish_on_interrupt()
            persist_session()
            print(f"\n[Limite atingido]: {exc}\n")
            continue
        except KeyboardInterrupt:
            if renderer is not None:
                renderer.finish_on_interrupt()
            if agent.n_turns == prev_turns:
                del agent.messages[prev_msg_count:]
                session_data["history"].pop()
            else:
                persist_session()
            print("\n[Interrompido pelo usuário]")
            if agent.last_rollback_message:
                print(f"\n[Rollback automático]: {agent.last_rollback_message}")
            print("-" * 60)
            continue
        except Exception as exc:
            if renderer is not None:
                renderer.finish_on_interrupt()
            if agent.n_turns == prev_turns:
                del agent.messages[prev_msg_count:]
                session_data["history"].pop()
            else:
                persist_session()
            print(f"\n{_format_provider_error(exc)}")
            if agent.last_rollback_message:
                print(f"\n[Rollback automático]: {agent.last_rollback_message}")
            print("-" * 60)
            continue

        session_data["history"].append(
            {
                "speaker": "agent",
                "text": response,
                "timestamp": datetime.now().isoformat(),
            }
        )
        persist_session()

        if renderer is None:
            print(f"\nAgente: {response}\n")
        else:
            print()
        print("-" * 60)

    session_data["ended_at"] = datetime.now().isoformat()
    session_data["total_interactions"] = len(session_data["history"]) // 2
    if session_data["history"] or persisted_file is not None:
        persist_session()
    session_data["raw_agent_memory"] = agent.get_session_memory()

    print("\n" + "=" * 60)
    print(f"Sessão finalizada: {session_data['session_id']}")
    print(f"Interações guardadas na memória: {session_data['total_interactions']}")
    if persisted_file is not None:
        print(f"Sessão salva em: {persisted_file}")
    print("=" * 60)

    return session_data


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(
        terminal_chat(
            session_id=cli_args.session_id,
            sessions_dir=cli_args.sessions_dir,
            disabled_guardrails=cli_args.disable_guardrail,
            checkpointing=cli_args.checkpointing,
            provider_name=cli_args.provider,
            model_name=cli_args.model,
            auto_approve=cli_args.auto_approve,
            streaming=cli_args.streaming,
        )
    )
