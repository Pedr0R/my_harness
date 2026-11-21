import argparse
import asyncio
from datetime import datetime
import pathlib
import re
import sys
from typing import Any

from agent import Agent, AgentLimitError
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
from tools import TOOLS, TOOL_SCHEMAS


__all__ = [
    "DEFAULT_SESSIONS_DIR",
    "GUARDRAIL_CHOICES",
    "SESSION_ID_RE",
    "build_guardrails_config",
    "generate_session_id",
    "list_saved_sessions",
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
    return parser.parse_args(argv)


def _print_banner(resumed: bool, session_data: dict[str, Any], agent: Agent) -> None:
    print("=" * 60)
    print("🤖 Chat Interativo com SWE-Agent")
    print("Comandos: digite 'exit' ou 'sair' para encerrar a conversa.")
    print("          digite 'memory' para visualizar a memória da sessão.")
    print("          digite '/load <session_id|last>' para carregar uma sessão salva.")
    print("          digite '/sessions' para listar as sessões salvas.")
    print("          digite '/undo' para desfazer as alterações do último turno.")
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
        tools=TOOLS,
        tool_schemas=TOOL_SCHEMAS,
        checkpointing=checkpointing,
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
        try:
            response = await agent.run(user_input)
        except (AgentLimitError, StopIteration) as exc:
            persist_session()
            print(f"\n[Limite atingido]: {exc}\n")
            continue
        except KeyboardInterrupt:
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

        print(f"\nAgente: {response}\n")
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
        )
    )
