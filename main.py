import argparse
import asyncio
from datetime import datetime
import json
import pathlib
import re
import sys
from typing import Any

from agent import Agent
from guardrails import GuardrailsConfig, build_guardrails
from providers import get_provider


DEFAULT_SESSIONS_DIR = ".sessions"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
GUARDRAIL_CHOICES = (
    "path_sandboxing",
    "command_safety",
    "secrets_redaction",
    "loop_detection",
    "all",
)


def validate_session_id(raw_id: str) -> str:
    """Valida e normaliza um identificador de sessão, bloqueando path traversal."""
    cleaned = raw_id.strip()
    if cleaned.lower().endswith(".json"):
        cleaned = cleaned[:-5]
    if not cleaned or ".." in cleaned or not SESSION_ID_RE.match(cleaned):
        raise ValueError(
            f"Identificador de sessão inválido: {raw_id!r}. "
            "Use apenas letras, números, '.', '_' ou '-'."
        )
    return cleaned


def resolve_session_path(sessions_dir: pathlib.Path, raw_id: str) -> tuple[str, pathlib.Path]:
    """Resolve o caminho seguro <sessions_dir>/<session_id>.json."""
    session_id = validate_session_id(raw_id)
    base_dir = sessions_dir.resolve()
    target = (base_dir / f"{session_id}.json").resolve()
    if target.parent != base_dir:
        raise ValueError(f"Caminho de sessão fora do diretório permitido: {raw_id!r}")
    return session_id, target


def resolve_session_target(sessions_dir: pathlib.Path, raw_id: str) -> tuple[str, pathlib.Path]:
    """Resolve um ID de sessão ou o atalho 'last' para a sessão válida mais recente."""
    if raw_id.strip().lower() == "last":
        valid_sessions = [entry for entry in list_saved_sessions(sessions_dir) if entry["valid"]]
        if not valid_sessions:
            raise FileNotFoundError(f"Nenhuma sessão salva encontrada em {sessions_dir}.")
        return resolve_session_path(sessions_dir, valid_sessions[0]["session_id"])
    return resolve_session_path(sessions_dir, raw_id)


def _format_provider_error(exc: Exception) -> str:
    """Formata erros de API/provedor em uma mensagem concisa para o terminal."""
    text = str(exc)
    retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?s)", text, flags=re.IGNORECASE)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        hint = f" Aguarde ~{retry_match.group(1)} e tente novamente." if retry_match else " Aguarde alguns segundos e tente novamente."
        return f"[Limite de requisições da API (429)]: cota temporariamente excedida.{hint}"
    first_line = text.splitlines()[0].strip() if text else type(exc).__name__
    return f"[Erro ao consultar o provedor]: {type(exc).__name__}: {first_line}"


def generate_session_id(sessions_dir: pathlib.Path) -> str:
    """Gera um ID único no formato session_YYYYMMDD_HHMMSS sem sobrescrever sessões existentes."""
    base = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    candidate = base
    suffix = 1
    while (sessions_dir / f"{candidate}.json").exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def list_saved_sessions(sessions_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Lista arquivos de sessão salvos ordenados do mais recente para o mais antigo."""
    if not sessions_dir.is_dir():
        return []

    entries: list[tuple[float, dict[str, Any]]] = []
    for path in sessions_dir.glob("*.json"):
        mtime = path.stat().st_mtime
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON raiz não é um objeto")
            history = data.get("history", [])
            interactions = len(history) // 2 if isinstance(history, list) else 0
            entries.append(
                (
                    mtime,
                    {
                        "session_id": str(data.get("session_id") or path.stem),
                        "updated_at": str(data.get("updated_at") or data.get("started_at") or ""),
                        "n_turns": data.get("n_turns", 0),
                        "interactions": interactions,
                        "valid": True,
                    },
                )
            )
        except Exception:
            entries.append(
                (
                    mtime,
                    {
                        "session_id": path.stem,
                        "updated_at": "",
                        "n_turns": 0,
                        "interactions": 0,
                        "valid": False,
                    },
                )
            )

    entries.sort(key=lambda item: (item[1]["updated_at"], item[0]), reverse=True)
    return [item[1] for item in entries]


def _restore_session_data(data: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    """Reconstrói session_data a partir do JSON carregado de disco."""
    now_iso = datetime.now().isoformat()
    raw_history = data.get("history", [])
    history = list(raw_history) if isinstance(raw_history, list) else []
    return {
        "session_id": str(data.get("session_id") or fallback_id),
        "started_at": str(data.get("started_at") or now_iso),
        "updated_at": str(data.get("updated_at") or now_iso),
        "history": history,
    }


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
        "--disable-guardrail",
        action="append",
        choices=GUARDRAIL_CHOICES,
        default=[],
        help="Desabilita um guardrail específico (pode ser repetido; use 'all' para desabilitar todos).",
    )
    return parser.parse_args(argv)


async def terminal_chat(
    session_id: str | None = None,
    sessions_dir: str | pathlib.Path = DEFAULT_SESSIONS_DIR,
    disabled_guardrails: list[str] | None = None,
) -> dict[str, Any]:
    provider = get_provider("gemini", model="gemini-3.5-flash-lite")
    guardrails_config = build_guardrails_config(disabled_guardrails)
    agent = Agent(
        name="SWE-Agent",
        model=provider,
        instructions="Você é um assistente de engenharia de software conciso e prestativo. Mantenha o contexto das mensagens anteriores.",
        guardrails=build_guardrails(guardrails_config),
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
            "history": [],  # Array com o log de cada interação usuário/agente
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

    print("=" * 60)
    print("🤖 Chat Interativo com SWE-Agent")
    print("Comandos: digite 'exit' ou 'sair' para encerrar a conversa.")
    print("          digite 'memory' para visualizar a memória da sessão.")
    print("          digite '/load <session_id|last>' para carregar uma sessão salva.")
    print("          digite '/sessions' para listar as sessões salvas.")
    if resumed:
        interactions = len(session_data["history"]) // 2
        print(
            f"Sessão retomada: {session_data['session_id']} "
            f"({interactions} interações, {agent.n_turns} turnos internos)"
        )
    print("=" * 60 + "\n")

    while True:
        try:
            # Executa a captura do input do usuário sem travar o loop de eventos
            user_input = await asyncio.to_thread(input, "Você: ")
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando sessão via interrupção...")
            break

        # Ignora mensagens vazias
        if not user_input:
            continue

        lower_cmd = user_input.lower()

        # Comando simples de saída
        if lower_cmd in ("exit", "sair", "quit", "q"):
            print("\nEncerrando conversa...")
            break

        # Comando para listar sessões salvas
        if lower_cmd == "/sessions":
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
            continue

        # Comando para carregar outra sessão em tempo de execução
        if lower_cmd == "/load" or lower_cmd.startswith("/load "):
            target_raw = user_input[5:].strip()
            if not target_raw:
                print("\n[Uso]: /load <session_id|last> (digite '/sessions' para listar)\n")
                continue
            try:
                validated_id, target_file = resolve_session_target(sessions_path, target_raw)
                loaded = agent.load_from_file(target_file)
                session_data = _restore_session_data(loaded, fallback_id=validated_id)
                current_file = target_file
                persisted_file = target_file
                interactions = len(session_data["history"]) // 2
                print(
                    f"\n[Sessão carregada]: {session_data['session_id']} "
                    f"(interações: {interactions}, turnos internos: {agent.n_turns}, "
                    f"custo: ${agent.cost:.4f})"
                )
                print("-" * 60 + "\n")
            except (FileNotFoundError, ValueError) as exc:
                print(f"\n[Erro ao carregar sessão]: {exc}\n")
            continue

        # Comando extra para visualizar a memória atual da sessão
        if lower_cmd == "memory":
            print("\n[Memória da Sessão Atual]:")
            print(f"Sessão: {session_data['session_id']}")
            print(f"Total de turnos internos: {agent.n_turns}")
            print(f"Custo estimado: ${agent.cost:.4f}")
            print(f"Total de mensagens gravadas: {len(session_data['history'])}")
            for item in session_data["history"]:
                print(f"  - [{item['timestamp']}] {item['speaker'].upper()}: {item['text'][:80]}...")
            print("-" * 60 + "\n")
            continue

        # Registra a entrada do usuário no array da sessão (com rollback caso o turno falhe antes de responder)
        prev_msg_count = len(agent.messages)
        prev_turns = agent.n_turns
        user_turn = {
            "speaker": "user",
            "text": user_input,
            "timestamp": datetime.now().isoformat(),
        }
        session_data["history"].append(user_turn)

        print("\n[Agente pensando...]")
        try:
            response = await agent.run(user_input)
        except StopIteration as exc:
            persist_session()
            print(f"\n[Limite atingido]: {exc}\n")
            continue
        except Exception as exc:
            if agent.n_turns == prev_turns:
                del agent.messages[prev_msg_count:]
                session_data["history"].pop()
            else:
                persist_session()
            print(f"\n{_format_provider_error(exc)}\n")
            print("-" * 60)
            continue

        # Registra a resposta do agente no array da sessão
        agent_turn = {
            "speaker": "agent",
            "text": response,
            "timestamp": datetime.now().isoformat(),
        }
        session_data["history"].append(agent_turn)

        # Salvamento contínuo após cada turno concluído
        persist_session()

        print(f"\nAgente: {response}\n")
        print("-" * 60)

    # Ao sair, atualiza o resumo final no dicionário da sessão e persiste se houver histórico
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
        )
    )
