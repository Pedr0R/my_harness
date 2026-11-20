from __future__ import annotations

from datetime import datetime
import json
import os
import pathlib
import re
from typing import Any


DEFAULT_SESSIONS_DIR = ".sessions"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def serialize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Converte uma mensagem do histórico do agente para uma estrutura segura para JSON."""
    serialized: dict[str, Any] = {}
    for key, value in msg.items():
        if key == "raw_content":
            if value is None:
                continue
            if hasattr(value, "model_dump"):
                try:
                    serialized[key] = value.model_dump(mode="json", exclude_none=True)
                except Exception:
                    continue
            elif isinstance(value, (dict, list, str, int, float, bool)):
                serialized[key] = value
            continue
        serialized[key] = value
    return serialized


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


def restore_session_data(data: dict[str, Any], fallback_id: str) -> dict[str, Any]:
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


def save_session_file(
    path: str | pathlib.Path,
    n_turns: int,
    cost: float,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Salva atomicamente o estado da sessão e das mensagens do agente em disco (JSON)."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now().isoformat()
    meta = dict(metadata) if metadata else {}

    payload: dict[str, Any] = {
        "session_id": meta.pop("session_id", target.stem),
        "started_at": meta.pop("started_at", now_iso),
        "updated_at": meta.pop("updated_at", now_iso),
        "n_turns": n_turns,
        "cost": cost,
        "messages": [serialize_message(m) for m in messages],
        "history": meta.pop("history", []),
    }
    payload.update(meta)

    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return target


def load_session_file(path: str | pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]], int, float]:
    """Carrega e valida um arquivo JSON de sessão, retornando (data, messages, n_turns, cost)."""
    source = pathlib.Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Arquivo de sessão não encontrado: {source}")

    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Arquivo de sessão corrompido ou JSON inválido ({source}): {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Formato de sessão inválido em {source}: esperado objeto JSON.")

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list) or not all(isinstance(m, dict) for m in raw_messages):
        raise ValueError(f"Campo 'messages' inválido no arquivo de sessão: {source}")

    n_turns = data.get("n_turns", 0)
    cost = data.get("cost", 0.0)
    if not isinstance(n_turns, int) or isinstance(n_turns, bool) or n_turns < 0:
        raise ValueError(f"Campo 'n_turns' inválido no arquivo de sessão: {source}")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
        raise ValueError(f"Campo 'cost' inválido no arquivo de sessão: {source}")

    return data, [dict(m) for m in raw_messages], n_turns, float(cost)
