from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import fnmatch
import json
import os
import pathlib
import re
import stat
import subprocess
from typing import Any

from guardrails import BaseGuardrail
from guardrails.path_sandbox import is_sensitive_path
from providers import ModelProvider, get_provider
from workspace import CheckpointResult, WorkspaceCheckpoint


_MAX_CHECKPOINTS = 20


def _is_limit_stop(exc: BaseException) -> bool:
    """Identifica paradas esperadas por limite de turnos/custo (incluindo PEP 479 em corrotinas)."""
    return isinstance(exc, StopIteration) or isinstance(exc.__cause__, StopIteration)


# ---- Ferramentas: bash + ferramentas de arquivos e exploração ------------------
def _truncate(s: str, n: int = 25_000) -> str:
    """Trunca strings longas para não sobrecarregar o contexto."""
    return s if len(s) <= n else s[:n] + "\n[truncado]"


def tool_bash(cmd: str) -> str:
    """Executa um comando bash/shell no sistema operacional."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return _truncate(f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")


def tool_read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Lê linhas de um arquivo com numeração e suporte a paginação."""
    lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return _truncate(
        "\n".join(f"{i + 1:4}: {ln}" for i, ln in enumerate(lines[offset : offset + limit]))
    )


def tool_write_file(path: str, content: str) -> str:
    """Escreve ou sobrescreve conteúdo em um arquivo."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"escreveu {len(content)} bytes"


def tool_search_replace(path: str, search: str, replace: str) -> str:
    """Substitui uma ocorrência única de texto em um arquivo."""
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if text.count(search) != 1:
        return f"ERRO: o termo de busca ocorre {text.count(search)}x; deve ser único no arquivo"
    p.write_text(text.replace(search, replace, 1), encoding="utf-8")
    return "OK"


IGNORED_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache", ".sessions"}
)


def _is_link_or_junction(path: str) -> bool:
    if os.name == "nt":
        try:
            return bool(os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except OSError:
            return False
    return os.path.islink(path)


def _resolve_safe_walk_entry(
    entry_str: str,
    entry_path: pathlib.Path,
    search_root: pathlib.Path,
) -> pathlib.Path | None:
    resolved = entry_path
    if _is_link_or_junction(entry_str):
        try:
            resolved = entry_path.resolve()
        except OSError:
            return None
    if is_sensitive_path(resolved, search_root) is not None:
        return None
    return resolved


def _filter_walk_dirs(
    dirpath: str,
    current_dir: pathlib.Path,
    dirnames: list[str],
    search_root: pathlib.Path,
) -> list[str]:
    kept: list[str] = []
    for d in sorted(dirnames):
        if d in IGNORED_DIRS:
            continue
        dir_str = os.path.join(dirpath, d)
        resolved = _resolve_safe_walk_entry(dir_str, current_dir / d, search_root)
        if resolved is None or resolved.name in IGNORED_DIRS:
            continue
        kept.append(d)
    return kept


def _grep_file(
    file_path: pathlib.Path | str,
    display_path: str,
    pattern: str,
    compiled: re.Pattern[str] | None,
    matches: list[str],
    max_results: int,
) -> bool:
    try:
        with open(file_path, "rb") as fh:
            head = fh.read(8192)
            if b"\x00" in head:
                return False
            raw = head if len(head) < 8192 else head + fh.read()
    except OSError:
        return False

    text = raw.decode("utf-8", errors="replace")
    if compiled is None and pattern not in text:
        return False

    for line_no, line in enumerate(text.splitlines(), start=1):
        matched = compiled.search(line) is not None if compiled is not None else pattern in line
        if matched:
            if len(matches) >= max_results:
                return True
            matches.append(f"{display_path}:{line_no}: {line}")
    return False


def tool_grep(
    pattern: str,
    path: str = ".",
    regex: bool = False,
    file_pattern: str = "*",
    max_results: int = 50,
) -> str:
    """Busca texto literal ou regex em um arquivo ou recursivamente em um diretório."""
    if not pattern:
        return "ERRO: padrão de busca não pode ser vazio"
    if max_results <= 0:
        return "ERRO: max_results deve ser maior que zero"

    compiled: re.Pattern[str] | None = None
    if regex:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"ERRO: expressão regular inválida: {exc}"

    target = pathlib.Path(path)
    if not target.exists():
        return f"ERRO: caminho não encontrado: {path}"

    matches: list[str] = []
    truncated = False

    if target.is_file():
        truncated = _grep_file(target, target.as_posix(), pattern, compiled, matches, max_results)
    elif target.is_dir():
        search_root = target.resolve()
        for dirpath, dirnames, filenames in os.walk(search_root):
            current_dir = pathlib.Path(dirpath)
            rel_dir_parts = current_dir.relative_to(search_root).parts
            rel_prefix = "/".join(rel_dir_parts) + "/" if rel_dir_parts else ""

            dirnames[:] = _filter_walk_dirs(dirpath, current_dir, dirnames, search_root)

            for filename in sorted(filenames):
                if file_pattern != "*" and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                file_str = os.path.join(dirpath, filename)
                resolved_file = _resolve_safe_walk_entry(file_str, current_dir / filename, search_root)
                if resolved_file is None:
                    continue
                rel_path = f"{rel_prefix}{filename}"
                if _grep_file(resolved_file, rel_path, pattern, compiled, matches, max_results):
                    truncated = True
                    break
            if truncated:
                break

    if not matches:
        return "Nenhuma ocorrência encontrada."

    output = "\n".join(matches)
    if truncated:
        output += f"\n[limite de {max_results} resultados atingido]"
    return _truncate(output)


def tool_list_dir(path: str = ".", max_depth: int = 1) -> str:
    """Lista arquivos e subdiretórios de forma hierárquica até max_depth níveis."""
    if max_depth <= 0:
        return "ERRO: max_depth deve ser maior que zero"

    target = pathlib.Path(path)
    if not target.exists():
        return f"ERRO: caminho não encontrado: {path}"
    if not target.is_dir():
        return f"ERRO: o caminho não é um diretório: {path}"

    search_root = target.resolve()
    lines: list[str] = []

    def _walk_level(current_dir: pathlib.Path, depth: int) -> None:
        try:
            with os.scandir(current_dir) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            return

        indent = "  " * depth
        for entry in entries:
            if entry.name in IGNORED_DIRS:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
            resolved = _resolve_safe_walk_entry(entry.path, current_dir / entry.name, search_root)
            if resolved is None:
                continue
            try:
                is_dir = resolved.is_dir()
            except OSError:
                continue
            if is_dir and (entry.name in IGNORED_DIRS or resolved.name in IGNORED_DIRS):
                continue
            lines.append(f"{indent}{entry.name}/" if is_dir else f"{indent}{entry.name}")
            if is_dir and depth + 1 < max_depth:
                _walk_level(resolved, depth + 1)

    _walk_level(search_root, 0)

    if not lines:
        return "Diretório vazio."
    return _truncate("\n".join(lines))


def tool_find_files(pattern: str = "*", path: str = ".") -> str:
    """Busca arquivos recursivamente em um diretório usando padrão glob."""
    if not pattern or not pattern.strip():
        return "ERRO: padrão de busca não pode ser vazio"

    target = pathlib.Path(path)
    if not target.exists():
        return f"ERRO: caminho não encontrado: {path}"
    if not target.is_dir():
        return f"ERRO: o caminho não é um diretório: {path}"

    normalized_pattern = pattern.replace("\\", "/")
    match_rel_path = "/" in normalized_pattern
    search_root = target.resolve()
    matches: list[str] = []

    for dirpath, dirnames, filenames in os.walk(search_root):
        current_dir = pathlib.Path(dirpath)
        rel_dir_parts = current_dir.relative_to(search_root).parts
        rel_prefix = "/".join(rel_dir_parts) + "/" if rel_dir_parts else ""

        dirnames[:] = _filter_walk_dirs(dirpath, current_dir, dirnames, search_root)

        for filename in filenames:
            rel_path = f"{rel_prefix}{filename}"
            candidate = rel_path if match_rel_path else filename
            if not fnmatch.fnmatch(candidate, normalized_pattern):
                continue
            file_str = os.path.join(dirpath, filename)
            if _resolve_safe_walk_entry(file_str, current_dir / filename, search_root) is None:
                continue
            matches.append(rel_path)

    if not matches:
        return "Nenhum arquivo encontrado."

    matches.sort()
    return _truncate("\n".join(matches))


TOOLS = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "search_replace": tool_search_replace,
    "grep": tool_grep,
    "list_dir": tool_list_dir,
    "find_files": tool_find_files,
}

TOOL_SCHEMAS = [
    {
        "name": "bash",
        "description": "Executa um comando de shell no ambiente do sistema operacional.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "A linha de comando exata a ser executada."}
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "read_file",
        "description": "Lê o conteúdo de um arquivo com numeração de linhas e paginação opcional.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho absoluto ou relativo do arquivo a ser lido."},
                "offset": {"type": "integer", "description": "Linha inicial de leitura (base 0). Padrão é 0."},
                "limit": {
                    "type": "integer",
                    "description": "Quantidade máxima de linhas a serem lidas. Padrão é 2000.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Cria ou sobrescreve um arquivo com o conteúdo textual fornecido.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho do arquivo a ser gravado."},
                "content": {"type": "string", "description": "Conteúdo textual completo a ser gravado no arquivo."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_replace",
        "description": "Substitui uma ocorrência única de texto dentro de um arquivo.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho do arquivo a ser modificado."},
                "search": {
                    "type": "string",
                    "description": "Texto exato a ser encontrado (deve ser único no arquivo).",
                },
                "replace": {"type": "string", "description": "Novo texto que substituirá o termo de busca."},
            },
            "required": ["path", "search", "replace"],
        },
    },
    {
        "name": "grep",
        "description": "Busca texto literal ou expressão regular em um arquivo ou recursivamente em um diretório.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Texto literal ou padrão regex a ser buscado.",
                },
                "path": {
                    "type": "string",
                    "description": "Arquivo ou diretório alvo da busca. Padrão é '.'.",
                },
                "regex": {
                    "type": "boolean",
                    "description": "Se verdadeiro, interpreta 'pattern' como expressão regular. Padrão é false.",
                },
                "file_pattern": {
                    "type": "string",
                    "description": (
                        "Filtro glob para nomes de arquivos ao percorrer diretórios (ex.: '*.py'). Padrão é '*'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Número máximo de ocorrências retornadas. Padrão é 50.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "Lista arquivos e subdiretórios de um diretório de forma hierárquica "
            "até a profundidade máxima especificada."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Diretório alvo a ser listado. Padrão é '.'.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Profundidade máxima de listagem (1 = filhos imediatos). Padrão é 1.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_files",
        "description": (
            "Busca arquivos recursivamente em um diretório usando padrão glob "
            "(ex.: '*.py' ou 'pkg/*.py')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Padrão glob para filtrar nomes ou caminhos relativos de arquivos "
                        "(ex.: '*.py', 'pkg/*.py'). Padrão é '*'."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Diretório base da busca. Padrão é '.'.",
                },
            },
            "required": [],
        },
    },
]


# ---- Contexto hierárquico em Markdown (descoberta de AGENTS.md) ----------------
def discover_context(cwd: pathlib.Path = pathlib.Path.cwd()) -> str:
    """Varre os diretórios da raiz até a pasta atual em busca de arquivos AGENTS.md."""
    parts = []
    for p in reversed([cwd, *cwd.parents]):  # da raiz até o diretório atual
        md = p / "AGENTS.md"
        if md.exists():
            parts.append(f"<ctx path='{md}'>\n{md.read_text(encoding='utf-8', errors='replace')}\n</ctx>")
    return "\n".join(parts)


def _serialize_message(msg: dict[str, Any]) -> dict[str, Any]:
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


# ---- Agente e Loop de Execução -------------------------------------------------
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
            raise StopIteration(f"max_turns atingido ({self.max_turns})")
        if self.cost >= self.max_cost:
            raise StopIteration(f"max_cost atingido (${self.max_cost:.2f})")

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
        # Preserva o prompt de sistema + os últimos 30% dos turnos na íntegra, descartando o meio
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
        for guardrail in self.guardrails:
            if hasattr(guardrail, "reset"):
                guardrail.reset()

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
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        now_iso = datetime.now().isoformat()
        meta = dict(metadata) if metadata else {}

        payload: dict[str, Any] = {
            "session_id": meta.pop("session_id", target.stem),
            "started_at": meta.pop("started_at", now_iso),
            "updated_at": meta.pop("updated_at", now_iso),
            "n_turns": self.n_turns,
            "cost": self.cost,
            "messages": [_serialize_message(m) for m in self.messages],
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

    def load_from_file(self, path: str | pathlib.Path) -> dict[str, Any]:
        """Carrega o estado salvo de um arquivo JSON e restaura messages, n_turns e cost."""
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

        self.messages = [dict(m) for m in raw_messages]
        self.n_turns = n_turns
        self.cost = float(cost)
        self._checkpoints.clear()
        self.last_rollback_message = None
        for guardrail in self.guardrails:
            if hasattr(guardrail, "reset"):
                guardrail.reset()
        return data

    async def run(self, task: str) -> str:
        """Executa um turno de raciocínio e ação do agente para atender à tarefa."""
        self.last_rollback_message = None

        # Se for o início da sessão, inicializa com a instrução do sistema
        if not self.messages:
            default_instructions = (
                f"Você é um agente de engenharia de software conciso e prestativo.\n\n{discover_context()}"
            )
            system_content = self.instructions or default_instructions
            self.messages.append({"role": "system", "content": system_content})

        # Adiciona a mensagem do usuário à memória contínua da sessão
        self.messages.append({"role": "user", "content": task})

        checkpoint: WorkspaceCheckpoint | None = None
        if self.checkpointing:
            checkpoint = WorkspaceCheckpoint(root_dir=self.workspace_dir)
            checkpoint.snapshot()

        try:
            while True:
                await self._check_limits()
                await self._maybe_compact()
                resp = await self.model.complete(self.messages, tools=TOOL_SCHEMAS)
                self.n_turns += 1
                self.cost += resp.get("cost", 0.0)

                # Normaliza mensagem do assistente para o histórico
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

                    effective_args = dict(call_args) if isinstance(call_args, dict) else call_args
                    blocked = False
                    blocked_reason: str | None = None

                    for guardrail in self.guardrails:
                        if not getattr(guardrail, "enabled", True):
                            continue
                        check = guardrail.check_tool_call(call_name, effective_args)
                        if check.modified_args is not None:
                            effective_args = check.modified_args
                        if not check.allowed:
                            blocked = True
                            blocked_reason = check.reason
                            break

                    if blocked:
                        if not blocked_reason:
                            out: Any = "Blocked by safety rules."
                        elif blocked_reason.lower().startswith("blocked"):
                            out = blocked_reason
                        else:
                            out = f"Blocked by guardrail: {blocked_reason}"
                    else:
                        try:
                            fn = TOOLS.get(call_name)
                            if not fn:
                                out = f"ERRO: Ferramenta desconhecida '{call_name}'"
                            else:
                                out = fn(**effective_args)
                        except Exception as e:
                            out = f"ERRO: {type(e).__name__}: {e}"

                    sanitized = str(out)
                    for guardrail in self.guardrails:
                        if not getattr(guardrail, "enabled", True):
                            continue
                        sanitized = guardrail.sanitize_output(call_name, sanitized)

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call_name,
                            "content": _truncate(sanitized),
                        }
                    )
        except BaseException as exc:
            # StopIteration (max_turns/max_cost) preserva trabalho parcial como parada esperada;
            # qualquer outra falha ou interrupção dispara rollback automático do workspace.
            if checkpoint is not None and checkpoint.is_active:
                if _is_limit_stop(exc):
                    self._record_checkpoint(checkpoint)
                else:
                    result = checkpoint.rollback()
                    self.last_rollback_message = result.message
            raise
