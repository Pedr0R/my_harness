from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import pathlib
import subprocess
from typing import Any

from providers import ModelProvider, get_provider


# ---- Ferramentas: bash + 3 ferramentas de arquivos ----------------------------
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


TOOLS = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "search_replace": tool_search_replace,
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
                "limit": {"type": "integer", "description": "Quantidade máxima de linhas a serem lidas. Padrão é 2000."},
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
                "search": {"type": "string", "description": "Texto exato a ser encontrado (deve ser único no arquivo)."},
                "replace": {"type": "string", "description": "Novo texto que substituirá o termo de busca."},
            },
            "required": ["path", "search", "replace"],
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
        return data

    async def run(self, task: str) -> str:
        """Executa um turno de raciocínio e ação do agente para atender à tarefa."""
        # Se for o início da sessão, inicializa com a instrução do sistema
        if not self.messages:
            system_content = self.instructions or f"Você é um agente de engenharia de software conciso e prestativo.\n\n{discover_context()}"
            self.messages.append({"role": "system", "content": system_content})

        # Adiciona a mensagem do usuário à memória contínua da sessão
        self.messages.append({"role": "user", "content": task})

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
                return resp.get("content", "")

            # Execução sequencial das chamadas de ferramentas solicitadas pelo modelo
            for call in tool_calls:
                call_name = call["name"] if isinstance(call, dict) else call.name
                call_args = call["args"] if isinstance(call, dict) else call.args
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", f"call_{call_name}")

                try:
                    fn = TOOLS.get(call_name)
                    if not fn:
                        out = f"ERRO: Ferramenta desconhecida '{call_name}'"
                    else:
                        out = fn(**call_args)
                except Exception as e:
                    out = f"ERRO: {type(e).__name__}: {e}"

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": call_name,
                        "content": _truncate(str(out)),
                    }
                )
