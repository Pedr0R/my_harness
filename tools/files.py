from __future__ import annotations

import pathlib

from tools.common import ToolSpec, _truncate


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


FILE_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="read_file",
        description="Lê o conteúdo de um arquivo com numeração de linhas e paginação opcional.",
        parameters={
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
        handler=tool_read_file,
        checks_path=True,
    ),
    ToolSpec(
        name="write_file",
        description="Cria ou sobrescreve um arquivo com o conteúdo textual fornecido.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho do arquivo a ser gravado."},
                "content": {"type": "string", "description": "Conteúdo textual completo a ser gravado no arquivo."},
            },
            "required": ["path", "content"],
        },
        handler=tool_write_file,
        checks_path=True,
    ),
    ToolSpec(
        name="search_replace",
        description="Substitui uma ocorrência única de texto dentro de um arquivo.",
        parameters={
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
        handler=tool_search_replace,
        checks_path=True,
    ),
)
