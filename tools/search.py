from __future__ import annotations

import fnmatch
import os
import pathlib
import re

from tools.common import (
    IGNORED_DIRS,
    ToolSpec,
    _filter_walk_dirs,
    _resolve_safe_walk_entry,
    _truncate,
)


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


SEARCH_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="grep",
        description="Busca texto literal ou expressão regular em um arquivo ou recursivamente em um diretório.",
        parameters={
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
        handler=tool_grep,
        checks_path=True,
        default_path=".",
    ),
    ToolSpec(
        name="list_dir",
        description=(
            "Lista arquivos e subdiretórios de um diretório de forma hierárquica "
            "até a profundidade máxima especificada."
        ),
        parameters={
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
        handler=tool_list_dir,
        checks_path=True,
        default_path=".",
    ),
    ToolSpec(
        name="find_files",
        description=(
            "Busca arquivos recursivamente em um diretório usando padrão glob "
            "(ex.: '*.py' ou 'pkg/*.py')."
        ),
        parameters={
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
        handler=tool_find_files,
        checks_path=True,
        default_path=".",
    ),
)
