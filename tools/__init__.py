from __future__ import annotations

from collections.abc import Awaitable
import functools
import pathlib
from typing import Any

from tools.common import IGNORED_DIRS, ToolHandler, ToolSpec, _truncate
from tools.files import FILE_TOOL_SPECS, tool_read_file, tool_search_replace, tool_write_file
from tools.search import SEARCH_TOOL_SPECS, tool_find_files, tool_grep, tool_list_dir
from tools.shell import BASH_SPEC, tool_bash, tool_bash_async

ALL_TOOL_SPECS: tuple[ToolSpec, ...] = (
    BASH_SPEC,
    *FILE_TOOL_SPECS,
    *SEARCH_TOOL_SPECS,
)

TOOLS: dict[str, ToolHandler] = {spec.name: spec.handler for spec in ALL_TOOL_SPECS}
TOOL_SCHEMAS: list[dict[str, Any]] = [spec.to_schema() for spec in ALL_TOOL_SPECS]
FILE_TOOLS: frozenset[str] = frozenset(spec.name for spec in ALL_TOOL_SPECS if spec.checks_path)
DEFAULT_PATH_TOOLS: frozenset[str] = frozenset(
    spec.name for spec in ALL_TOOL_SPECS if spec.default_path is not None
)
APPROVAL_REQUIRED_TOOLS: frozenset[str] = frozenset(
    spec.name for spec in ALL_TOOL_SPECS if spec.requires_approval
)


def _bind_path_tool(
    fn: ToolHandler,
    resolved_root: pathlib.Path | None,
) -> ToolHandler:
    fn_name = getattr(fn, "__name__", "tool")

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> str | Awaitable[str]:
        if "root_dir" in kwargs:
            raise TypeError(f"{fn_name}() got an unexpected keyword argument 'root_dir'")
        return fn(*args, **kwargs, root_dir=resolved_root)

    return _wrapped


def _bind_bash_tool(resolved_root: pathlib.Path | None) -> ToolHandler:
    @functools.wraps(tool_bash_async)
    async def _wrapped(cmd: str) -> str:
        return await tool_bash_async(cmd, cwd=resolved_root)

    return _wrapped


def build_tools(root_dir: str | pathlib.Path | None = None) -> dict[str, ToolHandler]:
    """Constrói o mapa de ferramentas nativas vinculadas a um diretório raiz específico."""
    resolved_root = pathlib.Path(root_dir).resolve() if root_dir is not None else None
    bound: dict[str, ToolHandler] = {}
    for spec in ALL_TOOL_SPECS:
        if spec.name == "bash":
            bound[spec.name] = _bind_bash_tool(resolved_root)
        elif spec.checks_path:
            bound[spec.name] = _bind_path_tool(spec.handler, resolved_root)
        else:
            bound[spec.name] = spec.handler
    return bound


__all__ = [
    "ALL_TOOL_SPECS",
    "APPROVAL_REQUIRED_TOOLS",
    "DEFAULT_PATH_TOOLS",
    "FILE_TOOLS",
    "IGNORED_DIRS",
    "TOOLS",
    "TOOL_SCHEMAS",
    "ToolHandler",
    "ToolSpec",
    "_truncate",
    "build_tools",
    "tool_bash",
    "tool_bash_async",
    "tool_find_files",
    "tool_grep",
    "tool_list_dir",
    "tool_read_file",
    "tool_search_replace",
    "tool_write_file",
]
