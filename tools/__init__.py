from __future__ import annotations

from typing import Any, Callable

from tools.common import IGNORED_DIRS, ToolSpec, _truncate
from tools.files import FILE_TOOL_SPECS, tool_read_file, tool_search_replace, tool_write_file
from tools.search import SEARCH_TOOL_SPECS, tool_find_files, tool_grep, tool_list_dir
from tools.shell import BASH_SPEC, tool_bash

ALL_TOOL_SPECS: tuple[ToolSpec, ...] = (
    BASH_SPEC,
    *FILE_TOOL_SPECS,
    *SEARCH_TOOL_SPECS,
)

TOOLS: dict[str, Callable[..., str]] = {spec.name: spec.handler for spec in ALL_TOOL_SPECS}
TOOL_SCHEMAS: list[dict[str, Any]] = [spec.to_schema() for spec in ALL_TOOL_SPECS]
FILE_TOOLS: frozenset[str] = frozenset(spec.name for spec in ALL_TOOL_SPECS if spec.checks_path)
DEFAULT_PATH_TOOLS: frozenset[str] = frozenset(
    spec.name for spec in ALL_TOOL_SPECS if spec.default_path is not None
)

__all__ = [
    "ALL_TOOL_SPECS",
    "DEFAULT_PATH_TOOLS",
    "FILE_TOOLS",
    "IGNORED_DIRS",
    "TOOLS",
    "TOOL_SCHEMAS",
    "ToolSpec",
    "_truncate",
    "tool_bash",
    "tool_find_files",
    "tool_grep",
    "tool_list_dir",
    "tool_read_file",
    "tool_search_replace",
    "tool_write_file",
]
