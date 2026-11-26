from __future__ import annotations

import asyncio
import pathlib
import subprocess

from tools.common import ToolSpec, _truncate


def tool_bash(cmd: str, *, cwd: str | pathlib.Path | None = None) -> str:
    """Executa um comando bash/shell no sistema operacional."""
    resolved_cwd = str(pathlib.Path(cwd).resolve()) if cwd is not None else None
    r = subprocess.run(
        cmd,
        shell=True,
        cwd=resolved_cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return _truncate(f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")


async def tool_bash_async(cmd: str, *, cwd: str | pathlib.Path | None = None) -> str:
    """Executa tool_bash em thread dedicada para não bloquear o event loop asyncio."""
    return await asyncio.to_thread(tool_bash, cmd, cwd=cwd)


BASH_SPEC = ToolSpec(
    name="bash",
    description="Executa um comando de shell no ambiente do sistema operacional.",
    parameters={
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "A linha de comando exata a ser executada."}
        },
        "required": ["cmd"],
    },
    handler=tool_bash_async,
    requires_approval=True,
)
