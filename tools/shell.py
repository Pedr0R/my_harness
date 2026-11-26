from __future__ import annotations

import asyncio
import subprocess

from tools.common import ToolSpec, _truncate


def tool_bash(cmd: str) -> str:
    """Executa um comando bash/shell no sistema operacional."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return _truncate(f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")


async def tool_bash_async(cmd: str) -> str:
    """Executa tool_bash em thread dedicada para não bloquear o event loop asyncio."""
    return await asyncio.to_thread(tool_bash, cmd)


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
