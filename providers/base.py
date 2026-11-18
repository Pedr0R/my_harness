from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """Representa uma solicitação de chamada de ferramenta feita pelo modelo."""
    id: str
    name: str
    args: dict[str, Any]

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


@dataclass
class ModelResponse:
    """Resposta normalizada de um provedor de modelo para o loop do agente."""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    cost: float = 0.0
    role: str = "assistant"

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Converte a resposta para o formato de dicionário compatível com o histórico."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in self.tool_calls
            ],
            "cost": self.cost,
        }


@runtime_checkable
class ModelProvider(Protocol):
    """Protocolo que todo provedor de modelo deve implementar."""
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Recebe mensagens e ferramentas no formato padrão e retorna uma ModelResponse normalizada."""
        ...
