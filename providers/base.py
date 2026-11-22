from __future__ import annotations

from collections.abc import AsyncIterator
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
    raw_content: Any = None  # Preserva o objeto de conteúdo nativo (ex: thought_signature do Gemini)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Converte a resposta para o formato de dicionário compatível com o histórico."""
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in self.tool_calls
            ],
            "cost": self.cost,
        }
        if self.raw_content is not None:
            d["raw_content"] = self.raw_content
        return d


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


@dataclass(frozen=True)
class StreamChunk:
    """Evento incremental emitido por um provedor durante o streaming de um turno."""

    text_delta: str = ""
    response: ModelResponse | None = None


@runtime_checkable
class StreamingModelProvider(ModelProvider, Protocol):
    """Extensão opcional do protocolo para provedores com suporte nativo a streaming."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        """Emite chunks incrementais e encerra com a ModelResponse consolidada."""
        ...


async def stream_or_complete(
    provider: ModelProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> AsyncIterator[StreamChunk]:
    """Consome streaming nativo do provedor ou adapta complete() para emissão única."""
    stream_fn = getattr(provider, "stream", None)
    if callable(stream_fn):
        async for chunk in stream_fn(messages, tools):
            yield chunk
        return

    response = await provider.complete(messages, tools)
    if response.content:
        yield StreamChunk(text_delta=response.content)
    yield StreamChunk(response=response)
