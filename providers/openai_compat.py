from __future__ import annotations

import json
import os
from typing import Any
import httpx
from dotenv import load_dotenv

from providers.base import ModelProvider, ModelResponse, ToolCall

load_dotenv()


class OpenAICompatibleProvider:
    """
    Adaptador universal para provedores compatíveis com a API OpenAI Chat Completions.
    Funciona diretamente com OpenAI, Grok (xAI), DeepSeek, Ollama, Together AI, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Formata as ferramentas (tools) para o padrão JSON Schema da OpenAI
        formatted_tools = None
        if tools:
            formatted_tools = []
            for t in tools:
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )

        # Converte as mensagens universais para o formato esperado pela API
        formatted_messages = []
        for m in messages:
            msg_dict = {"role": m.get("role", "user"), "content": m.get("content", "")}
            if m.get("tool_calls"):
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.get("id") if isinstance(tc, dict) else tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.get("name") if isinstance(tc, dict) else tc.name,
                            "arguments": json.dumps(tc.get("args") if isinstance(tc, dict) else tc.args),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            if m.get("tool_call_id"):
                msg_dict["tool_call_id"] = m["tool_call_id"]
            formatted_messages.append(msg_dict)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        if formatted_tools:
            payload["tools"] = formatted_tools

        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Erro de API [{response.status_code}] retornado por {self.base_url}: {response.text}"
                )
            data = response.json()

        choice = data["choices"][0]["message"]
        content_text = choice.get("content") or ""

        tool_calls: list[ToolCall] = []
        if choice.get("tool_calls"):
            for tc in choice["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"raw": args_str}

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{fn.get('name')}"),
                        name=fn.get("name", ""),
                        args=args,
                    )
                )

        cost = self._calculate_cost(data.get("usage"))

        return ModelResponse(
            content=content_text,
            tool_calls=tool_calls,
            cost=cost,
        )

    def _calculate_cost(self, usage: dict[str, Any] | None) -> float:
        """Estima o custo financeiro aproximado com base na quantidade de tokens utilizados."""
        if not usage:
            return 0.0
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        # Estimativa genérica média por 1M tokens ($2.50 input / $10.00 output)
        return round((prompt_tokens * 2.5 + completion_tokens * 10.0) / 1_000_000, 6)
