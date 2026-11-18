from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

from providers.base import ModelProvider, ModelResponse, ToolCall

load_dotenv()


class GeminiProvider:
    """Adaptador para a API do Google Gemini utilizando o novo SDK google-genai."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
    ):
        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.project_id = project_id or os.getenv("GEMINI_PROJECT_ID")
        self.location = location or os.getenv("GEMINI_LOCATION")

        # No SDK google-genai, api_key e project/location são mutuamente exclusivos (Developer API vs Vertex AI)
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        elif self.project_id:
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location or "us-central1",
            )
        else:
            self.client = genai.Client()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        system_instruction, contents = self._convert_messages(messages)
        gemini_tools = self._convert_tools(tools) if tools else None

        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            tools=[gemini_tools] if gemini_tools else None,
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        # Extrai textos sem disparar o warning do SDK em respostas mistas ou puramente de tool calls
        content_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            text_parts = [
                part.text
                for part in response.candidates[0].content.parts
                if getattr(part, "text", None)
            ]
            content_text = "".join(text_parts)

        tool_calls: list[ToolCall] = []
        if response.function_calls:
            for i, fc in enumerate(response.function_calls):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc.name}_{i}",
                        name=fc.name,
                        args=dict(fc.args) if fc.args else {},
                    )
                )

        cost = self._calculate_cost(response.usage_metadata)

        # Captura o Content original para preservar thought_signature em turnos subsequentes com ferramentas
        raw_content = None
        if response.candidates and response.candidates[0].content:
            raw_content = response.candidates[0].content

        return ModelResponse(
            content=content_text,
            tool_calls=tool_calls,
            cost=cost,
            raw_content=raw_content,
        )

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[types.Content]]:
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")

            if role == "system":
                system_parts.append(str(msg.get("content", "")))
                continue

            if role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=str(msg.get("content", "")))],
                    )
                )

            elif role == "assistant":
                # Preserva o Content original com thought_signature intacto se disponível
                raw_c = msg.get("raw_content")
                if raw_c is not None and isinstance(raw_c, types.Content):
                    contents.append(raw_c)
                    continue

                parts: list[types.Part] = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=str(msg["content"])))
                for tc in msg.get("tool_calls", []):
                    name = tc["name"] if isinstance(tc, dict) else tc.name
                    args = tc["args"] if isinstance(tc, dict) else tc.args
                    parts.append(types.Part.from_function_call(name=name, args=args))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))

            elif role == "tool":
                name = msg.get("name", "function_result")
                content = msg.get("content", "")
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response={"result": str(content)},
                            )
                        ],
                    )
                )

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    def _convert_tools(self, tools: list[dict[str, Any]]) -> types.Tool:
        declarations = []
        for t in tools:
            decl: dict[str, Any] = {
                "name": t["name"],
                "description": t.get("description", t["name"]),
            }
            if "parameters" in t:
                decl["parameters"] = t["parameters"]
            declarations.append(decl)
        return types.Tool(function_declarations=declarations)

    def _calculate_cost(self, usage: Any) -> float:
        if not usage:
            return 0.0
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cost = (prompt_tokens * 0.10 + candidates_tokens * 0.40) / 1_000_000
        return round(cost, 6)
