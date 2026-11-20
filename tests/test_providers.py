from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
from types import SimpleNamespace
import unittest
from typing import Any
from unittest import mock

from google.genai import types

from providers import (
    FallbackProvider,
    GeminiProvider,
    ModelProvider,
    ModelResponse,
    OpenAICompatibleProvider,
    ToolCall,
    get_provider,
)


class StubProvider:
    def __init__(
        self,
        model: str,
        response: ModelResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.model = model
        self.response = response
        self.error = error
        self.call_count = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.response or ModelResponse(content="ok")


class TestBaseDataclasses(unittest.TestCase):
    def test_tool_call_accessors(self) -> None:
        tc = ToolCall(id="tc_1", name="read_file", args={"path": "a.py"})
        self.assertEqual(tc.id, "tc_1")
        self.assertEqual(tc["name"], "read_file")
        self.assertEqual(tc["args"], {"path": "a.py"})
        self.assertEqual(tc.get("name"), "read_file")
        self.assertEqual(tc.get("missing", "fallback"), "fallback")

    def test_model_response_accessors_and_to_dict(self) -> None:
        tc = ToolCall(id="tc_1", name="bash", args={"cmd": "echo hi"})
        resp = ModelResponse(content="done", tool_calls=[tc], cost=0.05)

        self.assertEqual(resp["content"], "done")
        self.assertEqual(resp.get("cost"), 0.05)
        self.assertEqual(resp.get("unknown", 99), 99)

        as_dict = resp.to_dict()
        self.assertEqual(
            as_dict,
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"id": "tc_1", "name": "bash", "args": {"cmd": "echo hi"}}],
                "cost": 0.05,
            },
        )
        self.assertNotIn("raw_content", as_dict)

        raw = {"role": "model", "parts": [{"text": "done"}]}
        resp_with_raw = ModelResponse(content="done", raw_content=raw)
        self.assertEqual(resp_with_raw.to_dict()["raw_content"], raw)


class TestGetProviderFactory(unittest.TestCase):
    def test_builds_each_known_provider_and_env_default(self) -> None:
        with mock.patch("providers.gemini.genai.Client") as mock_client:
            with mock.patch.dict(os.environ, {}, clear=True):
                gemini = get_provider("gemini", model="gemini-test", api_key="fake-key")
                self.assertIsInstance(gemini, GeminiProvider)
                self.assertIsInstance(gemini, ModelProvider)
                self.assertEqual(gemini.model_name, "gemini-test")
                mock_client.assert_called_with(api_key="fake-key")

                grok = get_provider("grok", api_key="k-grok")
                self.assertIsInstance(grok, OpenAICompatibleProvider)
                self.assertEqual(grok.base_url, "https://api.x.ai/v1")
                self.assertEqual(grok.model, "grok-2-latest")

                openai_p = get_provider("openai", model="gpt-test", api_key="k-oai")
                self.assertIsInstance(openai_p, OpenAICompatibleProvider)
                self.assertEqual(openai_p.base_url, "https://api.openai.com/v1")
                self.assertEqual(openai_p.model, "gpt-test")

                deepseek = get_provider("deepseek", api_key="k-ds")
                self.assertIsInstance(deepseek, OpenAICompatibleProvider)
                self.assertEqual(deepseek.base_url, "https://api.deepseek.com")
                self.assertEqual(deepseek.model, "deepseek-chat")

                ollama = get_provider("ollama")
                self.assertIsInstance(ollama, OpenAICompatibleProvider)
                self.assertEqual(ollama.api_key, "ollama")
                self.assertEqual(ollama.base_url, "http://localhost:11434/v1")

                fb = get_provider("fallback", providers=[grok, ollama])
                self.assertIsInstance(fb, FallbackProvider)
                self.assertEqual(fb.providers, [grok, ollama])

            with mock.patch.dict(os.environ, {"DEFAULT_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen2.5"}, clear=True):
                default_p = get_provider()
                self.assertIsInstance(default_p, OpenAICompatibleProvider)
                self.assertEqual(default_p.model, "qwen2.5")

    def test_invalid_factory_arguments_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_provider("fallback")

        with self.assertRaises(ValueError):
            get_provider("unknown_vendor")


class TestFallbackProvider(unittest.TestCase):
    def test_empty_providers_list_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            FallbackProvider([])

    def test_returns_primary_without_calling_secondary_when_healthy(self) -> None:
        primary = StubProvider("primary", response=ModelResponse(content="from primary"))
        secondary = StubProvider("secondary", response=ModelResponse(content="from secondary"))
        fb = FallbackProvider([primary, secondary])

        res = asyncio.run(fb.complete([{"role": "user", "content": "hi"}], []))

        self.assertEqual(res.content, "from primary")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(secondary.call_count, 0)

    def test_fails_over_to_secondary_when_primary_raises(self) -> None:
        primary = StubProvider("primary", error=RuntimeError("rate limited"))
        secondary = StubProvider("secondary", response=ModelResponse(content="from secondary"))
        fb = FallbackProvider([primary, secondary])

        res = asyncio.run(fb.complete([{"role": "user", "content": "hi"}], []))

        self.assertEqual(res.content, "from secondary")
        self.assertEqual(primary.call_count, 1)
        self.assertEqual(secondary.call_count, 1)

    def test_aggregates_errors_when_all_providers_fail(self) -> None:
        p1 = StubProvider("m1", error=TimeoutError("timed out"))
        p2 = StubProvider("m2", error=RuntimeError("503 unavailable"))
        fb = FallbackProvider([p1, p2])

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(fb.complete([{"role": "user", "content": "hi"}], []))

        msg = str(ctx.exception)
        self.assertIn("m1", msg)
        self.assertIn("timed out", msg)
        self.assertIn("m2", msg)
        self.assertIn("503 unavailable", msg)


class TestOpenAICompatibleProvider(unittest.TestCase):
    def _run_with_server(
        self,
        status_code: int,
        response_body: dict[str, Any] | str,
        coro_fn: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        captured: list[dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                captured.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(raw_body),
                    }
                )
                payload_bytes = (
                    json.dumps(response_body).encode("utf-8")
                    if isinstance(response_body, dict)
                    else response_body.encode("utf-8")
                )
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{port}/v1"
            result = asyncio.run(coro_fn(base_url))
            return result, captured
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_complete_text_response_and_request_formatting(self) -> None:
        response_json = {
            "choices": [{"message": {"role": "assistant", "content": "Hello back!"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "x.py"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "contents"},
        ]
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]

        async def call(base_url: str) -> ModelResponse:
            provider = OpenAICompatibleProvider(model="test-model", api_key="secret-token", base_url=base_url)
            return await provider.complete(messages, tools)

        resp, captured = self._run_with_server(200, response_json, call)

        self.assertEqual(resp.content, "Hello back!")
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.cost, round((1000 * 2.5 + 500 * 10.0) / 1_000_000, 6))

        self.assertEqual(len(captured), 1)
        req = captured[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["authorization"], "Bearer secret-token")
        self.assertEqual(req["body"]["model"], "test-model")
        self.assertEqual(req["body"]["tools"][0]["type"], "function")
        self.assertEqual(req["body"]["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(req["body"]["messages"][2]["tool_calls"][0]["function"]["arguments"], '{"path": "x.py"}')
        self.assertEqual(req["body"]["messages"][3]["tool_call_id"], "c1")

    def test_complete_parses_valid_and_malformed_tool_call_json(self) -> None:
        response_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_ok",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                            },
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{not-json"},
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        async def call(base_url: str) -> ModelResponse:
            provider = OpenAICompatibleProvider(model="test-model", api_key="k", base_url=base_url)
            return await provider.complete([{"role": "user", "content": "run"}], [])

        resp, _ = self._run_with_server(200, response_json, call)

        self.assertEqual(resp.content, "")
        self.assertEqual(len(resp.tool_calls), 2)
        self.assertEqual(resp.tool_calls[0], ToolCall(id="call_ok", name="bash", args={"cmd": "ls"}))
        self.assertEqual(resp.tool_calls[1], ToolCall(id="call_bad", name="bash", args={"raw": "{not-json"}))

    def test_complete_raises_runtime_error_on_http_4xx_and_5xx(self) -> None:
        for status in (429, 500):
            async def call(base_url: str) -> ModelResponse:
                provider = OpenAICompatibleProvider(model="m", api_key="k", base_url=base_url)
                return await provider.complete([{"role": "user", "content": "hi"}], [])

            with self.assertRaises(RuntimeError) as ctx:
                self._run_with_server(status, {"error": "boom"}, call)
            self.assertIn(str(status), str(ctx.exception))

    def test_calculate_cost(self) -> None:
        provider = OpenAICompatibleProvider(model="m", api_key="k")
        self.assertEqual(provider._calculate_cost(None), 0.0)
        self.assertEqual(
            provider._calculate_cost({"prompt_tokens": 2000, "completion_tokens": 1000}),
            round((2000 * 2.5 + 1000 * 10.0) / 1_000_000, 6),
        )


class TestGeminiProvider(unittest.TestCase):
    def _make_provider(self) -> tuple[GeminiProvider, mock.MagicMock]:
        with mock.patch("providers.gemini.genai.Client") as mock_client_cls:
            mock_client = mock.MagicMock()
            mock_client_cls.return_value = mock_client
            provider = GeminiProvider(model="gemini-test", api_key="fake-key")
        return provider, mock_client

    def test_convert_messages_handles_roles_and_preserves_thought_signature(self) -> None:
        provider, _ = self._make_provider()
        live_raw = types.Content(
            role="model",
            parts=[types.Part(text="live", thought_signature=b"sig_live")],
        )
        dict_raw = types.Content(
            role="model",
            parts=[types.Part(text="rehydrated", thought_signature=b"sig_dict")],
        ).model_dump(mode="json", exclude_none=True)

        messages = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ignored", "raw_content": live_raw},
            {"role": "assistant", "content": "ignored", "raw_content": dict_raw},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [ToolCall(id="c1", name="read_file", args={"path": "a.txt"})],
            },
            {"role": "tool", "name": "read_file", "content": "file body"},
        ]

        system_instruction, contents = provider._convert_messages(messages)

        self.assertEqual(system_instruction, "sys1\n\nsys2")
        self.assertEqual(len(contents), 5)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "hello")
        self.assertIs(contents[1], live_raw)
        self.assertEqual(contents[1].parts[0].thought_signature, b"sig_live")
        self.assertEqual(contents[2].parts[0].thought_signature, b"sig_dict")
        self.assertEqual(contents[3].role, "model")
        self.assertEqual(contents[3].parts[0].text, "calling tool")
        self.assertEqual(contents[3].parts[1].function_call.name, "read_file")
        self.assertEqual(contents[4].role, "user")
        self.assertEqual(contents[4].parts[0].function_response.name, "read_file")
        self.assertEqual(contents[4].parts[0].function_response.response, {"result": "file body"})

    def test_convert_tools_and_calculate_cost(self) -> None:
        provider, _ = self._make_provider()
        tool = provider._convert_tools(
            [
                {
                    "name": "bash",
                    "description": "Run shell command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                }
            ]
        )
        self.assertIsInstance(tool, types.Tool)
        self.assertEqual(len(tool.function_declarations), 1)
        self.assertEqual(tool.function_declarations[0].name, "bash")

        self.assertEqual(provider._calculate_cost(None), 0.0)
        usage = SimpleNamespace(prompt_token_count=1000, candidates_token_count=500)
        self.assertEqual(
            provider._calculate_cost(usage),
            round((1000 * 0.10 + 500 * 0.40) / 1_000_000, 6),
        )

    def test_complete_extracts_text_function_calls_raw_content_and_cost(self) -> None:
        provider, mock_client = self._make_provider()
        native_content = types.Content(
            role="model",
            parts=[types.Part(text="Checking file...", thought_signature=b"sig_resp")],
        )
        fake_response = SimpleNamespace(
            candidates=[SimpleNamespace(content=native_content)],
            function_calls=[SimpleNamespace(name="read_file", args={"path": "README.md"})],
            usage_metadata=SimpleNamespace(prompt_token_count=2000, candidates_token_count=400),
        )
        mock_client.aio.models.generate_content = mock.AsyncMock(return_value=fake_response)

        tools = [
            {
                "name": "read_file",
                "description": "Read file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
        result = asyncio.run(
            provider.complete(
                [{"role": "system", "content": "be concise"}, {"role": "user", "content": "read"}],
                tools,
            )
        )

        self.assertEqual(result.content, "Checking file...")
        self.assertEqual(
            result.tool_calls,
            [ToolCall(id="call_read_file_0", name="read_file", args={"path": "README.md"})],
        )
        self.assertIs(result.raw_content, native_content)
        self.assertEqual(result.cost, round((2000 * 0.10 + 400 * 0.40) / 1_000_000, 6))

        mock_client.aio.models.generate_content.assert_awaited_once()
        kwargs = mock_client.aio.models.generate_content.await_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-test")
        self.assertEqual(kwargs["config"].system_instruction, "be concise")
        self.assertEqual(len(kwargs["config"].tools), 1)


if __name__ == "__main__":
    unittest.main()
