from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from typing import Any
from unittest import mock

from google.genai import types

from agent import Agent, AgentLimitError, discover_context
from guardrails.base import BaseGuardrail, GuardrailResult
import main
from providers.base import ModelResponse, ToolCall
from tools import TOOL_SCHEMAS


class FakeProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append(([dict(m) for m in messages], list(tools)))
        if not self._responses:
            return ModelResponse(content="")
        return self._responses.pop(0)


class TrackingGuardrail(BaseGuardrail):
    name = "tracking"

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.reset_count = 0

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> GuardrailResult:
        return GuardrailResult(allowed=True)

    def reset(self) -> None:
        self.reset_count += 1


class TestAgentInitializationAndLoop(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_explicit_instructions_set_system_prompt_once(self) -> None:
        provider = FakeProvider([ModelResponse(content="first"), ModelResponse(content="second")])
        agent = Agent(model=provider, instructions="Custom system prompt.")

        ans1 = asyncio.run(agent.run("hello"))
        ans2 = asyncio.run(agent.run("again"))

        self.assertEqual(ans1, "first")
        self.assertEqual(ans2, "second")
        system_msgs = [m for m in agent.messages if m["role"] == "system"]
        self.assertEqual(system_msgs, [{"role": "system", "content": "Custom system prompt."}])

    def test_default_instructions_include_discovered_context(self) -> None:
        provider = FakeProvider([ModelResponse(content="ready")])
        agent = Agent(model=provider)

        with mock.patch("agent.discover_context", return_value="<ctx path='x'>PROJECT_CTX</ctx>"):
            asyncio.run(agent.run("ping"))

        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertIn("PROJECT_CTX", agent.messages[0]["content"])

    def test_discover_context_uses_current_working_directory_dynamically(self) -> None:
        sub = self.root / "dynamic_proj"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("DYNAMIC_CTX_RULES", encoding="utf-8")

        prev_cwd = pathlib.Path.cwd()
        try:
            os.chdir(sub)
            ctx = discover_context()
        finally:
            os.chdir(prev_cwd)

        self.assertIn("DYNAMIC_CTX_RULES", ctx)

    def test_multi_step_tool_calls_feed_results_back_to_model(self) -> None:
        target = self.root / "hello.txt"
        provider = FakeProvider(
            [
                ModelResponse(
                    content="writing",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="write_file",
                            args={"path": str(target), "content": "payload_123"},
                        )
                    ],
                    cost=0.01,
                ),
                ModelResponse(
                    content="reading",
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="read_file",
                            args={"path": str(target)},
                        )
                    ],
                    cost=0.02,
                ),
                ModelResponse(content="All done.", cost=0.03),
            ]
        )
        agent = Agent(model=provider, instructions="sys")

        result = asyncio.run(agent.run("create and read file"))

        self.assertEqual(result, "All done.")
        self.assertEqual(target.read_text(encoding="utf-8"), "payload_123")
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(provider.calls[0][1], TOOL_SCHEMAS)

        second_turn_messages = provider.calls[1][0]
        self.assertEqual(second_turn_messages[-1]["role"], "tool")
        self.assertEqual(second_turn_messages[-1]["tool_call_id"], "call_1")
        self.assertEqual(second_turn_messages[-1]["name"], "write_file")
        self.assertIn("escreveu", second_turn_messages[-1]["content"])

        third_turn_messages = provider.calls[2][0]
        self.assertEqual(third_turn_messages[-1]["role"], "tool")
        self.assertEqual(third_turn_messages[-1]["tool_call_id"], "call_2")
        self.assertIn("payload_123", third_turn_messages[-1]["content"])

    def test_unknown_tool_and_tool_exception_return_error_messages_without_crashing(self) -> None:
        missing_path = self.root / "missing.txt"
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="c_unknown", name="nonexistent_tool", args={}),
                        ToolCall(id="c_boom", name="read_file", args={"path": str(missing_path)}),
                    ]
                ),
                ModelResponse(content="Recovered from errors."),
            ]
        )
        agent = Agent(model=provider, instructions="sys")

        result = asyncio.run(agent.run("trigger errors"))

        self.assertEqual(result, "Recovered from errors.")
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        self.assertTrue(tool_msgs[0]["content"].startswith("ERRO:"))
        self.assertIn("nonexistent_tool", tool_msgs[0]["content"])
        self.assertTrue(tool_msgs[1]["content"].startswith("ERRO: FileNotFoundError:"))

    def test_custom_injected_tools_and_schemas_override_defaults(self) -> None:
        called: list[str] = []

        def custom_ping(msg: str) -> str:
            called.append(msg)
            return f"pong:{msg}"

        custom_schemas = [
            {
                "name": "ping",
                "description": "Custom ping tool",
                "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
            }
        ]
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=[ToolCall(id="c1", name="ping", args={"msg": "oi"})]),
                ModelResponse(tool_calls=[ToolCall(id="c2", name="bash", args={"cmd": "echo hi"})]),
                ModelResponse(content="ok"),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            tools={"ping": custom_ping},
            tool_schemas=custom_schemas,
        )

        result = asyncio.run(agent.run("teste"))

        self.assertEqual(result, "ok")
        self.assertEqual(called, ["oi"])
        self.assertEqual(provider.calls[0][1], custom_schemas)
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(tool_msgs[0]["content"], "pong:oi")
        self.assertIn("Ferramenta desconhecida 'bash'", tool_msgs[1]["content"])


class TestAgentLimitsAndCompaction(unittest.TestCase):
    def test_turn_and_cost_accumulation_and_max_turns_limit(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(content="one", cost=0.15),
                ModelResponse(content="two", cost=0.25),
                ModelResponse(content="three", cost=0.35),
            ]
        )
        agent = Agent(model=provider, instructions="sys", max_turns=2, max_cost=10.0)

        asyncio.run(agent.run("turn 1"))
        asyncio.run(agent.run("turn 2"))
        self.assertEqual(agent.n_turns, 2)
        self.assertAlmostEqual(agent.cost, 0.40)

        with self.assertRaises(AgentLimitError) as ctx:
            asyncio.run(agent.run("turn 3"))
        self.assertIn("max_turns", str(ctx.exception))
        self.assertEqual(len(provider.calls), 2)

    def test_max_cost_limit_raises_stop_iteration(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(content="expensive", cost=1.50),
                ModelResponse(content="never reached", cost=0.10),
            ]
        )
        agent = Agent(model=provider, instructions="sys", max_turns=10, max_cost=1.00)

        asyncio.run(agent.run("first"))
        self.assertAlmostEqual(agent.cost, 1.50)

        with self.assertRaises(AgentLimitError) as ctx:
            asyncio.run(agent.run("second"))
        self.assertIn("max_cost", str(ctx.exception))
        self.assertEqual(len(provider.calls), 1)

    def test_maybe_compact_skips_when_below_threshold(self) -> None:
        provider = FakeProvider([ModelResponse(content="should not be called")])
        agent = Agent(
            model=provider,
            compact_at_tokens=10_000,
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )

        asyncio.run(agent._maybe_compact())

        self.assertEqual(len(provider.calls), 0)
        self.assertEqual(len(agent.messages), 2)

    def test_maybe_compact_preserves_system_inserts_summary_and_keeps_min_four_messages(self) -> None:
        provider = FakeProvider([ModelResponse(content="SUMMARY_SHORT")])
        history = [{"role": "system", "content": "SYS_PROMPT"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg_{i}_" + ("x" * 30)}
            for i in range(7)
        ]
        agent = Agent(model=provider, compact_at_tokens=100, messages=list(history))

        asyncio.run(agent._maybe_compact())

        self.assertEqual(len(provider.calls), 1)
        sent_messages, sent_tools = provider.calls[0]
        self.assertEqual(sent_tools, [])
        self.assertEqual(sent_messages[-1]["role"], "user")

        self.assertEqual(agent.messages[0], history[0])
        self.assertEqual(agent.messages[1], {"role": "assistant", "content": "SUMMARY_SHORT"})
        self.assertEqual(agent.messages[2:], history[-4:])

    def test_maybe_compact_keeps_thirty_percent_when_larger_than_four(self) -> None:
        provider = FakeProvider([ModelResponse(content="SUMMARY_LONG")])
        history = [{"role": "system", "content": "SYS_PROMPT"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn_{i}_" + ("y" * 20)}
            for i in range(19)
        ]
        expected_keep = int(len(history) * 0.30)
        self.assertGreater(expected_keep, 4)

        agent = Agent(model=provider, compact_at_tokens=100, messages=list(history))
        asyncio.run(agent._maybe_compact())

        self.assertEqual(agent.messages[0], history[0])
        self.assertEqual(agent.messages[1], {"role": "assistant", "content": "SUMMARY_LONG"})
        self.assertEqual(agent.messages[2:], history[-expected_keep:])


class TestAgentSessionPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reset_session_and_get_session_memory(self) -> None:
        guardrail = TrackingGuardrail()
        provider = FakeProvider([ModelResponse(content="ok", cost=0.12)])
        agent = Agent(model=provider, name="TestBot", instructions="sys", guardrails=[guardrail])

        asyncio.run(agent.run("hello"))
        mem = agent.get_session_memory()
        self.assertEqual(mem["agent_name"], "TestBot")
        self.assertEqual(mem["turns"], 1)
        self.assertAlmostEqual(mem["estimated_cost"], 0.12)
        self.assertEqual(len(mem["messages"]), 3)

        mem["messages"].clear()
        self.assertEqual(len(agent.messages), 3)

        agent.reset_session()
        self.assertEqual(agent.messages, [])
        self.assertEqual(agent.n_turns, 0)
        self.assertEqual(guardrail.reset_count, 1)

    def test_save_and_load_roundtrip_with_pydantic_raw_content_and_metadata(self) -> None:
        raw_content = types.Content(
            role="model",
            parts=[types.Part(text="thinking", thought_signature=b"sig_bytes")],
        )
        provider = FakeProvider(
            [
                ModelResponse(content="hello", cost=0.25, raw_content=raw_content),
                ModelResponse(content="plain", cost=0.05, raw_content=None),
            ]
        )
        agent = Agent(model=provider, instructions="sys")
        asyncio.run(agent.run("q1"))
        asyncio.run(agent.run("q2"))

        session_path = self.root / "nested" / "session_1.json"
        saved_path = agent.save_to_file(
            session_path,
            metadata={"session_id": "custom_id", "history": ["q1", "hello"], "tag": "v1"},
        )
        self.assertEqual(saved_path, session_path)
        self.assertEqual(list(session_path.parent.glob("*.tmp")), [])

        raw_json = json.loads(session_path.read_text(encoding="utf-8"))
        self.assertEqual(raw_json["session_id"], "custom_id")
        self.assertEqual(raw_json["history"], ["q1", "hello"])
        self.assertEqual(raw_json["tag"], "v1")
        assistant_msgs = [m for m in raw_json["messages"] if m["role"] == "assistant"]
        self.assertIn("raw_content", assistant_msgs[0])
        self.assertIsInstance(assistant_msgs[0]["raw_content"], dict)
        self.assertNotIn("raw_content", assistant_msgs[1])

        rehydrated = types.Content.model_validate(assistant_msgs[0]["raw_content"])
        self.assertEqual(rehydrated.parts[0].thought_signature, b"sig_bytes")

        guardrail = TrackingGuardrail()
        restored = Agent(model=FakeProvider([]), guardrails=[guardrail])
        loaded_data = restored.load_from_file(session_path)

        self.assertEqual(loaded_data["session_id"], "custom_id")
        self.assertEqual(restored.n_turns, 2)
        self.assertAlmostEqual(restored.cost, 0.30)
        self.assertEqual(len(restored.messages), len(agent.messages))
        self.assertEqual(guardrail.reset_count, 1)

    def test_load_from_file_rejects_missing_corrupt_and_invalid_payloads(self) -> None:
        agent = Agent(model=FakeProvider([]))

        with self.assertRaises(FileNotFoundError):
            agent.load_from_file(self.root / "missing.json")

        corrupt = self.root / "corrupt.json"
        corrupt.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(ValueError):
            agent.load_from_file(corrupt)

        invalid_payloads = [
            "[]",
            json.dumps({"messages": "not_a_list"}),
            json.dumps({"messages": [123]}),
            json.dumps({"messages": [], "n_turns": -1}),
            json.dumps({"messages": [], "n_turns": True}),
            json.dumps({"messages": [], "cost": -0.5}),
            json.dumps({"messages": [], "cost": False}),
        ]
        for idx, body in enumerate(invalid_payloads):
            path = self.root / f"invalid_{idx}.json"
            path.write_text(body, encoding="utf-8")
            with self.assertRaises(ValueError, msg=f"Failed for payload {idx}: {body}"):
                agent.load_from_file(path)


class FailingStepProvider:
    def __init__(self, steps: list[ModelResponse | BaseException]) -> None:
        self._steps = list(steps)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class TestAgentWorkspaceCheckpointing(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name).resolve() / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo, check=True)
        (self.repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=self.repo, check=True, capture_output=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_auto_rollback_on_unhandled_provider_error_after_tool_edit(self) -> None:
        app_file = self.repo / "app.py"
        new_file = self.repo / "generated.txt"
        provider = FailingStepProvider(
            [
                ModelResponse(
                    content="editing files",
                    tool_calls=[
                        ToolCall(
                            id="call_edit",
                            name="write_file",
                            args={"path": str(app_file), "content": "VALUE = 999\n"},
                        ),
                        ToolCall(
                            id="call_create",
                            name="write_file",
                            args={"path": str(new_file), "content": "partial\n"},
                        ),
                    ],
                ),
                RuntimeError("provider outage on turn 2"),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            checkpointing=True,
            workspace_dir=self.repo,
        )

        with self.assertRaisesRegex(RuntimeError, "provider outage"):
            asyncio.run(agent.run("update app"))

        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 1\n")
        self.assertFalse(new_file.exists())
        self.assertIsNotNone(agent.last_rollback_message)
        self.assertIn("Workspace restaurado", agent.last_rollback_message or "")

    def test_undo_last_turn_reverts_completed_turn_and_reports_empty_when_exhausted(self) -> None:
        app_file = self.repo / "app.py"
        new_file = self.repo / "extra.txt"
        provider = FakeProvider(
            [
                ModelResponse(
                    content="writing",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="write_file",
                            args={"path": str(app_file), "content": "VALUE = 2\n"},
                        ),
                        ToolCall(
                            id="call_2",
                            name="write_file",
                            args={"path": str(new_file), "content": "hello\n"},
                        ),
                    ],
                ),
                ModelResponse(content="Done."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            checkpointing=True,
            workspace_dir=self.repo,
        )

        reply = asyncio.run(agent.run("apply changes"))
        self.assertEqual(reply, "Done.")
        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 2\n")
        self.assertTrue(new_file.exists())

        undo_res = agent.undo_last_turn()
        self.assertTrue(undo_res.ok)
        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 1\n")
        self.assertFalse(new_file.exists())

        empty_res = agent.undo_last_turn()
        self.assertFalse(empty_res.ok)
        self.assertFalse(empty_res.active)

    def test_limit_stop_preserves_partial_work_without_auto_rollback(self) -> None:
        app_file = self.repo / "app.py"
        provider = FakeProvider(
            [
                ModelResponse(
                    content="step 1",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="write_file",
                            args={"path": str(app_file), "content": "VALUE = 42\n"},
                        )
                    ],
                ),
                ModelResponse(content="unreachable"),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            max_turns=1,
            checkpointing=True,
            workspace_dir=self.repo,
        )

        with self.assertRaises(AgentLimitError) as ctx:
            asyncio.run(agent.run("edit and stop on limit"))
        self.assertIn("max_turns", str(ctx.exception))
        self.assertIsNone(agent.last_rollback_message)
        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 42\n")

        undo_res = agent.undo_last_turn()
        self.assertTrue(undo_res.ok)
        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 1\n")

    def test_cli_parse_args_no_checkpoint_flag(self) -> None:
        default_args = main.parse_args([])
        self.assertTrue(default_args.checkpointing)

        disabled_args = main.parse_args(["--no-checkpoint"])
        self.assertFalse(disabled_args.checkpointing)

    def test_cli_parse_args_provider_and_model_flags(self) -> None:
        default_args = main.parse_args([])
        self.assertIsNone(default_args.provider)
        self.assertIsNone(default_args.model)

        custom_args = main.parse_args(["--provider", "ollama", "--model", "llama3.2"])
        self.assertEqual(custom_args.provider, "ollama")
        self.assertEqual(custom_args.model, "llama3.2")


if __name__ == "__main__":
    unittest.main()
