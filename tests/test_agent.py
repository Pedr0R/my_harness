from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from google.genai import types

from agent import (
    Agent,
    AgentEvent,
    AgentLimitError,
    ApprovalAction,
    ApprovalDecision,
    EventType,
    discover_context,
)
from guardrails import CommandSafetyGuardrail, PathSandboxingGuardrail, SecretsRedactionGuardrail
from guardrails.base import BaseGuardrail, GuardrailResult
import main
from providers.base import ModelResponse, StreamChunk, ToolCall
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


class TestAgentHumanInTheLoop(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sync_callback_approving_write_file_writes_to_disk(self) -> None:
        target = self.root / "approved.txt"
        seen: list[tuple[str, dict[str, Any]]] = []

        def approve_cb(tool_name: str, args: dict[str, Any]) -> bool:
            seen.append((tool_name, args))
            return True

        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_w1",
                            name="write_file",
                            args={"path": str(target), "content": "approved_body"},
                        )
                    ]
                ),
                ModelResponse(content="Escrito com sucesso."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            auto_approve=False,
            approval_callback=approve_cb,
        )

        reply = asyncio.run(agent.run("write file"))
        self.assertEqual(reply, "Escrito com sucesso.")
        self.assertEqual(target.read_text(encoding="utf-8"), "approved_body")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "write_file")
        self.assertEqual(seen[0][1]["path"], str(target))

    def test_async_callback_rejecting_write_file_skips_disk_and_informs_model(self) -> None:
        target = self.root / "rejected.txt"

        async def reject_cb(_tool_name: str, _args: dict[str, Any]) -> ApprovalAction:
            return ApprovalAction.REJECT

        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_w_rej",
                            name="write_file",
                            args={"path": str(target), "content": "should_not_exist"},
                        )
                    ]
                ),
                ModelResponse(content="Entendido, não escrevi o arquivo."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            auto_approve=False,
            approval_callback=reject_cb,
        )

        reply = asyncio.run(agent.run("try write"))
        self.assertEqual(reply, "Entendido, não escrevi o arquivo.")
        self.assertFalse(target.exists())
        self.assertEqual(len(provider.calls), 2)

        tool_msg = provider.calls[1][0][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_w_rej")
        self.assertEqual(tool_msg["name"], "write_file")
        self.assertIn("recusada", tool_msg["content"].lower())
        self.assertIn("write_file", tool_msg["content"])

    def test_reject_decision_with_reason_surfaces_reason_and_skips_execution(self) -> None:
        target = self.root / "protected.txt"
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_reason",
                            name="write_file",
                            args={"path": str(target), "content": "x"},
                        )
                    ]
                ),
                ModelResponse(content="Plano ajustado."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            auto_approve=False,
            approval_callback=lambda _name, _args: ApprovalDecision(
                ApprovalAction.REJECT,
                reason="arquivo protegido",
            ),
        )

        reply = asyncio.run(agent.run("write protected"))
        self.assertEqual(reply, "Plano ajustado.")
        self.assertFalse(target.exists())
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("arquivo protegido", tool_msgs[0]["content"])

    def test_approve_all_approves_call_flips_auto_approve_and_skips_subsequent_prompts(self) -> None:
        file_one = self.root / "one.txt"
        file_two = self.root / "two.txt"
        calls: list[str] = []

        def approve_all_cb(tool_name: str, _args: dict[str, Any]) -> ApprovalDecision:
            calls.append(tool_name)
            return ApprovalDecision(ApprovalAction.APPROVE_ALL)

        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="write_file", args={"path": str(file_one), "content": "1"}),
                        ToolCall(id="c2", name="write_file", args={"path": str(file_two), "content": "2"}),
                    ]
                ),
                ModelResponse(content="Ambos gravados."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            auto_approve=False,
            approval_callback=approve_all_cb,
        )

        reply = asyncio.run(agent.run("write two files"))
        self.assertEqual(reply, "Ambos gravados.")
        self.assertTrue(agent.auto_approve)
        self.assertEqual(calls, ["write_file"])
        self.assertEqual(file_one.read_text(encoding="utf-8"), "1")
        self.assertEqual(file_two.read_text(encoding="utf-8"), "2")

    def test_read_only_tools_skip_approval_callback_when_manual(self) -> None:
        sample = self.root / "readme.txt"
        sample.write_text("hello_hitl\n", encoding="utf-8")
        callback_calls: list[str] = []

        def spy_cb(tool_name: str, _args: dict[str, Any]) -> bool:
            callback_calls.append(tool_name)
            return False

        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="c_read", name="read_file", args={"path": str(sample)}),
                        ToolCall(id="c_list", name="list_dir", args={"path": str(self.root)}),
                    ]
                ),
                ModelResponse(content="Leitura feita."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            auto_approve=False,
            approval_callback=spy_cb,
        )

        reply = asyncio.run(agent.run("inspect workspace"))
        self.assertEqual(reply, "Leitura feita.")
        self.assertEqual(callback_calls, [])
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertIn("hello_hitl", tool_msgs[0]["content"])
        self.assertIn("readme.txt", tool_msgs[1]["content"])

    def test_custom_require_approval_gates_custom_tool(self) -> None:
        executed: list[str] = []

        def custom_deploy(env: str) -> str:
            executed.append(env)
            return f"deployed:{env}"

        custom_schemas = [
            {
                "name": "deploy",
                "description": "Deploy service",
                "parameters": {
                    "type": "object",
                    "properties": {"env": {"type": "string"}},
                    "required": ["env"],
                },
            }
        ]
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=[ToolCall(id="cd1", name="deploy", args={"env": "prod"})]),
                ModelResponse(content="Deploy bloqueado pelo usuário."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            tools={"deploy": custom_deploy},
            tool_schemas=custom_schemas,
            auto_approve=False,
            require_approval={"deploy"},
            approval_callback=lambda _name, _args: False,
        )

        reply = asyncio.run(agent.run("deploy prod"))
        self.assertEqual(reply, "Deploy bloqueado pelo usuário.")
        self.assertEqual(executed, [])
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertIn("recusada", tool_msgs[0]["content"].lower())

    def test_fail_closed_when_manual_without_callback(self) -> None:
        target = self.root / "no_cb.txt"
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="c_nocb", name="write_file", args={"path": str(target), "content": "nope"})
                    ]
                ),
                ModelResponse(content="Sem callback."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            auto_approve=False,
            approval_callback=None,
        )

        reply = asyncio.run(agent.run("write without callback"))
        self.assertEqual(reply, "Sem callback.")
        self.assertFalse(target.exists())
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertIn("recusada", tool_msgs[0]["content"].lower())

    def test_guardrail_block_and_unknown_tool_short_circuit_before_approval_callback(self) -> None:
        callback_calls: list[str] = []

        def spy_cb(tool_name: str, _args: dict[str, Any]) -> bool:
            callback_calls.append(tool_name)
            return True

        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="c_env",
                            name="write_file",
                            args={"path": str(self.root / ".env"), "content": "SECRET=1"},
                        ),
                        ToolCall(id="c_bash", name="bash", args={"cmd": "git reset --hard"}),
                        ToolCall(id="c_unk", name="does_not_exist", args={}),
                    ]
                ),
                ModelResponse(content="Handled."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            guardrails=[
                PathSandboxingGuardrail(root_dir=self.root),
                CommandSafetyGuardrail(),
            ],
            auto_approve=False,
            require_approval={"write_file", "bash", "does_not_exist"},
            approval_callback=spy_cb,
        )

        reply = asyncio.run(agent.run("blocked calls"))
        self.assertEqual(reply, "Handled.")
        self.assertEqual(callback_calls, [])
        self.assertFalse((self.root / ".env").exists())

    def test_keyboard_interrupt_during_approval_triggers_workspace_rollback(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
        app_file = repo / "app.py"
        app_file.write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)

        decisions = [True]

        def interrupt_on_second(_tool_name: str, _args: dict[str, Any]) -> bool:
            if decisions:
                return decisions.pop(0)
            raise KeyboardInterrupt

        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="write_file", args={"path": str(app_file), "content": "VALUE = 2\n"}),
                        ToolCall(id="c2", name="bash", args={"cmd": "echo hi"}),
                    ]
                )
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            checkpointing=True,
            workspace_dir=repo,
            auto_approve=False,
            approval_callback=interrupt_on_second,
        )

        with self.assertRaises(KeyboardInterrupt):
            asyncio.run(agent.run("edit then interrupt"))

        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 1\n")
        self.assertIsNotNone(agent.last_rollback_message)

    def test_cli_parse_args_auto_approve_flags(self) -> None:
        self.assertFalse(main.parse_args([]).auto_approve)
        self.assertTrue(main.parse_args(["--auto-approve"]).auto_approve)
        self.assertTrue(main.parse_args(["-y"]).auto_approve)

    def test_format_approval_prompt_tools_and_truncation(self) -> None:
        bash_prompt = main.format_approval_prompt("bash", {"cmd": "line1\nline2\r\nline3"})
        self.assertEqual(bash_prompt, "Executar [bash: 'line1 line2 line3']? (s/n/a): ")

        long_bash = main.format_approval_prompt("bash", {"cmd": "x" * 30}, max_preview=15)
        self.assertEqual(long_bash, "Executar [bash: 'xxxxxxxxxxxx...']? (s/n/a): ")

        write_prompt = main.format_approval_prompt("write_file", {"path": "dir/a.py", "content": "hello"})
        self.assertEqual(write_prompt, "Executar [write_file: 'dir/a.py' (5 chars)]? (s/n/a): ")

        sr_prompt = main.format_approval_prompt(
            "search_replace",
            {"path": "dir/a.py", "search": "old", "replace": "new"},
        )
        self.assertEqual(sr_prompt, "Executar [search_replace: 'dir/a.py']? (s/n/a): ")

        custom_prompt = main.format_approval_prompt("deploy", {"env": "prod", "force": True})
        self.assertEqual(custom_prompt, "Executar [deploy: env=prod, force=True]? (s/n/a): ")

    def test_parse_approval_response_tokens(self) -> None:
        for token in ("s", "sim", "y", "yes", " S "):
            decision = main.parse_approval_response(token)
            self.assertEqual(decision, ApprovalDecision(ApprovalAction.APPROVE))

        for token in ("n", "nao", "não", "no", "", "   "):
            decision = main.parse_approval_response(token)
            self.assertEqual(decision, ApprovalDecision(ApprovalAction.REJECT))

        for token in ("a", "all", "sempre", "todos", " ALL "):
            decision = main.parse_approval_response(token)
            self.assertEqual(decision, ApprovalDecision(ApprovalAction.APPROVE_ALL))

        for token in ("maybe", "x", "123"):
            self.assertIsNone(main.parse_approval_response(token))

    def test_prompt_tool_approval_reprompts_and_handles_eof(self) -> None:
        buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=["invalid", "s"]), mock.patch("sys.stdout", buf):
            decision = asyncio.run(main._prompt_tool_approval("bash", {"cmd": "echo ok"}))
        self.assertEqual(decision, ApprovalDecision(ApprovalAction.APPROVE))
        self.assertIn("Resposta inválida", buf.getvalue())

        eof_buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=EOFError), mock.patch("sys.stdout", eof_buf):
            eof_decision = asyncio.run(main._prompt_tool_approval("bash", {"cmd": "echo ok"}))
        self.assertEqual(eof_decision, ApprovalDecision(ApprovalAction.REJECT))
        self.assertIn("EOF", eof_buf.getvalue())

    def test_handle_auto_approve_command_toggle_explicit_and_invalid(self) -> None:
        agent = Agent(model=FakeProvider([]), auto_approve=False)
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            main._handle_auto_approve_command("", agent)
            self.assertTrue(agent.auto_approve)

            main._handle_auto_approve_command("off", agent)
            self.assertFalse(agent.auto_approve)

            main._handle_auto_approve_command("on", agent)
            self.assertTrue(agent.auto_approve)

            main._handle_auto_approve_command("wat", agent)
            self.assertTrue(agent.auto_approve)

        out = buf.getvalue()
        self.assertIn("ATIVADO", out)
        self.assertIn("DESATIVADO", out)
        self.assertIn("/auto-approve [on|off]", out)


class _StreamingFakeProvider:
    def __init__(self, turns: list[list[StreamChunk]]) -> None:
        self._turns = list(turns)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        raise AssertionError("complete should not be called when stream is implemented")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(([dict(m) for m in messages], list(tools)))
        chunks = self._turns.pop(0) if self._turns else [StreamChunk(response=ModelResponse(content=""))]
        for chunk in chunks:
            yield chunk


class TestAgentStreaming(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_run_stream_emits_ordered_events_across_tool_calls_with_native_streaming_provider(self) -> None:
        target = self.root / "streamed.txt"
        tool_call = ToolCall(
            id="call_write_1",
            name="write_file",
            args={"path": str(target), "content": "streamed_content"},
        )
        provider = _StreamingFakeProvider(
            [
                [
                    StreamChunk(text_delta="Vou "),
                    StreamChunk(text_delta="criar."),
                    StreamChunk(
                        response=ModelResponse(
                            content="Vou criar.",
                            tool_calls=[tool_call],
                            cost=0.01,
                        )
                    ),
                ],
                [
                    StreamChunk(text_delta="Tudo "),
                    StreamChunk(text_delta="pronto!"),
                    StreamChunk(
                        response=ModelResponse(
                            content="Tudo pronto!",
                            cost=0.02,
                        )
                    ),
                ],
            ]
        )
        agent = Agent(model=provider, instructions="sys")

        async def collect() -> list[AgentEvent]:
            events: list[AgentEvent] = []
            async with contextlib.aclosing(agent.run_stream("create file")) as stream:
                async for ev in stream:
                    events.append(ev)
            return events

        events = asyncio.run(collect())
        self.assertEqual(
            [ev.type for ev in events],
            [
                EventType.TOKEN,
                EventType.TOKEN,
                EventType.TOOL_START,
                EventType.TOOL_END,
                EventType.TOKEN,
                EventType.TOKEN,
                EventType.FINAL,
            ],
        )
        self.assertEqual([ev.text for ev in events[:2]], ["Vou ", "criar."])
        self.assertEqual(events[2].tool_name, "write_file")
        self.assertEqual(events[2].tool_args, {"path": str(target), "content": "streamed_content"})
        self.assertEqual(events[2].tool_call_id, "call_write_1")
        self.assertEqual(events[3].tool_name, "write_file")
        self.assertEqual(events[3].tool_call_id, "call_write_1")
        self.assertIn("escreveu", events[3].tool_output or "")
        self.assertEqual([ev.text for ev in events[4:6]], ["Tudo ", "pronto!"])
        self.assertEqual(events[6].text, "Tudo pronto!")

        self.assertEqual(target.read_text(encoding="utf-8"), "streamed_content")
        self.assertEqual(agent.n_turns, 2)
        self.assertAlmostEqual(agent.cost, 0.03)

    def test_run_stream_falls_back_for_non_streaming_provider_and_matches_run_state(self) -> None:
        target_run = self.root / "run.txt"
        target_stream = self.root / "stream.txt"

        def build_responses(path: pathlib.Path) -> list[ModelResponse]:
            return [
                ModelResponse(
                    content="escrevendo",
                    tool_calls=[
                        ToolCall(id="c1", name="write_file", args={"path": str(path), "content": "ok"})
                    ],
                    cost=0.05,
                ),
                ModelResponse(content="concluido", cost=0.07),
            ]

        agent_run = Agent(model=FakeProvider(build_responses(target_run)), instructions="sys")
        agent_stream = Agent(model=FakeProvider(build_responses(target_stream)), instructions="sys")

        run_reply = asyncio.run(agent_run.run("write"))

        async def collect() -> list[AgentEvent]:
            async with contextlib.aclosing(agent_stream.run_stream("write")) as stream:
                return [ev async for ev in stream]

        stream_events = asyncio.run(collect())
        self.assertEqual(
            [ev.type for ev in stream_events],
            [
                EventType.TOKEN,
                EventType.TOOL_START,
                EventType.TOOL_END,
                EventType.TOKEN,
                EventType.FINAL,
            ],
        )
        self.assertEqual(run_reply, "concluido")
        self.assertEqual(stream_events[-1].text, run_reply)
        self.assertEqual(agent_stream.n_turns, agent_run.n_turns)
        self.assertAlmostEqual(agent_stream.cost, agent_run.cost)
        self.assertEqual(
            [
                {k: v for k, v in m.items() if k != "tool_calls"}
                for m in agent_stream.messages
                if m["role"] != "tool"
            ],
            [
                {k: v for k, v in m.items() if k != "tool_calls"}
                for m in agent_run.messages
                if m["role"] != "tool"
            ],
        )
        self.assertEqual(target_run.read_text(encoding="utf-8"), "ok")
        self.assertEqual(target_stream.read_text(encoding="utf-8"), "ok")

    def test_run_stream_preserves_hitl_rejection_between_tool_start_and_tool_end(self) -> None:
        target = self.root / "blocked.txt"
        provider = FakeProvider(
            [
                ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="c_rej", name="write_file", args={"path": str(target), "content": "x"})
                    ],
                ),
                ModelResponse(content="Cancelado conforme solicitado."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            auto_approve=False,
            approval_callback=lambda _name, _args: ApprovalDecision(
                ApprovalAction.REJECT,
                reason="bloqueado pelo usuário",
            ),
        )

        async def collect() -> list[AgentEvent]:
            async with contextlib.aclosing(agent.run_stream("try write")) as stream:
                return [ev async for ev in stream]

        events = asyncio.run(collect())
        self.assertEqual(
            [ev.type for ev in events],
            [EventType.TOOL_START, EventType.TOOL_END, EventType.TOKEN, EventType.FINAL],
        )
        self.assertFalse(target.exists())
        self.assertIn("bloqueado pelo usuário", events[1].tool_output or "")
        self.assertEqual(events[3].text, "Cancelado conforme solicitado.")

    def test_run_stream_rolls_back_workspace_on_early_consumer_close(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
        app_file = repo / "app.py"
        app_file.write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)

        provider = _StreamingFakeProvider(
            [
                [
                    StreamChunk(text_delta="Editando..."),
                    StreamChunk(
                        response=ModelResponse(
                            content="Editando...",
                            tool_calls=[
                                ToolCall(
                                    id="c1",
                                    name="write_file",
                                    args={"path": str(app_file), "content": "VALUE = 99\n"},
                                )
                            ],
                        )
                    ),
                ],
                [
                    StreamChunk(text_delta="Nunca exibido"),
                    StreamChunk(response=ModelResponse(content="Nunca exibido")),
                ],
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            checkpointing=True,
            workspace_dir=repo,
        )

        async def cancel_after_tool_end() -> None:
            async with contextlib.aclosing(agent.run_stream("edit and cancel")) as stream:
                async for ev in stream:
                    if ev.type is EventType.TOOL_END:
                        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 99\n")
                        break

        asyncio.run(cancel_after_tool_end())
        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 1\n")
        self.assertIsNotNone(agent.last_rollback_message)
        self.assertIn("Workspace restaurado", agent.last_rollback_message or "")


class TestTerminalStreamRenderer(unittest.TestCase):
    def test_cli_parse_args_no_stream_flag(self) -> None:
        self.assertTrue(main.parse_args([]).streaming)
        self.assertFalse(main.parse_args(["--no-stream"]).streaming)

    def test_renderer_streams_tokens_pauses_for_tool_and_resumes(self) -> None:
        class _TrackingBuffer(io.StringIO):
            def __init__(self) -> None:
                super().__init__()
                self.flush_count = 0

            def flush(self) -> None:
                self.flush_count += 1
                super().flush()

        buf = _TrackingBuffer()
        renderer = main.TerminalStreamRenderer(stream=buf)

        events = [
            AgentEvent(type=EventType.TOKEN, text="Vou "),
            AgentEvent(type=EventType.TOKEN, text="verificar."),
            AgentEvent(
                type=EventType.TOOL_START,
                tool_name="read_file",
                tool_args={"path": "app.py"},
                tool_call_id="c1",
            ),
            AgentEvent(
                type=EventType.TOOL_END,
                tool_name="read_file",
                tool_output="   1: VALUE = 1\n   2: VALUE = 2",
                tool_call_id="c1",
            ),
            AgentEvent(type=EventType.TOKEN, text="Tudo "),
            AgentEvent(type=EventType.TOKEN, text="certo!"),
            AgentEvent(type=EventType.FINAL, text="Tudo certo!"),
        ]
        for ev in events:
            renderer.handle(ev)

        output = buf.getvalue()
        self.assertEqual(renderer.final_text, "Tudo certo!")
        self.assertGreaterEqual(buf.flush_count, len(events))
        self.assertIn("\nAgente: Vou verificar.\n", output)
        self.assertIn("[Executando ferramenta]: read_file (app.py)\n", output)
        self.assertIn("[Ferramenta concluída]: read_file ->    1: VALUE = 1\n", output)
        self.assertIn("\nTudo certo!\n", output)

    def test_renderer_tool_first_then_tokens_prints_agente_prefix(self) -> None:
        buf = io.StringIO()
        renderer = main.TerminalStreamRenderer(stream=buf)

        for ev in [
            AgentEvent(
                type=EventType.TOOL_START,
                tool_name="bash",
                tool_args={"cmd": "echo hi"},
                tool_call_id="c1",
            ),
            AgentEvent(
                type=EventType.TOOL_END,
                tool_name="bash",
                tool_output="exit=0\nstdout:\nhi",
                tool_call_id="c1",
            ),
            AgentEvent(type=EventType.TOKEN, text="Pronto."),
            AgentEvent(type=EventType.FINAL, text="Pronto."),
        ]:
            renderer.handle(ev)

        output = buf.getvalue()
        self.assertIn("[Executando ferramenta]: bash (echo hi)\n", output)
        self.assertIn("[Ferramenta concluída]: bash -> exit=0\n", output)
        self.assertIn("\nAgente: Pronto.\n", output)

    def test_renderer_handles_final_without_preceding_tokens(self) -> None:
        buf = io.StringIO()
        renderer = main.TerminalStreamRenderer(stream=buf)

        renderer.handle(AgentEvent(type=EventType.FINAL, text="Resposta única."))

        self.assertEqual(renderer.final_text, "Resposta única.")
        self.assertEqual(buf.getvalue(), "\nAgente: Resposta única.\n")

    def test_renderer_finish_on_interrupt_closes_open_text_line_idempotently(self) -> None:
        buf = io.StringIO()
        renderer = main.TerminalStreamRenderer(stream=buf)

        renderer.handle(AgentEvent(type=EventType.TOKEN, text="Parcial"))
        renderer.finish_on_interrupt()
        renderer.finish_on_interrupt()

        self.assertEqual(buf.getvalue(), "\nAgente: Parcial\n")


class TestAsyncToolExecution(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_async_tool_executes_sanitizes_secrets_and_feeds_result_back_to_model(self) -> None:
        seen_inputs: list[str] = []

        async def async_fetch(query: str) -> str:
            await asyncio.sleep(0)
            seen_inputs.append(query)
            return "GEMINI_API_KEY=AIzaSyFakeAsyncSecretKey1234567890abcd\nstatus=ok"

        custom_schemas = [
            {
                "name": "async_fetch",
                "description": "Async fetch tool",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=[ToolCall(id="c_async", name="async_fetch", args={"query": "cfg"})]),
                ModelResponse(content="Segredo protegido."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            guardrails=[SecretsRedactionGuardrail()],
            tools={"async_fetch": async_fetch},
            tool_schemas=custom_schemas,
        )

        reply = asyncio.run(agent.run("fetch config"))
        self.assertEqual(reply, "Segredo protegido.")
        self.assertEqual(seen_inputs, ["cfg"])

        second_turn_messages = provider.calls[1][0]
        tool_msg = second_turn_messages[-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "c_async")
        self.assertIn("[REDACTED]", tool_msg["content"])
        self.assertIn("status=ok", tool_msg["content"])
        self.assertNotIn("AIzaSyFakeAsyncSecretKey", tool_msg["content"])

    def test_async_tool_exception_returns_error_message_without_crashing(self) -> None:
        async def failing_async() -> str:
            await asyncio.sleep(0)
            raise ValueError("async boom")

        custom_schemas = [
            {
                "name": "failing_async",
                "description": "Failing async tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=[ToolCall(id="c_fail", name="failing_async", args={})]),
                ModelResponse(content="Recuperado."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            tools={"failing_async": failing_async},
            tool_schemas=custom_schemas,
        )

        reply = asyncio.run(agent.run("trigger async error"))
        self.assertEqual(reply, "Recuperado.")
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["content"], "ERRO: ValueError: async boom")

    def test_async_tool_blocked_by_hitl_is_never_invoked(self) -> None:
        calls = 0

        async def guarded_async() -> str:
            nonlocal calls
            calls += 1
            return "executed"

        custom_schemas = [
            {
                "name": "guarded_async",
                "description": "Guarded async tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=[ToolCall(id="c_hitl", name="guarded_async", args={})]),
                ModelResponse(content="Recusado."),
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            tools={"guarded_async": guarded_async},
            tool_schemas=custom_schemas,
            auto_approve=False,
            require_approval={"guarded_async"},
            approval_callback=lambda _name, _args: False,
        )

        reply = asyncio.run(agent.run("run guarded"))
        self.assertEqual(reply, "Recusado.")
        self.assertEqual(calls, 0)
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertIn("recusada", tool_msgs[0]["content"].lower())

    def test_async_tool_cancellation_propagates_and_triggers_workspace_rollback(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
        app_file = repo / "app.py"
        app_file.write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)

        async def cancel_tool() -> str:
            await asyncio.sleep(0)
            raise asyncio.CancelledError

        from tools import TOOLS, TOOL_SCHEMAS

        tools = dict(TOOLS)
        tools["cancel_tool"] = cancel_tool
        schemas = list(TOOL_SCHEMAS) + [
            {
                "name": "cancel_tool",
                "description": "Cancels execution",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="write_file", args={"path": str(app_file), "content": "VALUE = 2\n"}),
                        ToolCall(id="c2", name="cancel_tool", args={}),
                    ]
                )
            ]
        )
        agent = Agent(
            model=provider,
            instructions="sys",
            tools=tools,
            tool_schemas=schemas,
            checkpointing=True,
            workspace_dir=repo,
        )

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(agent.run("edit then cancel"))

        self.assertEqual(app_file.read_text(encoding="utf-8"), "VALUE = 1\n")
        self.assertIsNotNone(agent.last_rollback_message)
        self.assertIn("Workspace restaurado", agent.last_rollback_message or "")

    def test_agent_runs_default_bash_tool_via_async_handler(self) -> None:
        cmd = f'"{sys.executable}" -c "print(\'default_bash_async_ok\')"'
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=[ToolCall(id="c_bash", name="bash", args={"cmd": cmd})]),
                ModelResponse(content="Concluído."),
            ]
        )
        agent = Agent(model=provider, instructions="sys")

        reply = asyncio.run(agent.run("run bash"))
        self.assertEqual(reply, "Concluído.")
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("default_bash_async_ok", tool_msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
