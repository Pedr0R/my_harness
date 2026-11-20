#!/usr/bin/env python3
"""Verification control harness for my_harness.

Drives the real interactive CLI (`python main.py`) in an isolated scratch
workspace, runs read-only health checks (`doctor`), verifies provider failover
across a real local HTTP boundary (`verify-fallback`), and cleans up scratch
state without deleting proof artifacts (`cleanup`).
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


PROMPT_MARKER = "Você: "
BANNER_MARKER = "Chat Interativo com SWE-Agent"
EXIT_COMMANDS = {"exit", "sair", "quit", "q"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def find_repo_root() -> pathlib.Path:
    """Resolve the repository root containing main.py and agent.py."""
    here = pathlib.Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "main.py").is_file() and (candidate / "agent.py").is_file():
            return candidate
    raise RuntimeError(f"Could not locate my_harness repo root from {here}")


def base_verify_dir() -> pathlib.Path:
    root = os.getenv("MY_HARNESS_VERIFY_ROOT")
    if root:
        return pathlib.Path(root).resolve()
    return pathlib.Path(tempfile.gettempdir()) / "my-harness-verify"


def validate_run_id(run_id: str) -> str:
    if not run_id or not RUN_ID_RE.match(run_id) or ".." in run_id:
        raise ValueError(
            f"Invalid run_id {run_id!r}. Use only letters, digits, '.', '_', '-'."
        )
    return run_id


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"


def get_git_info(repo_root: pathlib.Path) -> dict[str, str]:
    info: dict[str, str] = {"commit": "unknown", "branch": "unknown"}
    for key, args in [
        ("commit", ["git", "rev-parse", "--short", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ]:
        try:
            r = subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                info[key] = r.stdout.strip()
        except Exception:
            pass
    return info


def load_repo_env_without_printing(repo_root: pathlib.Path) -> tuple[dict[str, str], dict[str, bool]]:
    """Build a child process environment inheriting current env plus repo .env values."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    dotenv_path = repo_root / ".env"
    dotenv_keys_present: dict[str, bool] = {}
    if dotenv_path.is_file():
        try:
            from dotenv import dotenv_values

            parsed = dotenv_values(dotenv_path)
            for k, v in parsed.items():
                if v is not None:
                    dotenv_keys_present[k] = bool(v.strip())
                    if k not in env or not env[k]:
                        env[k] = v
        except Exception:
            pass

    auth_summary = {
        "GEMINI_API_KEY": bool(env.get("GEMINI_API_KEY", "").strip()),
        "GEMINI_PROJECT_ID": bool(env.get("GEMINI_PROJECT_ID", "").strip()),
        "OPENAI_API_KEY": bool(env.get("OPENAI_API_KEY", "").strip()),
        "GROK_API_KEY": bool(env.get("GROK_API_KEY", "").strip()),
        "DEEPSEEK_API_KEY": bool(env.get("DEEPSEEK_API_KEY", "").strip()),
    }
    return env, auth_summary


class IncrementalStreamBuffer:
    """Thread-safe incremental UTF-8 reader for subprocess stdout/stderr."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._cond = threading.Condition()
        self._text = ""
        self._eof = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = self._stream.read(1024)
                if not chunk:
                    tail = decoder.decode(b"", final=True)
                    with self._cond:
                        if tail:
                            self._text += tail
                        self._eof = True
                        self._cond.notify_all()
                    break
                decoded = decoder.decode(chunk, final=False)
                if decoded:
                    with self._cond:
                        self._text += decoded
                        self._cond.notify_all()
        except Exception:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def wait_for_marker(self, marker: str, start_offset: int, timeout: float) -> tuple[bool, int, str]:
        """Wait until marker appears at/after start_offset or EOF is reached."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                idx = self._text.find(marker, start_offset)
                if idx != -1:
                    end = idx + len(marker)
                    return True, end, self._text[start_offset:end]
                if self._eof:
                    return False, len(self._text), self._text[start_offset:]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, len(self._text), self._text[start_offset:]
                self._cond.wait(timeout=min(remaining, 0.25))

    def wait_for_eof(self, start_offset: int, timeout: float) -> tuple[bool, int, str]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._eof:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, len(self._text), self._text[start_offset:]
                self._cond.wait(timeout=min(remaining, 0.25))
            return True, len(self._text), self._text[start_offset:]

    def full_text(self) -> str:
        with self._cond:
            return self._text


def snapshot_workspace(workspace_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Capture a lightweight manifest of all files inside the scratch workspace."""
    manifest: dict[str, dict[str, Any]] = {}
    if not workspace_dir.exists():
        return manifest
    for p in sorted(workspace_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(workspace_dir).as_posix()
        raw = p.read_bytes()
        entry: dict[str, Any] = {
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if len(raw) <= 16_384:
            entry["content"] = raw.decode("utf-8", errors="replace")
        else:
            entry["content"] = raw[:16_384].decode("utf-8", errors="replace") + "\n[truncated]"
        manifest[rel] = entry
    return manifest


def diff_snapshots(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    before_keys = set(before.keys())
    after_keys = set(after.keys())
    created = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified = sorted(
        k for k in (before_keys & after_keys) if before[k]["sha256"] != after[k]["sha256"]
    )
    return {"created": created, "modified": modified, "deleted": deleted}


def parse_seed_files(seed_args: list[str], workspace_dir: pathlib.Path) -> list[str]:
    seeded: list[str] = []
    for item in seed_args:
        if "=" not in item:
            raise ValueError(f"--seed-file expects RELPATH=CONTENT, got: {item!r}")
        rel_str, content = item.split("=", 1)
        rel_path = pathlib.Path(rel_str)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Seed file path must stay inside workspace: {rel_str!r}")
        target = workspace_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Support literal \n escapes for convenience in shell commands
        target.write_text(content.replace("\\n", "\n"), encoding="utf-8")
        seeded.append(rel_path.as_posix())
    return seeded


def terminate_exact_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def pid_is_my_harness(pid: int) -> bool:
    """Confirm that a still-running PID actually belongs to a python main.py process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            r = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            cmdline = (r.stdout or "").strip().lower()
            return bool(cmdline) and "main.py" in cmdline
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        r = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "main.py" in (r.stdout or "").lower()
    except Exception:
        return False


def kill_recorded_pid(pid: int) -> str:
    if not pid_is_my_harness(pid):
        return "not-running-or-not-owned"
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "killed"
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        if pid_is_my_harness(pid):
            os.kill(pid, signal.SIGKILL)
        return "killed"
    except OSError:
        return "already-exited"


def run_interactive_session(
    repo_root: pathlib.Path,
    workspace_dir: pathlib.Path,
    pid_file: pathlib.Path,
    inputs: list[str],
    startup_timeout: float,
    turn_timeout: float,
    auto_exit: bool,
) -> dict[str, Any]:
    env, _ = load_repo_env_without_printing(repo_root)
    cmd = [sys.executable, "-u", str(repo_root / "main.py")]

    workspace_dir.mkdir(parents=True, exist_ok=True)
    before_snapshot = snapshot_workspace(workspace_dir)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace_dir),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "cmd": cmd,
                "workspace_dir": str(workspace_dir),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_buf = IncrementalStreamBuffer(proc.stdout)
    stderr_buf = IncrementalStreamBuffer(proc.stderr)

    turns_log: list[dict[str, Any]] = []
    startup_output = ""
    ready = False
    error: str | None = None
    cursor = 0

    try:
        found, cursor, startup_output = stdout_buf.wait_for_marker(
            PROMPT_MARKER, start_offset=0, timeout=startup_timeout
        )
        ready = found and (BANNER_MARKER in startup_output)
        if not found:
            error = f"Timed out or exited before initial prompt {PROMPT_MARKER!r}"
        else:
            effective_inputs = list(inputs)
            if auto_exit and (
                not effective_inputs
                or (
                    effective_inputs[-1] != "__EOF__"
                    and effective_inputs[-1].strip().lower() not in EXIT_COMMANDS
                )
            ):
                effective_inputs.append("exit")

            for idx, user_text in enumerate(effective_inputs, start=1):
                turn_start = time.monotonic()
                if user_text == "__EOF__":
                    proc.stdin.close()
                    exited, cursor, chunk = stdout_buf.wait_for_eof(cursor, turn_timeout)
                    turns_log.append(
                        {
                            "turn": idx,
                            "input": "__EOF__",
                            "output": chunk,
                            "duration_s": round(time.monotonic() - turn_start, 3),
                            "completed": exited,
                        }
                    )
                    if not exited:
                        error = f"Turn {idx} (__EOF__) timed out waiting for process exit"
                    break

                try:
                    proc.stdin.write((user_text + "\n").encode("utf-8"))
                    proc.stdin.flush()
                except BrokenPipeError:
                    error = f"Turn {idx} failed: process stdin closed unexpectedly"
                    break

                if user_text.strip().lower() in EXIT_COMMANDS:
                    exited, cursor, chunk = stdout_buf.wait_for_eof(cursor, turn_timeout)
                    turns_log.append(
                        {
                            "turn": idx,
                            "input": user_text,
                            "output": chunk,
                            "duration_s": round(time.monotonic() - turn_start, 3),
                            "completed": exited,
                        }
                    )
                    if not exited:
                        error = f"Turn {idx} ({user_text!r}) timed out waiting for exit"
                    break

                got_prompt, cursor, chunk = stdout_buf.wait_for_marker(
                    PROMPT_MARKER, start_offset=cursor, timeout=turn_timeout
                )
                turns_log.append(
                    {
                        "turn": idx,
                        "input": user_text,
                        "output": chunk,
                        "duration_s": round(time.monotonic() - turn_start, 3),
                        "completed": got_prompt,
                    }
                )
                if not got_prompt:
                    error = (
                        f"Turn {idx} ({user_text!r}) did not return to prompt {PROMPT_MARKER!r} "
                        f"within {turn_timeout}s"
                    )
                    break

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            terminate_exact_process(proc)
            if not error:
                error = "Process did not exit cleanly; terminated by harness"
    finally:
        terminate_exact_process(proc)
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        if pid_file.exists():
            try:
                pid_file.unlink()
            except OSError:
                pass

    stderr_buf.wait_for_eof(0, timeout=2.0)
    after_snapshot = snapshot_workspace(workspace_dir)
    fs_diff = diff_snapshots(before_snapshot, after_snapshot)

    return {
        "command": cmd,
        "workspace_dir": str(workspace_dir),
        "ready": ready,
        "exit_code": proc.returncode,
        "duration_s": round(time.monotonic() - t0, 3),
        "error": error,
        "startup_output": startup_output,
        "turns": turns_log,
        "full_stdout": stdout_buf.full_text(),
        "full_stderr": stderr_buf.full_text(),
        "workspace_diff": fs_diff,
        "workspace_files_after": after_snapshot,
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    repo_root = find_repo_root()
    git_info = get_git_info(repo_root)
    _, auth_summary = load_repo_env_without_printing(repo_root)

    checks: dict[str, Any] = {
        "repo_root": str(repo_root),
        "git": git_info,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "required_files": {
            rel: (repo_root / rel).is_file()
            for rel in [
                "main.py",
                "agent.py",
                "providers/__init__.py",
                "providers/base.py",
                "providers/factory.py",
                "providers/fallback.py",
                "providers/gemini.py",
                "providers/openai_compat.py",
            ]
        },
        "dependencies": {
            pkg: importlib.util.find_spec(mod) is not None
            for pkg, mod in [
                ("google-genai", "google.genai"),
                ("httpx", "httpx"),
                ("python-dotenv", "dotenv"),
            ]
        },
        "auth_configured": auth_summary,
        "gemini_auth_ready": auth_summary["GEMINI_API_KEY"] or auth_summary["GEMINI_PROJECT_ID"],
    }

    # Boot check in an ephemeral temp dir that is immediately cleaned up (zero API call unless --ping)
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="my-harness-doctor-"))
    try:
        inputs = ["Responda apenas OK", "exit"] if args.ping else ["exit"]
        boot = run_interactive_session(
            repo_root=repo_root,
            workspace_dir=tmp_dir / "ws",
            pid_file=tmp_dir / "session.json",
            inputs=inputs,
            startup_timeout=args.timeout,
            turn_timeout=args.timeout,
            auto_exit=True,
        )
        checks["boot_check"] = {
            "mode": "live-ping" if args.ping else "cold-boot-and-exit",
            "ready": boot["ready"],
            "exit_code": boot["exit_code"],
            "duration_s": boot["duration_s"],
            "error": boot["error"],
            "stderr_empty": not bool(boot["full_stderr"].strip()),
        }
        if args.ping and boot["turns"]:
            checks["boot_check"]["ping_output"] = boot["turns"][0]["output"].strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    healthy = (
        all(checks["required_files"].values())
        and all(checks["dependencies"].values())
        and checks["gemini_auth_ready"]
        and checks["boot_check"]["ready"]
        and checks["boot_check"]["exit_code"] == 0
        and checks["boot_check"]["error"] is None
    )
    checks["healthy"] = healthy
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if healthy else 1


def format_transcript(
    feature: str,
    entry_point: str,
    run_id: str,
    git_info: dict[str, str],
    result: dict[str, Any],
) -> str:
    lines = [
        f"Feature: {feature}",
        f"Entry point: {entry_point}",
        f"Run ID: {run_id}",
        f"Git: {git_info['branch']} @ {git_info['commit']}",
        f"Workspace: {result['workspace_dir']}",
        f"Command: {' '.join(result['command'])}",
        f"Exit code: {result['exit_code']} (duration: {result['duration_s']}s)",
        "=" * 72,
        "--- STARTUP ---",
        result["startup_output"],
    ]
    for turn in result["turns"]:
        lines.append(f"\n--- TURN {turn['turn']} INPUT ({turn['duration_s']}s) ---")
        lines.append(f">>> {turn['input']}")
        lines.append("--- OUTPUT ---")
        lines.append(turn["output"])
    if result["full_stderr"].strip():
        lines.append("\n--- STDERR ---")
        lines.append(result["full_stderr"])
    lines.append("\n--- WORKSPACE DIFF ---")
    lines.append(json.dumps(result["workspace_diff"], indent=2, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def cmd_drive(args: argparse.Namespace) -> int:
    repo_root = find_repo_root()
    run_id = validate_run_id(args.run_id or default_run_id())
    feature = validate_run_id(args.feature)

    scratch_run_dir = base_verify_dir() / "scratch" / run_id
    workspace_dir = (
        pathlib.Path(args.workspace_dir).resolve()
        if args.workspace_dir
        else scratch_run_dir / "workspace"
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    seeded = parse_seed_files(args.seed_file or [], workspace_dir)

    evidence_dir = (
        pathlib.Path(args.evidence_dir).resolve()
        if args.evidence_dir
        else base_verify_dir() / "evidence" / run_id / feature
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info(repo_root)
    result = run_interactive_session(
        repo_root=repo_root,
        workspace_dir=workspace_dir,
        pid_file=scratch_run_dir / "session.json",
        inputs=args.input or [],
        startup_timeout=args.startup_timeout,
        turn_timeout=args.turn_timeout,
        auto_exit=not args.no_auto_exit,
    )

    report = {
        "feature": feature,
        "entry_point": args.entry_point,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "seeded_files": seeded,
        **result,
    }

    report_path = evidence_dir / "report.json"
    transcript_path = evidence_dir / "transcript.txt"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    transcript_path.write_text(
        format_transcript(feature, args.entry_point, run_id, git_info, result),
        encoding="utf-8",
    )

    summary = {
        "ok": result["ready"] and result["exit_code"] == 0 and result["error"] is None,
        "run_id": run_id,
        "feature": feature,
        "entry_point": args.entry_point,
        "workspace_dir": str(workspace_dir),
        "evidence_dir": str(evidence_dir),
        "report_path": str(report_path),
        "transcript_path": str(transcript_path),
        "exit_code": result["exit_code"],
        "error": result["error"],
        "turns_completed": len(result["turns"]),
        "workspace_diff": result["workspace_diff"],
        "stderr_nonempty": bool(result["full_stderr"].strip()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 1


def cmd_verify_fallback(args: argparse.Namespace) -> int:
    """Exercise get_provider + OpenAICompatibleProvider + FallbackProvider over real local HTTP."""
    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from agent import Agent
    from providers import get_provider

    run_id = validate_run_id(args.run_id or default_run_id())
    feature = validate_run_id(args.feature)
    evidence_dir = (
        pathlib.Path(args.evidence_dir).resolve()
        if args.evidence_dir
        else base_verify_dir() / "evidence" / run_id / feature
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    git_info = get_git_info(repo_root)

    observed_requests: list[dict[str, Any]] = []
    lock = threading.Lock()

    class LocalOpenAIStubHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *msg_args: Any) -> None:
            return  # keep stderr clean

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception:
                body = {"raw": raw_body.decode("utf-8", errors="replace")}

            with lock:
                observed_requests.append(
                    {
                        "path": self.path,
                        "model": body.get("model"),
                        "message_count": len(body.get("messages", [])),
                        "tool_names": [
                            t.get("function", {}).get("name") for t in body.get("tools", [])
                        ],
                    }
                )

            if self.path == "/failing/v1/chat/completions":
                payload = json.dumps(
                    {"error": {"message": "Simulated 429 rate limit on primary provider"}}
                ).encode("utf-8")
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path == "/healthy/v1/chat/completions":
                payload = json.dumps(
                    {
                        "id": "chatcmpl-verify-fallback",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Fallback response from local OpenAI-compatible server.",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 20},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalOpenAIStubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        primary = get_provider(
            "openai",
            model="primary-failing-model",
            api_key="verify-local-key",
            base_url=f"http://127.0.0.1:{port}/failing/v1",
            timeout=10.0,
        )
        secondary = get_provider(
            "openai",
            model="secondary-healthy-model",
            api_key="verify-local-key",
            base_url=f"http://127.0.0.1:{port}/healthy/v1",
            timeout=10.0,
        )
        fallback = get_provider("fallback", providers=[primary, secondary])
        agent = Agent(
            name="Verify-Fallback-Agent",
            model=fallback,
            instructions="Verify fallback routing.",
        )
        response_text = asyncio.run(agent.run(args.prompt))
        memory = agent.get_session_memory()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    ok = (
        len(observed_requests) == 2
        and observed_requests[0]["path"] == "/failing/v1/chat/completions"
        and observed_requests[1]["path"] == "/healthy/v1/chat/completions"
        and "Fallback response" in response_text
        and agent.n_turns == 1
        and agent.cost > 0
    )

    report = {
        "feature": feature,
        "entry_point": "providers.get_provider('fallback', providers=[...]) + Agent.run()",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "boundary": f"ephemeral local HTTP server 127.0.0.1:{port} (stopped after run)",
        "prompt": args.prompt,
        "response": response_text,
        "agent_turns": agent.n_turns,
        "agent_cost": agent.cost,
        "observed_http_requests": observed_requests,
        "session_memory_message_count": len(memory["messages"]),
        "ok": ok,
    }
    report_path = evidence_dir / "report.json"
    transcript_path = evidence_dir / "transcript.txt"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    transcript_path.write_text(
        "\n".join(
            [
                f"Feature: {feature}",
                f"Run ID: {run_id}",
                f"Git: {git_info['branch']} @ {git_info['commit']}",
                f"Prompt: {args.prompt}",
                f"HTTP sequence: {json.dumps(observed_requests, ensure_ascii=False)}",
                f"Agent response: {response_text}",
                f"Turns: {agent.n_turns}, Estimated cost: ${agent.cost:.6f}",
                f"OK: {ok}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": ok,
                "run_id": run_id,
                "feature": feature,
                "evidence_dir": str(evidence_dir),
                "report_path": str(report_path),
                "transcript_path": str(transcript_path),
                "observed_http_requests": observed_requests,
                "response": response_text,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    run_id = validate_run_id(args.run_id)
    scratch_run_dir = base_verify_dir() / "scratch" / run_id
    evidence_run_dir = base_verify_dir() / "evidence" / run_id

    pid_action = "none"
    pid_file = scratch_run_dir / "session.json"
    if pid_file.is_file():
        try:
            data = json.loads(pid_file.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
            pid_action = kill_recorded_pid(pid)
        except Exception as exc:
            pid_action = f"error:{exc}"

    scratch_removed = False
    if scratch_run_dir.exists():
        shutil.rmtree(scratch_run_dir)
        scratch_removed = True

    evidence_files = (
        sorted(str(p.relative_to(evidence_run_dir)) for p in evidence_run_dir.rglob("*") if p.is_file())
        if evidence_run_dir.exists()
        else []
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "pid_action": pid_action,
                "scratch_dir": str(scratch_run_dir),
                "scratch_removed": scratch_removed,
                "evidence_dir": str(evidence_run_dir),
                "evidence_preserved": evidence_run_dir.exists(),
                "evidence_file_count": len(evidence_files),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verification harness for my_harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Read-only health check for my_harness")
    p_doc.add_argument("--timeout", type=float, default=30.0, help="Boot timeout in seconds")
    p_doc.add_argument(
        "--ping",
        action="store_true",
        help="Send one minimal live Gemini prompt to verify API auth end to end",
    )
    p_doc.set_defaults(func=cmd_doctor)

    p_drive = sub.add_parser("drive", help="Drive python main.py in an isolated workspace")
    p_drive.add_argument("--feature", required=True, help="Feature ID (e.g. chat-session)")
    p_drive.add_argument(
        "--entry-point",
        default="python main.py",
        help="User entry point label recorded in proof artifacts",
    )
    p_drive.add_argument("--run-id", help="Isolation run ID (defaults to timestamp + PID)")
    p_drive.add_argument(
        "--input",
        action="append",
        default=[],
        help="User line to send at each 'Você: ' prompt (repeatable; use '__EOF__' to close stdin)",
    )
    p_drive.add_argument(
        "--seed-file",
        action="append",
        default=[],
        help="Seed a file in the scratch workspace before launch as RELPATH=CONTENT (repeatable)",
    )
    p_drive.add_argument("--workspace-dir", help="Override scratch workspace path")
    p_drive.add_argument("--evidence-dir", help="Override evidence output directory")
    p_drive.add_argument("--startup-timeout", type=float, default=35.0)
    p_drive.add_argument("--turn-timeout", type=float, default=60.0)
    p_drive.add_argument(
        "--no-auto-exit",
        action="store_true",
        help="Do not append 'exit' automatically after the last --input",
    )
    p_drive.set_defaults(func=cmd_drive)

    p_fb = sub.add_parser(
        "verify-fallback",
        help="Verify get_provider + OpenAICompatibleProvider + FallbackProvider over local HTTP",
    )
    p_fb.add_argument("--feature", default="provider-fallback")
    p_fb.add_argument("--run-id", help="Isolation run ID")
    p_fb.add_argument("--evidence-dir", help="Override evidence output directory")
    p_fb.add_argument(
        "--prompt",
        default="Confirm fallback routing succeeded.",
        help="Prompt sent through Agent.run()",
    )
    p_fb.set_defaults(func=cmd_verify_fallback)

    p_clean = sub.add_parser(
        "cleanup", help="Remove scratch workspace and any stranded PID for a run (keeps evidence)"
    )
    p_clean.add_argument("--run-id", required=True, help="Run ID to clean up")
    p_clean.set_defaults(func=cmd_cleanup)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
