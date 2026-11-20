---
name: verify-my-harness
description: "Drive and verify my_harness end to end: launch the interactive Python terminal chat CLI (python main.py) in an isolated scratch workspace, exercise built-in REPL commands (memory, exit/sair/quit/q, EOF), live agent file/shell tools (write_file, read_file, search_replace, bash), and provider fallback routing, capturing durable transcripts and disk-state proof artifacts."
---

# Verify `my_harness`

`my_harness` is a Python SWE-agent harness whose primary user-facing surface is an interactive terminal chat REPL (`python main.py`, backed by `agent.py` and `providers/gemini.py` using `gemini-3.5-flash-lite`) and whose secondary surface is its pluggable provider Python API (`providers/`: `get_provider`, `GeminiProvider`, `OpenAICompatibleProvider`, `FallbackProvider`).

Read [`features/README.md`](./features/README.md) first to select the mapped feature recipe(s) before driving the app.

## Launch

`my_harness` is a short-lived interactive CLI (no background server, database, or listening port). Launch means verifying the Python runtime and dependencies once, then spawning a fresh, isolated `python -u main.py` subprocess per verification drive with `cwd` pointing to a per-run disposable scratch directory.

1. **Dependencies (one-time per environment):** Python 3.13+ with `google-genai`, `httpx`, and `python-dotenv` installed (`pip install google-genai httpx python-dotenv`), plus valid Gemini credentials (`GEMINI_API_KEY` or `GEMINI_PROJECT_ID` in the environment or in the repo-root `.env`).
2. **Isolated launch per drive:** Use the helper script so `main.py` starts in `%TEMP%\my-harness-verify\scratch\<run-id>\workspace` (never in the repository root, because the agent's `write_file`, `search_replace`, and `bash` tools operate relative to `cwd`) while loading repo-root `.env` credentials into the child process environment with `PYTHONUNBUFFERED=1` and `PYTHONUTF8=1`:

```powershell
python .agents/skills/verify-my-harness/scripts/control_harness.py drive `
  --run-id verify-001 `
  --feature chat-session `
  --input "Responda apenas: OK" `
  --input "memory" `
  --input "exit"
```

3. **Ready signal:** `main.py` is ready for input only when stdout emits both the startup banner `🤖 Chat Interativo com SWE-Agent` and the prompt `Você: `. Note that cold import of `google-genai` on Windows can take 8–15 seconds; the harness defaults to `--startup-timeout 35`.
4. **Isolation:** Multiple runs execute safely side by side when each uses a distinct `--run-id` (separate scratch workspace and evidence folder). No local ports or shared files are locked, though concurrent live Gemini runs share the API key's rate limits and token quota.
5. **Teardown:** Every drive finishes by sending `exit` (or `sair`, `quit`, `q`, or `__EOF__`), confirming exit code `0` and the `Sessão finalizada:` footer, and removing only the run's scratch directory via `cleanup`.

## Doctor

Run this read-only health check before your first drive and after any unexpected failure:

```powershell
python .agents/skills/verify-my-harness/scripts/control_harness.py doctor
```

What `doctor` verifies without mutating the repo or spending API tokens:
- Python version and required packages (`google-genai`, `httpx`, `python-dotenv`).
- Presence of core source files (`main.py`, `agent.py`, `providers/*.py`) and current git branch/commit.
- Auth credential presence (`GEMINI_API_KEY` or `GEMINI_PROJECT_ID` in env or repo `.env`, reported as booleans only — secrets are never printed).
- Cold boot and clean `exit` of `python -u main.py` in a temporary directory (verifies `GeminiProvider` + `Agent` initialization, ready banner, `Você: ` prompt, exit code `0`, empty stderr, then deletes its temp dir).

To also prove live Gemini API auth and reachability with one minimal turn (incurs a tiny live API call), run:

```powershell
python .agents/skills/verify-my-harness/scripts/control_harness.py doctor --ping
```

If `"healthy": false`, fix the reported issue before driving features.

## Drive

Always drive through [`scripts/control_harness.py`](./scripts/control_harness.py) using literal prompt strings and real REPL inputs (never internal setters or monkey-patched `Agent` methods):

- **REPL prompt handle:** The harness synchronizes every turn on the exact stdout prompt `Você: `, sends one UTF-8 line, and waits until the next `Você: ` prompt (or process exit for `exit`/`sair`/`quit`/`q`/`__EOF__`).
- **Built-in REPL commands:**
  - `memory` — prints `[Memória da Sessão Atual]:`, `Total de turnos internos:`, `Custo estimado:`, `Total de mensagens gravadas:`, and timestamped `USER:` / `AGENT:` lines.
  - `exit`, `sair`, `quit`, `q` — prints `Encerrando conversa...` followed by `Sessão finalizada: session_...` and `Interações guardadas na memória: N`.
  - `__EOF__` (closes stdin) — triggers `EOFError`, prints `Encerrando sessão via interrupção...`, and still prints the `Sessão finalizada:` summary.
  - `""` (empty input) — ignored cleanly; returns directly to `Você: ` without invoking the model.
- **Seeding files in the scratch workspace:** Pass `--seed-file "rel/path.txt=line 1\nline 2"` (repeatable) before driving `file-tools` or `bash-tool`.
- **Provider fallback over a real HTTP boundary:** Drive `get_provider` + `OpenAICompatibleProvider` + `FallbackProvider` against an ephemeral local HTTP server (`127.0.0.1:0`) where the primary endpoint returns HTTP `429` and the secondary returns HTTP `200`:

```powershell
python .agents/skills/verify-my-harness/scripts/control_harness.py verify-fallback `
  --run-id verify-001 `
  --feature provider-fallback
```

See the per-feature recipes in [`features/`](./features/README.md) for exact commands and expected observable states.

## Evidence

Proof artifacts are written outside the scratch directory so cleanup never destroys them:

- **Default location:** `%TEMP%\my-harness-verify\evidence\<run-id>\<feature>\` (override base root with `MY_HARNESS_VERIFY_ROOT` or per-call with `--evidence-dir`).
- **Files captured per run:**
  - `transcript.txt` — chronological record containing feature ID, entry point, run ID, git branch/commit, workspace path, exact command, startup banner, every turn's input/output/duration, stderr, exit code, and workspace diff.
  - `report.json` — machine-readable proof containing the same metadata plus before/after file snapshots (`created`, `modified`, `deleted`, and file contents up to 16 KB per file) or observed HTTP requests for `verify-fallback`.

**Proof standards:**
1. Exercise the real user path (`python main.py` REPL for CLI/agent features; documented `providers` Python API for provider routing). Never bypass the REPL or call `tool_*` functions directly to claim a chat feature works.
2. Capture both the user action and the resulting state (e.g., the prompt sent, the agent reply, and the subsequent `memory` readout or final session summary).
3. Verify filesystem side effects alongside terminal output: when proving `write_file`, `search_replace`, or `bash`, inspect `workspace_diff` and `workspace_files_after` in `report.json` (not just the agent's claim in chat).
4. Mock only at a production network boundary: `verify-fallback` uses a real local HTTP server (`127.0.0.1:0`) behind `OpenAICompatibleProvider`'s `base_url` over `httpx`. `main.py` has no dry-run mode — every non-command prompt in `drive` makes a real network request to Google Gemini API.

## Cleanup

Clean up every run's scratch workspace after driving (including after failed or timed-out attempts):

```powershell
python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-001
```

What `cleanup` guarantees:
- Checks `%TEMP%\my-harness-verify\scratch\<run-id>\session.json` for a recorded PID from an interrupted run. If that PID is still alive **and** its process command line contains `main.py`, terminates only that exact PID (never kills by process name such as `python.exe`).
- Deletes `%TEMP%\my-harness-verify\scratch\<run-id>\`.
- Never deletes `%TEMP%\my-harness-verify\evidence\<run-id>\` — its JSON output explicitly reports `"evidence_preserved": true` and `"evidence_file_count"`.

After cleanup, confirm the proof files still exist at the reported `evidence_dir`.

## Helpers

- [`scripts/control_harness.py`](./scripts/control_harness.py) — standalone Python 3 CLI (stdlib + repo imports for `verify-fallback`):
  - `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor [--timeout 30] [--ping]`
  - `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --feature <id> [--run-id <id>] [--entry-point <label>] [--seed-file RELPATH=CONTENT]... [--input TEXT]... [--startup-timeout 35] [--turn-timeout 60] [--workspace-dir PATH] [--evidence-dir PATH] [--no-auto-exit]`
  - `python .agents/skills/verify-my-harness/scripts/control_harness.py verify-fallback [--run-id <id>] [--feature provider-fallback] [--prompt TEXT] [--evidence-dir PATH]`
  - `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id <id>`
