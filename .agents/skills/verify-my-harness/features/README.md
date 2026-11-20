# my_harness verification map

This directory is the maintained source for verifying the user-facing behavior of `my_harness`. Read this index before driving the app, then follow the matching feature file as the recipe.

## Baseline preconditions

- Run `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` and require `"healthy": true`.
- Choose a unique `--run-id` per verification run (for example `verify-20261119-01`) so concurrent runs isolate scratch workspaces under `%TEMP%\my-harness-verify\scratch\<run-id>\` and proof artifacts under `%TEMP%\my-harness-verify\evidence\<run-id>\`.
- Never launch `python main.py` directly in the repository root during verification; always let `control_harness.py drive` run it inside the per-run scratch workspace so tool calls cannot modify repository files.
- Ensure valid Gemini credentials (`GEMINI_API_KEY` or `GEMINI_PROJECT_ID`) are configured in the environment or repository `.env` before running any live chat feature.
- Run `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id <run-id>` after every run (including failed runs).

## Driving conventions

- Start every recipe from a fresh isolated workspace unless its preconditions seed specific files with `--seed-file`.
- Synchronize turns on the literal `Você: ` prompt via `control_harness.py drive --input ...` rather than piping raw input without prompt synchronization.
- Treat every command and quoted flag as literal.
- Clean up scratch workspaces after each run. Do not delete proof artifacts during cleanup.

## Proof and skip reporting

- Capture both the user action and the resulting state, not only a final status line.
- Every CLI drive produces `transcript.txt` and `report.json` containing the command, feature ID, entry point, git revision, stdout, stderr, exit code, turn timings, and workspace file diff.
- Every file or shell mutation proof must verify the resulting disk state in `report.json` (`workspace_diff` and `workspace_files_after`), not only the agent's text reply.
- Record the feature ID and entry point used with every artifact.
- Report an unreachable path with the attempted command and the unmet precondition (for example, missing `GROK_API_KEY` or local Ollama daemon not running).
- Do not report a skipped entry point as verified through a different path.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order:

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with control_harness.py` starts with `Preconditions:` and uses labeled bullets that pair each user action with an exact command and observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

Keep implementation details out of the map. Name only user paths, stable handles, required state, commands, and observable proof.

## Features

- [Interactive chat session](./chat-session.md) covers REPL launch, multi-turn context retention, blank-input handling, and exit paths (`exit`, `sair`, `quit`, `q`, `EOF`).
- [Session memory inspection](./session-memory.md) covers the `memory` command before and after live turns, including turn count, estimated cost, and logged history entries.
- [Session persistence and resumption](./session-persistence.md) covers automatic JSON persistence to `.sessions/<session_id>.json`, resuming via `--session <id>` and `/load <id>`, and listing sessions via `/sessions`.
- [Agent file tools](./file-tools.md) covers creating, reading, and editing workspace files through live chat prompts and verifying disk side effects.
- [Agent shell tool](./bash-tool.md) covers executing shell commands from live chat prompts and verifying both terminal output and workspace side effects.
- [Provider switching and fallback](./provider-fallback.md) covers selecting providers via `get_provider` and resilient HTTP failover via `FallbackProvider`.
