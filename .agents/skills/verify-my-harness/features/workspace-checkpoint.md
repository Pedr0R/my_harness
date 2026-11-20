# Workspace checkpointing and rollback

Workspace checkpointing automatically snapshots the Git workspace before each agent turn (without touching `git stash` or `.git/index`), automatically restores files if a turn fails mid-execution with an unhandled error or interruption, and lets a user manually revert the last completed turn with `/undo` or `/rollback`.

## Sub-features

- `checkpoint-auto-snapshot` captures an isolated snapshot before every turn in a Git repository when checkpointing is enabled (default in `python main.py`), preserving pre-existing unstaged edits, staged changes, and untracked files.
- `checkpoint-auto-rollback` restores modified or deleted files, removes files created during the aborted turn, and prints `[Rollback automático]: ...` when a turn crashes with an unhandled exception or interruption.
- `checkpoint-manual-undo` reverts the workspace changes made by the last completed turn via `/undo` or `/rollback` at the `Você: ` prompt, and reports `[Rollback indisponível]: ...` when no checkpoints remain.
- `checkpoint-opt-out` disables automatic snapshotting and rollback via `python main.py --no-checkpoint`.
- `checkpoint-safe-degradation` skips checkpointing cleanly in non-Git directories and never deletes session files inside `.sessions/`.

## How to get to it (user POV)

- Run `python main.py` inside a Git repository: each user turn is automatically checkpointed before execution.
- Type `/undo` or `/rollback` at the `Você: ` prompt to revert workspace edits from the last completed turn.
- Run `python main.py --no-checkpoint` to disable automatic checkpointing for the session.
- If an unhandled error occurs after the agent already edited files during a turn, observe `[Rollback automático]: ...` in the terminal output and the workspace restored to its pre-turn state.

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- Valid Gemini credentials are present for live REPL turns.
- Initialize a Git repository inside the isolated scratch workspace before driving live `/undo` flows (non-Git scratch directories degrade gracefully without active checkpoints).

- **Create a file in a live turn and revert it with `/undo`.** Seed or initialize a Git repo in the scratch workspace, ask the agent to create a file via `write_file`, then run `/undo` and exit. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-checkpoint --feature workspace-checkpoint-undo --entry-point "python main.py -> /undo" --input "Use a ferramenta write_file para criar temp_undo.txt com o conteúdo exato: rollback-me" --input "/undo" --input "exit"`
  The first turn creates `temp_undo.txt`, `/undo` prints `[Rollback]: Workspace restaurado para o checkpoint ...`, and `report.json` confirms `temp_undo.txt` is absent from `workspace_files_after` while `.sessions/<session_id>.json` remains intact.
- **Disable checkpointing with `--no-checkpoint`.** Launch the REPL with `--no-checkpoint`, attempt `/undo`, and exit. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-checkpoint-off --feature workspace-checkpoint-disabled --entry-point "python main.py --no-checkpoint" --cli-arg=--no-checkpoint --input "/undo" --input "exit"`
  Exit code is `0` and stdout prints `[Rollback indisponível]: Nenhum checkpoint disponível para desfazer.`.
- **Verify automatic rollback and Git edge cases via unit tests.** Run:
  `python -m pytest -v tests/test_workspace.py tests/test_agent.py`
  All tests pass, proving dirty-repo preservation, staged-index preservation, unborn `HEAD` handling, `.sessions/` exclusion, idempotent rollback, automatic rollback on mid-turn provider failure, and warm snapshot latency under 250ms on a 50-file repo.
- **Proof.** Inspect `transcript.txt` and `report.json` under `%TEMP%\my-harness-verify\evidence\verify-checkpoint\` and `%TEMP%\my-harness-verify\evidence\verify-checkpoint-off\`.

## Gotchas

- `control_harness.py` creates a plain (non-Git) scratch directory by default; checkpointing degrades gracefully there (`[Rollback indisponível]`) unless the workspace is a Git repository.
- Limit stops (`max_turns` / `max_cost`) intentionally preserve partial work instead of triggering automatic rollback; the user can still revert those changes manually with `/undo`.
- Running `/load <session_id>` or resetting the session clears the in-memory undo checkpoint stack.
- Clean up scratch state after verification with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-checkpoint`.
