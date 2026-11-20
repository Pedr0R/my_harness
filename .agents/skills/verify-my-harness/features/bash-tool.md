# Agent shell tool

Agent shell tool lets a user ask the SWE-Agent in the interactive chat to run an OS shell command (`bash`), observe stdout/stderr and exit code, and reflect the result in the conversation.

## Sub-features

- `bash-run` executes a shell command in the active workspace and returns stdout and exit code to the agent.
- `bash-side-effect` produces an observable workspace file or output from the executed command that can be independently verified on disk.
- `bash-nonzero-exit` surfaces a failing command's non-zero exit status without crashing the chat session.

## How to get to it (user POV)

- Run `python main.py` in a terminal and ask the agent at the `Você: ` prompt to run a shell command.

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- Valid Gemini credentials are present.

- **Run a shell command with observable output and file side effect.** Ask the agent to invoke the `bash` tool to run a portable Python one-liner that writes `shell_proof.txt` and prints a unique token, then exit cleanly. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-bash --feature bash-tool --entry-point "python main.py (bash tool)" --input "Use a ferramenta bash para executar exatamente este comando: python -c \"import pathlib; pathlib.Path('shell_proof.txt').write_text('BASH-OK-901', encoding='utf-8'); print('STDOUT-901')\" e me informe o stdout retornado." --input "memory" --input "exit"`
  The session exits with code `0`, `Agente:` reports `STDOUT-901`, `memory` shows `Total de turnos internos` >= `2` (one tool-call turn plus final response), and `report.json` shows `workspace_diff.created` containing `shell_proof.txt` with exact content `BASH-OK-901`.
- **Proof.** Inspect `%TEMP%\my-harness-verify\evidence\verify-bash\bash-tool\report.json` and `transcript.txt`. Confirm both the `STDOUT-901` mention in chat and `workspace_files_after["shell_proof.txt"]["content"] == "BASH-OK-901"`.

## Gotchas

- Despite the tool's name (`bash`), on Windows it executes through the default OS shell (`cmd.exe`), not GNU Bash. Use `python -c ...` or Windows-compatible commands in verification prompts so commands do not fail on Unix-only syntax (`cat`, `touch`, single-quoted bash strings).
- Tool commands time out after 120 seconds and output is truncated at 25,000 characters.
- Always run via `control_harness.py drive` in an isolated scratch workspace so shell side effects never touch repository files.
- Clean up scratch state after the run with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-bash`.
