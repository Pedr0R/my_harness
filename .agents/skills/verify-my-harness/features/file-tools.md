# Agent file tools

Agent file tools let a user ask the SWE-Agent in the interactive chat to create files (`write_file`), inspect numbered file contents (`read_file`), and apply exact single-match text replacements (`search_replace`) inside the active workspace.

## Sub-features

- `file-write` creates a new file on disk (including parent directories) with the requested text.
- `file-search-replace` updates a unique string inside an existing file without altering surrounding lines.
- `file-read` reads an existing file and reports its contents back in chat.
- `file-search-replace-ambiguous` refuses to silently overwrite when the search text appears zero or multiple times and reports the error cleanly in chat.

## How to get to it (user POV)

- Run `python main.py` in a terminal and ask the agent at the `Você: ` prompt to create, read, or edit a file in the current directory.

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- Valid Gemini credentials are present.
- Seed a baseline file `notes/todo.txt` inside the isolated scratch workspace via `--seed-file`.

- **Write, edit, and read round-trip.** Seed `notes/todo.txt` with `status=draft\nowner=verify\n`, then drive three turns in one isolated session: create `hello.txt`, replace `status=draft` with `status=done` in `notes/todo.txt`, and read both files back. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-files --feature file-tools --entry-point "python main.py (write_file, search_replace, read_file)" --seed-file "notes/todo.txt=status=draft\nowner=verify\n" --input "Use a ferramenta write_file para criar o arquivo hello.txt com o conteúdo exato: linha um" --input "Use a ferramenta search_replace no arquivo notes/todo.txt para trocar status=draft por status=done." --input "Use a ferramenta read_file para ler notes/todo.txt e responda exatamente com o conteúdo lido." --input "memory" --input "exit"`
  The session exits with code `0`, `memory` shows `Total de turnos internos` greater than 3 (reflecting tool-call turns plus final replies), the third `Agente:` reply shows `status=done` and `owner=verify`, and `report.json` shows `workspace_diff.created` containing `hello.txt` (with content `linha um`) and `workspace_diff.modified` containing `notes/todo.txt` (with `status=done` and `owner=verify`).
- **Proof.** Inspect `%TEMP%\my-harness-verify\evidence\verify-files\file-tools\report.json` and `transcript.txt`. Confirm both the chat output and `workspace_files_after["hello.txt"]["content"]` / `workspace_files_after["notes/todo.txt"]["content"]` on disk.

## Gotchas

- Never trust the agent's chat reply alone as proof that a file was created or edited; always assert `workspace_diff` and `workspace_files_after` in `report.json`.
- Always run inside an isolated scratch workspace (`control_harness.py drive` does this by default) so file edits never touch repository files.
- When `read_file` is called with a nonzero `offset`, the returned line-number prefix still starts counting from `1` for the sliced window rather than `offset + 1`.
- `search_replace` requires the search string to occur **exactly once** in the file; duplicate or missing matches return an error message to the agent instead of modifying the file.
- Clean up scratch state after the run with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-files`.
