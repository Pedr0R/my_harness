# Agent file tools

Agent file tools let a user ask the SWE-Agent in the interactive chat to create files (`write_file`), inspect numbered file contents (`read_file`), apply exact single-match text replacements (`search_replace`), search text or regex across files (`grep`), list directory trees (`list_dir`), and locate files by glob pattern (`find_files`) inside the active workspace.

## Sub-features

- `file-write` creates a new file on disk (including parent directories) with the requested text.
- `file-search-replace` updates a unique string inside an existing file without altering surrounding lines.
- `file-read` reads an existing file and reports its contents back in chat.
- `file-search-replace-ambiguous` refuses to silently overwrite when the search text appears zero or multiple times and reports the error cleanly in chat.
- `file-grep` searches literal substrings or regular expressions in a single file or recursively across a directory with optional `file_pattern` filtering and `max_results` capping, skipping ignored directories, sensitive files, escaping symlinks, and binary files.
- `file-list-dir` lists directory entries hierarchically up to `max_depth` levels with `/` suffixes on subdirectories and sorted output, skipping ignored directories, sensitive files/dirs, and escaping symlinks/junctions.
- `file-find-files` walks a directory recursively and returns sorted POSIX relative paths matching a filename or relative-path glob pattern, skipping ignored directories, sensitive files/dirs, and escaping symlinks/junctions.

## How to get to it (user POV)

- Run `python main.py` in a terminal and ask the agent at the `Você: ` prompt to create, read, edit, list, find, or search files in the current directory.

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- Valid Gemini credentials are present.
- Seed a baseline file `notes/todo.txt` inside the isolated scratch workspace via `--seed-file`.

- **Write, edit, and read round-trip.** Seed `notes/todo.txt` with `status=draft\nowner=verify\n`, then drive three turns in one isolated session: create `hello.txt`, replace `status=draft` with `status=done` in `notes/todo.txt`, and read both files back. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-files --feature file-tools --entry-point "python main.py (write_file, search_replace, read_file)" --seed-file "notes/todo.txt=status=draft\nowner=verify\n" --input "Use a ferramenta write_file para criar o arquivo hello.txt com o conteúdo exato: linha um" --input "Use a ferramenta search_replace no arquivo notes/todo.txt para trocar status=draft por status=done." --input "Use a ferramenta read_file para ler notes/todo.txt e responda exatamente com o conteúdo lido." --input "memory" --input "exit"`
  The session exits with code `0`, `memory` shows `Total de turnos internos` greater than 3 (reflecting tool-call turns plus final replies), the third `Agente:` reply shows `status=done` and `owner=verify`, and `report.json` shows `workspace_diff.created` containing `hello.txt` (with content `linha um`) and `workspace_diff.modified` containing `notes/todo.txt` (with `status=done` and `owner=verify`).
- **Recursive grep with `file_pattern`.** Seed a Python file and a Markdown file containing the same token, then ask the agent to call `grep` restricted to `*.py`. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-grep --feature file-tools --entry-point "python main.py (grep)" --seed-file "src/app.py=MARKER = 42\n" --seed-file "notes.md=MARKER in docs\n" --input "Use a ferramenta grep para buscar MARKER apenas em arquivos *.py e informe o arquivo e a linha encontrados." --input "memory" --input "exit"`
  The session exits with code `0`, `memory` shows the `grep` tool call and `src/app.py:1: MARKER = 42` in tool output, and `notes.md` is excluded.
- **Directory listing and glob file discovery (`list_dir` and `find_files`).** Seed nested Python and Markdown files, then ask the agent to inspect the workspace tree with `list_dir` (`max_depth=2`) and locate Python files under `pkg/` with `find_files`. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-explore --feature file-tools --entry-point "python main.py (list_dir, find_files)" --seed-file "pkg/mod.py=x = 1\n" --seed-file "pkg/readme.md=docs\n" --seed-file "root.txt=top\n" --input "Use a ferramenta list_dir com max_depth=2 para listar o diretório atual." --input "Use a ferramenta find_files com pattern='pkg/*.py' e informe os caminhos encontrados." --input "memory" --input "exit"`
  The session exits with code `0`, `memory` shows `list_dir` returning `pkg/`, indented `mod.py`/`readme.md`, and `root.txt`, and `find_files` returning `pkg/mod.py` while excluding `pkg/readme.md` and `root.txt`.
- **Proof.** Inspect `%TEMP%\my-harness-verify\evidence\verify-files\file-tools\report.json` and `transcript.txt` (or `verify-grep` / `verify-explore` for search and exploration runs). Confirm both the chat output and `workspace_files_after["hello.txt"]["content"]` / `workspace_files_after["notes/todo.txt"]["content"]` on disk.

## Gotchas

- Never trust the agent's chat reply alone as proof that a file was created or edited; always assert `workspace_diff` and `workspace_files_after` in `report.json`.
- Always run inside an isolated scratch workspace (`control_harness.py drive` does this by default) so file edits never touch repository files.
- When `read_file` is called with a nonzero `offset`, the returned line-number prefix still starts counting from `1` for the sliced window rather than `offset + 1`.
- `search_replace` requires the search string to occur **exactly once** in the file; duplicate or missing matches return an error message to the agent instead of modifying the file.
- `grep`, `list_dir`, and `find_files` default `path` to `"."` when omitted and share the same walk boundary: skipping `.git`, `__pycache__`, `.venv`, `venv`, `node_modules`, `.pytest_cache`, `.ruff_cache`, `.sessions`, sensitive files (`.env`, keys), and escaping symlinks/junctions (`grep` also skips binary files containing NUL bytes in the first 8192 bytes).
- Clean up scratch state after the run with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-files`.
