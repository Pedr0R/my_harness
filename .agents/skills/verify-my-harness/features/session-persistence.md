# Session persistence and resumption

Session persistence automatically saves conversation state (agent messages, turn count, cost, and chat history) to `.sessions/<session_id>.json` after every completed turn and lets a user list saved sessions or resume a prior conversation via CLI (`--session <id>`) or inside the REPL (`/load <id>`).

## Sub-features

- `persist-autosave` writes `.sessions/<session_id>.json` after each completed turn using the required JSON schema without creating empty files for zero-turn sessions.
- `persist-resume-cli` launches `python main.py --session <id>`, restores prior context, turns, cost, and history, and preserves context across new turns (including sessions with tool calls).
- `persist-resume-repl` lists saved sessions via `/sessions` and loads an existing session via `/load <id>` without restarting the terminal.
- `persist-invalid-session` reports a clear error for missing, invalid, or path-traversal session IDs without corrupting active state.
- `persist-gitignore` ensures `.sessions/` is excluded from Git tracking.

## How to get to it (user POV)

- Run `python main.py`, exchange one or more messages, and observe `.sessions/<session_id>.json` saved automatically (and its path printed on exit).
- Run `python main.py --session <session_id>` (or `--session last` for the most recent saved session) to resume a saved session from the terminal.
- Type `/sessions` at the `Você: ` prompt to list saved sessions, then `/load <session_id>` or `/load last` to restore one in place.

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- Valid Gemini credentials are present.
- Reuse the same `--run-id` across the save and resume steps so both drives share the same isolated scratch workspace `.sessions/` directory.

- **Save a live session with context and a tool call.** Start a fresh session, store a unique codeword, invoke `write_file`, and exit. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-persist --feature session-persistence-save --entry-point "python main.py (auto-save)" --input "Guarde exatamente o código PERSIST-884 e responda apenas: GUARDADO" --input "Use a ferramenta write_file para criar marcador.txt com o conteúdo exato: ok-persist" --input "exit"`
  Exit code is `0`, stdout prints `Sessão salva em: ...\.sessions\session_....json`, and `report.json` shows a new file `.sessions/<session_id>.json` containing `session_id`, `started_at`, `updated_at`, `n_turns`, `cost`, `messages`, and `history`, plus `marcador.txt` with `ok-persist`.
- **Resume via `--session <session_id>`.** Pass the saved `<session_id>` back to `main.py` in the same isolated workspace, ask for the codeword, inspect `memory`, and exit. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-persist --feature session-persistence-cli --entry-point "python main.py --session <id>" --cli-arg=--session --cli-arg=<session_id> --input "Qual foi o código que pedi para guardar na conversa anterior? Responda apenas o código." --input "memory" --input "exit"`
  The startup banner includes `Sessão retomada: <session_id>`, the agent replies with `PERSIST-884`, `memory` shows accumulated `Total de turnos internos` and `Total de mensagens gravadas: 6`, and `.sessions/<session_id>.json` is updated on disk with the new turn.
- **List and load via `/sessions` and `/load`.** Start a new session in the same workspace, list sessions, load `<session_id>`, verify `memory`, and exit. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-persist --feature session-persistence-repl --entry-point "python main.py -> /sessions, /load" --input "/sessions" --input "/load <session_id>" --input "memory" --input "exit"`
  `/sessions` lists `<session_id>`, `/load <session_id>` prints `[Sessão carregada]: <session_id>`, and `memory` shows the restored turn and message counts.
- **Proof.** Inspect `report.json` and `transcript.txt` for all three phases under `%TEMP%\my-harness-verify\evidence\verify-persist\`, plus confirm `git check-ignore -v .sessions/sample.json` matches `.gitignore`.

## Gotchas

- Opening `python main.py` and immediately typing `exit` (zero interactions) intentionally does **not** create a `.sessions/<session_id>.json` file.
- Passing `--cli-arg=--session` with an equals sign is required so `argparse` in `control_harness.py` does not interpret `--session` as its own flag.
- Gemini responses with tool calls store `raw_content` (including `thought_signature`) serialized inside `.sessions/<session_id>.json`; resuming a session that previously invoked tools must succeed without `TypeError` or Gemini API signature errors.
- Clean up scratch state after all phases complete with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-persist`.
