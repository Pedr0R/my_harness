# Interactive chat session

Interactive chat session lets a user launch `my_harness` in the terminal, exchange multi-turn messages that retain prior context, ignore accidental blank lines, and close the session cleanly with a final interaction summary.

## Sub-features

- `chat-launch` prints the interactive banner and waits at the `Você: ` prompt.
- `chat-blank` ignores empty or whitespace-only input without calling the model or adding to history.
- `chat-multiturn` carries user facts from an earlier turn into a later reply in the same session.
- `chat-exit-command` ends the session cleanly via `exit`, `sair`, `quit`, or `q` and prints the session summary.
- `chat-exit-eof` ends the session cleanly on stdin EOF (`Ctrl+Z`/`Ctrl+D`) and still prints the session summary.

## How to get to it (user POV)

- Run `python main.py` in a terminal and type messages at the `Você: ` prompt.
- Press Enter on a blank line at `Você: `.
- Type `exit`, `sair`, `quit`, or `q` at `Você: ` to end the session.
- Send EOF (close stdin / `Ctrl+D` on Unix / `Ctrl+Z` + Enter on Windows) at `Você: ` to end the session via interruption.

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- Valid Gemini credentials are present so live chat turns can reach the default Gemini provider.

- **Multi-turn chat, blank input, and `sair` exit.** Launch an isolated session, send a blank line, state a codeword on turn 1, ask for the codeword on turn 2, and close with `sair`. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-chat --feature chat-session --entry-point "python main.py (sair)" --input "" --input "Guarde exatamente este código: AZUL-742. Responda apenas: GUARDADO" --input "Qual foi o código que pedi para guardar? Responda apenas o código." --input "sair"`
  The startup banner shows `🤖 Chat Interativo com SWE-Agent`, the blank turn returns straight to `Você: ` without `[Agente pensando...]`, turn 2's `Agente:` reply contains `AZUL-742`, `sair` prints `Encerrando conversa...`, `Sessão finalizada: session_...`, and `Interações guardadas na memória: 2`, and exit code is `0`.
- **EOF exit path.** Launch a separate session and close stdin without typing an exit word. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-chat --feature chat-session-eof --entry-point "python main.py (EOF)" --input "__EOF__"`
  Stdout shows `Encerrando sessão via interrupção...`, `Sessão finalizada: session_...`, and `Interações guardadas na memória: 0`, and exit code is `0`.
- **Proof.** Inspect `transcript.txt` and `report.json` in `%TEMP%\my-harness-verify\evidence\verify-chat\chat-session\` and `%TEMP%\my-harness-verify\evidence\verify-chat\chat-session-eof\`. Both artifacts confirm `exit_code: 0`, empty stderr, the user inputs, the agent replies, and the final summary counts.

## Gotchas

- Cold startup on Windows can take 8–15 seconds before the banner and `Você: ` prompt appear; do not shorten `--startup-timeout` below 30 seconds.
- The banner only advertises `exit` and `sair`, while `quit` and `q` (any letter case) are also accepted exit commands.
- Every non-empty, non-command prompt makes a live Gemini API request; keep verification prompts short and deterministic to avoid flaky wording or extra latency.
- Clean up scratch state after the run with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-chat`; evidence in `%TEMP%\my-harness-verify\evidence\verify-chat\` stays intact.
