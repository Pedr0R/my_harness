# Session memory inspection

Session memory inspection lets a user type `memory` at any time during an interactive chat to view the current internal turn count, estimated session cost, total recorded messages, and a timestamped log of user and agent turns without ending the conversation.

## Sub-features

- `memory-empty` reports zero turns, `$0.0000` estimated cost, and zero recorded messages before any prompt is sent.
- `memory-populated` reports incremented turn count, updated message count, and timestamped `USER:` and `AGENT:` lines after a live turn.
- `memory-non-recording` does not add the `memory` command itself to the recorded chat history or final interaction count.

## How to get to it (user POV)

- Run `python main.py` in a terminal and type `memory` (any case) at the `Você: ` prompt before or after chatting.

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- Valid Gemini credentials are present for the live turn between the two `memory` checks.

- **Empty baseline, live turn, and populated readout.** Launch a fresh session, run `memory` before chatting, send one short prompt, run `memory` again, and exit. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py drive --run-id verify-mem --feature session-memory --entry-point "python main.py -> memory" --input "memory" --input "Responda apenas: MEMORIA-OK" --input "memory" --input "exit"`
  The first `memory` output prints `[Memória da Sessão Atual]:`, `Total de turnos internos: 0`, `Custo estimado: $0.0000`, and `Total de mensagens gravadas: 0`. The second `memory` output prints `Total de turnos internos: 1`, `Total de mensagens gravadas: 2`, one `USER: Responda apenas: MEMORIA-OK...` line, and one `AGENT: MEMORIA-OK...` line. Closing with `exit` reports `Interações guardadas na memória: 1` (confirming `memory` itself was not logged as an interaction), with exit code `0`.
- **Proof.** Inspect `%TEMP%\my-harness-verify\evidence\verify-mem\session-memory\transcript.txt` and `report.json`. Verify both the empty baseline and populated `memory` outputs appear in sequence inside the same session.

## Gotchas

- History preview lines always end with `...` even when the message is shorter than 80 characters (for example, `AGENT: MEMORIA-OK...`).
- Estimated cost is formatted to 4 decimal places (`$0.0000`); a single tiny Gemini Flash-Lite turn often costs less than `$0.00005` and legitimately displays as `$0.0000`. Do not assert that the printed string is nonzero on a short prompt.
- `memory` is case-insensitive (`MEMORY` works identically to `memory`), while leading/trailing spaces are stripped.
- Clean up scratch state after the run with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-mem`.
