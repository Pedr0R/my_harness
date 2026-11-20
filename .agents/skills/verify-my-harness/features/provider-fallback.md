# Provider switching and fallback

Provider switching and fallback let a user select a model provider through `get_provider` (`gemini`, `grok`, `openai`, `deepseek`, `ollama`, or `fallback`) and automatically fail over to the next provider when the primary provider returns an HTTP error or becomes unavailable.

## Sub-features

- `provider-factory` instantiates `GeminiProvider`, `OpenAICompatibleProvider`, and `FallbackProvider` from `get_provider(...)`.
- `provider-openai-compat` sends OpenAI Chat Completions requests (messages + tool schemas) to a configured `base_url` and calculates token cost from usage metadata.
- `provider-fallback-failover` attempts providers in order and seamlessly returns the secondary provider's response when the primary fails with HTTP `429`.
- `provider-third-party-live` connects to live external/local providers (`grok`, `openai`, `deepseek`, `ollama`) when their credentials or local daemon are present.

## How to get to it (user POV)

- Import `get_provider` (and optionally `FallbackProvider`) from `providers` as documented in `README.md` under "Trocar provedor", pass the resulting provider to `Agent(model=provider)`, and call `await agent.run(...)` (or configure `main.py` with that provider).

## Driving it with control_harness.py

Preconditions:

- `python .agents/skills/verify-my-harness/scripts/control_harness.py doctor` reports `"healthy": true`.
- No external API key is needed for the local HTTP boundary failover check (`verify-fallback` binds an ephemeral server on `127.0.0.1:0` and stops it automatically).
- Live third-party checks (`grok`, `openai`, `deepseek`, `ollama`) additionally require `GROK_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or a running local Ollama server at `http://localhost:11434/v1`.

- **Failover across a real HTTP boundary.** Drive `get_provider("openai", ...)` and `get_provider("fallback", providers=[primary, secondary])` inside `Agent.run(...)` against a local HTTP server whose primary endpoint returns HTTP `429` and whose secondary endpoint returns HTTP `200` with usage metadata. Run:
  `python .agents/skills/verify-my-harness/scripts/control_harness.py verify-fallback --run-id verify-fb --feature provider-fallback`
  Exit code is `0`, `observed_http_requests` records two HTTP POST calls in order (`/failing/v1/chat/completions` first, then `/healthy/v1/chat/completions`), the request payload includes the four native tool names (`bash`, `read_file`, `write_file`, `search_replace`), `response` contains `Fallback response from local OpenAI-compatible server.`, `agent_turns` is `1`, and `agent_cost` is greater than `0`.
- **Proof.** Inspect `%TEMP%\my-harness-verify\evidence\verify-fb\provider-fallback\report.json` and `transcript.txt`. Confirm the ordered HTTP request log, tool names, response text, turn count, and positive cost estimate.

## Gotchas

- `main.py` hardcodes `get_provider("gemini", model="gemini-3.5-flash-lite")` and does not accept a CLI flag or `DEFAULT_PROVIDER` override yet; verifying non-Gemini providers or `FallbackProvider` uses the documented Python API (`providers.get_provider` + `Agent`) rather than `python main.py` flags.
- If a run needs to prove live `grok`, `openai`, `deepseek`, or `ollama` connectivity and the corresponding API key is absent (check `auth_configured` in `doctor` output) or Ollama is not running on port `11434`, report that sub-feature as unreachable with the missing credential/service — never claim `verify-fallback` proved a live third-party provider.
- Clean up after the run with `python .agents/skills/verify-my-harness/scripts/control_harness.py cleanup --run-id verify-fb` (no scratch workspace is left behind by `verify-fallback`, and evidence stays preserved).
