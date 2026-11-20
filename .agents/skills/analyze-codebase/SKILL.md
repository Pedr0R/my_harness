---
name: analyze-codebase
description: "Run automated structural, interface contract, and test suite verification across my_harness: verifies tool-schema parity, provider protocol conformance, guardrails integrity, and executes unit tests."
---

# Analyze Codebase

Analyze the internal health and architectural integrity of `my_harness`.

## Checks Executed

1. **Tool Schema Parity**: Verifies that every tool defined in `TOOLS` in `agent.py` has a matching schema in `TOOL_SCHEMAS`, and that no orphan schemas exist.
2. **Provider Contract**: Inspects all provider classes in `providers/` (`GeminiProvider`, `OpenAICompatibleProvider`, `FallbackProvider`) to verify compliance with the `ModelProvider` protocol (`complete` method).
3. **Guardrails Integrity**: Verifies the presence and export of all required guardrail modules in `guardrails/` (`path_sandbox.py`, `command_safety.py`, `secrets_redaction.py`, `loop_detection.py`, `config.py`, `base.py`).
4. **Test Suite Execution**: Discovers and runs all unit tests in `tests/`, reporting total test count, failures, and errors.

## Usage

Run the deterministic analyzer script from the repository root:

```powershell
python .agents/skills/analyze-codebase/scripts/analyze.py
```

### Exit Codes

- `0`: All checks passed (`overall_ok: true`).
- `1`: One or more checks failed.
