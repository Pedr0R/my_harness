# My Harness

Um harness leve e modular para agentes autônomos de software (SWE Agents) em Python.
---

- **Provedores Desacoplados**: Troca simples entre **Google Gemini**, **Grok (xAI)**, **OpenAI**, **DeepSeek** e **Ollama**.
- **Fallback Automático**: Resiliência contra *rate limits* e indisponibilidade de APIs.
- **Ferramentas Nativas**:
  - `bash`: Execução de comandos no terminal.
  - `read_file`: Leitura de arquivos com offset e limite de linhas.
  - `write_file`: Criação e sobrescrita de arquivos.
  - `search_replace`: Substituição precisa de textos em arquivos.
  - `grep`: Busca de texto literal ou regex em arquivos e diretórios com filtro glob e limite de resultados.
  - `list_dir`: Listagem hierárquica de diretórios e arquivos com profundidade configurável (`max_depth`).
  - `find_files`: Busca recursiva de arquivos por padrão glob de nome ou caminho relativo.
- **Chat Interativo no Terminal**: Conversação contínua com histórico e memória em array/dicionário.
- **Streaming em Tempo Real**: Emissão progressiva de tokens e eventos de execução de ferramentas (`Agent.run_stream`), com suporte nativo em Gemini e APIs compatíveis via SSE, fallback para provedores sem streaming e flag `--no-stream`.
- **Persistência de Sessões**: Salvamento automático após cada turno em `.sessions/<session_id>.json` e retomada via CLI (`--session`) ou comando `/load`.
- **Guardrails de Segurança**: Sandboxing de caminhos (`read_file`/`write_file`/`search_replace`/`grep`/`list_dir`/`find_files`), bloqueio de comandos destrutivos no `bash`, redação automática de segredos (`[REDACTED]`) e detecção de loops de ferramentas.
- **Confirmação Humana (HitL)**: Aprovação interativa antes de executar chamadas com efeito colateral (`bash`, `write_file`, `search_replace`), com suporte a aprovar a sessão (`a`), flag `--auto-approve` (`-y`) e comando `/auto-approve`.
- **Checkpoint de Workspace e Rollback Automático**: Snapshots isolados via Git (`GIT_INDEX_FILE` temporário, sem poluir `git stash` ou `.git/index`), rollback automático em caso de falha/interrupção no meio do turno e comando manual `/undo` (ou `/rollback`), preservando `.sessions/` e alterações prévias do usuário.
- **Compactação de Contexto**: Redução automática de mensagens para evitar estourar a janela de contexto.

---

## Estrutura do Projeto

```text
my_harness/
├── .env                # Chaves de API e configurações de ambiente
├── .sessions/          # Histórico persistido de sessões em JSON (ignorado no Git)
├── main.py             # Chat interativo e entrypoint CLI
├── agent.py            # Loop de execução, compactação e limites do agente
├── session.py          # Persistência e validação de sessões em .sessions/
├── workspace.py        # Checkpoint/rollback via Git e política neutra de caminhos sensíveis
├── pyproject.toml      # Configuração do pytest e do linter ruff
├── tools/              # Ferramentas nativas desacopladas (ToolSpec, shell, files, search)
├── guardrails/         # Validação, sandboxing, redação de segredos e detecção de loop
├── tests/              # Suíte de testes unitários (pytest / unittest)
└── providers/          # Camada desacoplada de modelos
    ├── base.py         # Protocolos (ModelProvider, ModelResponse, ToolCall)
    ├── factory.py      # Fábrica de provedores (get_provider)
    ├── fallback.py     # Provedor com fallback resiliente
    ├── gemini.py       # Adaptador para Google Gemini
    └── openai_compat.py# Adaptador para APIs compatíveis com OpenAI (Grok, Ollama, etc.)
```

---

## Configs

1. Crie um arquivo `.env` na raiz do projeto (já incluso no `.gitignore`):

```env
GEMINI_API_KEY=sua_chave_gemini_aqui
# Opcionais:
# GROK_API_KEY=sua_chave_grok_aqui
# OPENAI_API_KEY=sua_chave_openai_aqui
```

2. Instale as dependências:

```bash
pip install google-genai httpx python-dotenv
```

---

Execute o chat interativo no terminal:

```bash
python main.py
```

Para retomar uma sessão salva anteriormente em `.sessions/` (ou usar `last` para a mais recente):

```bash
python main.py --session session_20261119_220000
python main.py --session last
```

Opcionalmente, você pode escolher o provedor/modelo, informar outro diretório de sessões com `--sessions-dir <caminho>` ou desabilitar guardrails específicos:

```bash
python main.py --provider ollama --model llama3.2
python main.py --disable-guardrail loop_detection
python main.py --disable-guardrail all
python main.py --no-checkpoint
python main.py --auto-approve
python main.py --no-stream
```

Também é possível controlar cada guardrail via variáveis de ambiente (`GUARDRAIL_PATH_SANDBOXING`, `GUARDRAIL_COMMAND_SAFETY`, `GUARDRAIL_SECRETS_REDACTION`, `GUARDRAIL_LOOP_DETECTION`) com `true`/`false`.

### Testes Unitários e Lint:

Instale as dependências de desenvolvimento (`pytest` e `ruff`) e execute a suíte (nenhuma chave de API é necessária):

```bash
pip install pytest ruff
pytest -v
ruff check .
```

Os testes também são compatíveis com `unittest` da biblioteca padrão:

```bash
python -m unittest discover -s tests -v
```

O workflow de CI no GitHub Actions (`.github/workflows/ci.yml`) executa automaticamente `pytest` na matriz Python 3.10–3.13, `ruff check .` e a análise estrutural do repositório em cada push e pull request.

### Comandos no Terminal:
- `exit` ou `sair` (também `quit` ou `q`): Encerra a conversa e exibe o resumo da sessão.
- `memory`: Exibe o histórico de mensagens gravado na memória da sessão atual.
- `/sessions`: Lista as sessões salvas em `.sessions/`.
- `/load <session_id|last>`: Carrega uma sessão existente (ou `last` para a mais recente) sem reiniciar o terminal.
- `/undo` (ou `/rollback`): Desfaz as alterações de arquivos realizadas pelo último turno concluído usando o checkpoint mais recente.
- `/auto-approve [on|off]`: Ativa, desativa ou alterna a aprovação automática de chamadas de ferramentas na sessão atual.

---

## Trocar provedor

No arquivo `main.py`:

```python
from providers import get_provider, FallbackProvider

# 1. Usar Gemini (padrão)
provider = get_provider("gemini", model="gemini-3.5-flash-lite")

# 2. Usar Grok (xAI)
# provider = get_provider("grok", model="grok-2-latest")

# 3. Usar Ollama local (sem gastar tokens)
# provider = get_provider("ollama", model="llama3.2")

# 4. Usar Fallback (tenta Grok; se falhar, usa Gemini)
# provider = FallbackProvider([
#     get_provider("grok"),
#     get_provider("gemini")
# ])
```
