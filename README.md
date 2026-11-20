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
- **Chat Interativo no Terminal**: Conversação contínua com histórico e memória em array/dicionário.
- **Persistência de Sessões**: Salvamento automático após cada turno em `.sessions/<session_id>.json` e retomada via CLI (`--session`) ou comando `/load`.
- **Guardrails de Segurança**: Sandboxing de caminhos (`read_file`/`write_file`/`search_replace`), bloqueio de comandos destrutivos no `bash`, redação automática de segredos (`[REDACTED]`) e detecção de loops de ferramentas.
- **Compactação de Contexto**: Redução automática de mensagens para evitar estourar a janela de contexto.

---

## Estrutura do Projeto

```text
my_harness/
├── .env                # Chaves de API e configurações de ambiente
├── .sessions/          # Histórico persistido de sessões em JSON (ignorado no Git)
├── main.py             # Chat interativo no terminal
├── agent.py            # Loop de execução, memória e ferramentas do agente
├── guardrails/         # Validação, sandboxing, redação de segredos e detecção de loop
├── tests/              # Suíte de testes unitários (unittest)
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

Opcionalmente, você pode informar outro diretório de sessões com `--sessions-dir <caminho>` ou desabilitar guardrails específicos:

```bash
python main.py --disable-guardrail loop_detection
python main.py --disable-guardrail all
```

Também é possível controlar cada guardrail via variáveis de ambiente (`GUARDRAIL_PATH_SANDBOXING`, `GUARDRAIL_COMMAND_SAFETY`, `GUARDRAIL_SECRETS_REDACTION`, `GUARDRAIL_LOOP_DETECTION`) com `true`/`false`.

### Testes Unitários:

```bash
python -m unittest discover -s tests -v
```

### Comandos no Terminal:
- `exit` ou `sair` (também `quit` ou `q`): Encerra a conversa e exibe o resumo da sessão.
- `memory`: Exibe o histórico de mensagens gravado na memória da sessão atual.
- `/sessions`: Lista as sessões salvas em `.sessions/`.
- `/load <session_id|last>`: Carrega uma sessão existente (ou `last` para a mais recente) sem reiniciar o terminal.

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
