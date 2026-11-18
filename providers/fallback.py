from __future__ import annotations

import logging
from typing import Any

from providers.base import ModelProvider, ModelResponse

logger = logging.getLogger(__name__)


class FallbackProvider:
    """
    Provedor composto (Composite Pattern) que implementa tolerância a falhas.
    Executa os provedores em ordem de prioridade. Se um falhar (rate limit, indisponibilidade, timeout),
    repassa a chamada automaticamente para o próximo.
    """

    def __init__(self, providers: list[ModelProvider]):
        if not providers:
            raise ValueError("Ao menos um provedor deve ser fornecido para o FallbackProvider.")
        self.providers = providers

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        errors: list[str] = []

        for index, provider in enumerate(self.providers):
            provider_name = getattr(provider, "model_name", getattr(provider, "model", provider.__class__.__name__))
            try:
                return await provider.complete(messages, tools)
            except Exception as e:
                err_msg = f"[{provider_name} (#{index + 1})]: {type(e).__name__} - {e}"
                logger.warning(f"Fallback acionado. Erro no provedor {err_msg}")
                errors.append(err_msg)

        raise RuntimeError(
            f"Todos os {len(self.providers)} provedores configurados falharam:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
