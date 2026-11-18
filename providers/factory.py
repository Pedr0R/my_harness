from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv

from providers.base import ModelProvider
from providers.fallback import FallbackProvider
from providers.gemini import GeminiProvider
from providers.openai_compat import OpenAICompatibleProvider

load_dotenv()


def get_provider(name: str | None = None, **kwargs: Any) -> ModelProvider:
    """
    Fábrica para instanciar provedores de modelos de forma desacoplada.

    Exemplos de uso:
        # Padrão (.env ou Gemini):
        provider = get_provider()

        # Específico por nome:
        provider = get_provider("gemini", model="gemini-3.5-flash-lite")
        provider = get_provider("grok", model="grok-2-latest")
        provider = get_provider("openai", model="gpt-4o")
        provider = get_provider("deepseek", model="deepseek-chat")
        provider = get_provider("ollama", model="llama3.2")

        # Com Fallback:
        provider = get_provider("fallback", providers=[
            get_provider("grok"),
            get_provider("gemini")
        ])
    """
    provider_name = (name or os.getenv("DEFAULT_PROVIDER", "gemini")).lower().strip()

    if "fallback" in provider_name:
        providers_list = kwargs.get("providers")
        if not providers_list:
            raise ValueError("O provedor 'fallback' exige o argumento `providers=[...]`.")
        return FallbackProvider(providers=providers_list)

    if provider_name.startswith("gemini"):
        model = kwargs.get("model", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
        return GeminiProvider(
            model=model,
            api_key=kwargs.get("api_key", os.getenv("GEMINI_API_KEY")),
            project_id=kwargs.get("project_id", os.getenv("GEMINI_PROJECT_ID")),
        )

    if provider_name.startswith("grok"):
        model = kwargs.get("model", os.getenv("GROK_MODEL", "grok-2-latest"))
        return OpenAICompatibleProvider(
            model=model,
            api_key=kwargs.get("api_key", os.getenv("GROK_API_KEY")),
            base_url=kwargs.get("base_url", "https://api.x.ai/v1"),
            timeout=kwargs.get("timeout", 60.0),
        )

    if provider_name.startswith("openai"):
        model = kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-4o"))
        return OpenAICompatibleProvider(
            model=model,
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
            base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
            timeout=kwargs.get("timeout", 60.0),
        )

    if provider_name.startswith("deepseek"):
        model = kwargs.get("model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
        return OpenAICompatibleProvider(
            model=model,
            api_key=kwargs.get("api_key", os.getenv("DEEPSEEK_API_KEY")),
            base_url=kwargs.get("base_url", "https://api.deepseek.com"),
            timeout=kwargs.get("timeout", 90.0),
        )

    if provider_name.startswith("ollama"):
        model = kwargs.get("model", os.getenv("OLLAMA_MODEL", "llama3.2"))
        return OpenAICompatibleProvider(
            model=model,
            api_key="ollama",
            base_url=kwargs.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")),
            timeout=kwargs.get("timeout", 120.0),
        )

    raise ValueError(
        f"Provedor não reconhecido: '{provider_name}'. "
        f"Opções disponíveis: 'gemini', 'grok', 'openai', 'deepseek', 'ollama', 'fallback'."
    )
