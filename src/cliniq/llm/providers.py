"""Provider-agnostic LLM interface for the narrative-generation step.

Every provider implements ``generate(prompt, context) -> str``, so the agent
never changes when the backend does; only CLINIQ_LLM_PROVIDER does.

  CLINIQ_LLM_PROVIDER=template   default; deterministic, offline, no API key
  CLINIQ_LLM_PROVIDER=ollama     OLLAMA_HOST (default http://localhost:11434),
                                 OLLAMA_MODEL (default llama3.1)
  CLINIQ_LLM_PROVIDER=openai     OPENAI_API_KEY, OPENAI_MODEL (default gpt-4o-mini)
  CLINIQ_LLM_PROVIDER=anthropic  ANTHROPIC_API_KEY, ANTHROPIC_MODEL
                                 (default claude-haiku-5-5)

``context`` carries the same structured facts (risk score, SHAP drivers,
retrieved passages) that are rendered into ``prompt``. Generative providers
send ``prompt``; TemplateProvider renders from ``context``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import requests


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; it is required for this LLM provider.")
    return value


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str: ...


class TemplateProvider(LLMProvider):
    """Deterministic narrative built from structured context.

    Every number comes from the model upstream and every clinical sentence is
    a verbatim quote of a retrieved chunk, so the output cannot contain an
    unsupported claim. That makes it a reproducible baseline for the
    guardrail and eval harness: if its output fails the groundedness check,
    the cause is retrieval, not generation. The prose is mechanical; use a
    generative provider for fluent narratives.
    """

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        lines = [
            f"Estimated 30-day readmission risk: {ctx.get('risk_score', 0) * 100:.1f}% "
            f"(cohort base rate: {ctx.get('base_rate', 0) * 100:.1f}%)."
        ]
        retrieved = ctx.get("retrieved_by_driver", {})
        for d in ctx.get("drivers", []):
            feature, value = d["feature"], d["value"]
            lines.append(f"\nContributing factor -- {feature} = {value}:")
            hits = retrieved.get(feature, [])
            if hits:
                top = hits[0]
                lines.append(f'  "{top["text"]}" ({top["authors"]}, {top["year"]})')
            else:
                lines.append("  No directly relevant passage was retrieved for this factor.")
        lines.append(
            "\nThis summary is generated from retrieved sources only; it is a "
            "decision-support aid, not a clinical diagnosis, and should be "
            "reviewed by a clinician before acting on it."
        )
        return "\n".join(lines)


class OllamaProvider(LLMProvider):
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return str(resp.json()["response"])


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or _require_env("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or _require_env("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return str(resp.json()["content"][0]["text"])


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "template": TemplateProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    name = name or os.environ.get("CLINIQ_LLM_PROVIDER", "template")
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Choose from: {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]()
