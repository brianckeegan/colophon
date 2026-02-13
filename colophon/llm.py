"""LLM provider configuration and adapter clients used by authoring agents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib import error, request


class LLMClient(Protocol):
    """Provider-agnostic interface for text generation."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate output text for a single prompt."""


class LLMError(RuntimeError):
    """Raised when an LLM API call fails or returns an invalid payload."""


@dataclass(slots=True)
class LLMConfig:
    """Provider-neutral LLM settings for API-backed text generation.

    Parameters
    ----------
    provider : str
        Provider alias (for example ``openai`` or ``claude``).
    model : str
        Model identifier used by the provider API.
    api_base_url : str | None
        Base URL for the provider endpoint.
    api_key_env : str | None
        Environment variable name containing the API token.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum completion token budget.
    timeout_seconds : float
        Request timeout in seconds.
    system_prompt : str
        Optional default system prompt.
    extra_headers : dict[str, str]
        Additional HTTP headers for provider-specific auth/routing.
    """

    provider: str = "none"
    model: str = ""
    api_base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_seconds: float = 30.0
    system_prompt: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat-completions endpoints.

    Parameters
    ----------
    config : LLMConfig
        Resolved LLM configuration for request construction.
    """

    config: LLMConfig

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate.

        Parameters
        ----------
        prompt : str
            Parameter description.
        system_prompt : str | None
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        api_key = _read_api_key(self.config.api_key_env)
        base_url = _require_value(self.config.api_base_url, "api_base_url")
        model = _require_value(self.config.model, "model")
        messages = _build_chat_messages(prompt=prompt, default_system_prompt=self.config.system_prompt, override=system_prompt)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)

        data = _post_json(
            url=f"{base_url.rstrip('/')}/chat/completions",
            payload=payload,
            headers=headers,
            timeout_seconds=self.config.timeout_seconds,
        )
        return _extract_openai_text(data)


@dataclass(slots=True)
class AnthropicClient:
    """Client for Anthropic Messages API-compatible endpoints.

    Parameters
    ----------
    config : LLMConfig
        Resolved LLM configuration for request construction.
    """

    config: LLMConfig

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate.

        Parameters
        ----------
        prompt : str
            Parameter description.
        system_prompt : str | None
            Parameter description.

        Returns
        -------
        str
            Return value description.
        """
        api_key = _read_api_key(self.config.api_key_env)
        base_url = _require_value(self.config.api_base_url, "api_base_url")
        model = _require_value(self.config.model, "model")

        payload = {
            "model": model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        resolved_system = (system_prompt or self.config.system_prompt).strip()
        if resolved_system:
            payload["system"] = resolved_system

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)

        data = _post_json(
            url=f"{base_url.rstrip('/')}/messages",
            payload=payload,
            headers=headers,
            timeout_seconds=self.config.timeout_seconds,
        )
        return _extract_anthropic_text(data)


def create_llm_client(config: LLMConfig) -> LLMClient | None:
    """Create an LLM client from provider config.

    Supported provider aliases:
    - none, off, disabled
    - openai, codex, openai_compatible
    - anthropic, claude
    - github, copilot
    """

    provider = config.provider.strip().lower()
    if provider in {"", "none", "off", "disabled"}:
        return None

    resolved = _apply_provider_presets(config)
    normalized_provider = resolved.provider.strip().lower()

    if normalized_provider in {"openai", "codex", "openai_compatible", "github", "copilot"}:
        return OpenAICompatibleClient(config=resolved)
    if normalized_provider in {"anthropic", "claude"}:
        return AnthropicClient(config=resolved)

    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _apply_provider_presets(config: LLMConfig) -> LLMConfig:
    """Apply provider presets.

    Parameters
    ----------
    config : LLMConfig
        Parameter description.

    Returns
    -------
    LLMConfig
        Return value description.
    """
    provider = config.provider.strip().lower()

    # Preserve explicit values; fill only missing fields from provider presets.
    base_url = config.api_base_url
    api_key_env = config.api_key_env

    if provider in {"openai", "codex", "openai_compatible"}:
        base_url = base_url or "https://api.openai.com/v1"
        api_key_env = api_key_env or "OPENAI_API_KEY"
    elif provider in {"anthropic", "claude"}:
        base_url = base_url or "https://api.anthropic.com/v1"
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"
    elif provider in {"github", "copilot"}:
        # Copilot-style chat models are commonly exposed through GitHub Models endpoint.
        base_url = base_url or "https://models.inference.ai.azure.com"
        api_key_env = api_key_env or "GITHUB_TOKEN"

    return LLMConfig(
        provider=config.provider,
        model=config.model,
        api_base_url=base_url,
        api_key_env=api_key_env,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
        system_prompt=config.system_prompt,
        extra_headers=dict(config.extra_headers),
    )


def _build_chat_messages(prompt: str, default_system_prompt: str, override: str | None) -> list[dict[str, str]]:
    """Build chat messages.

    Parameters
    ----------
    prompt : str
        Parameter description.
    default_system_prompt : str
        Parameter description.
    override : str | None
        Parameter description.

    Returns
    -------
    list[dict[str, str]]
        Return value description.
    """
    messages: list[dict[str, str]] = []
    system = (override or default_system_prompt).strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout_seconds: float) -> dict:
    """Post json.

    Parameters
    ----------
    url : str
        Parameter description.
    payload : dict
        Parameter description.
    headers : dict[str, str]
        Parameter description.
    timeout_seconds : float
        Parameter description.

    Returns
    -------
    dict
        Return value description.
    """
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:  # pragma: no cover - network behavior is mocked in tests.
        details = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"HTTP {exc.code} from LLM provider: {details}") from exc
    except error.URLError as exc:  # pragma: no cover - network behavior is mocked in tests.
        raise LLMError(f"Failed to reach LLM provider: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM provider returned non-JSON response") from exc


def _extract_openai_text(payload: dict) -> str:
    """Extract openai text.

    Parameters
    ----------
    payload : dict
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("OpenAI-compatible response missing choices")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        text = content.strip()
        if text:
            return text

    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        text = "".join(chunks).strip()
        if text:
            return text

    raise LLMError("OpenAI-compatible response missing message content")


def _extract_anthropic_text(payload: dict) -> str:
    """Extract anthropic text.

    Parameters
    ----------
    payload : dict
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise LLMError("Anthropic response missing content")

    chunks = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])

    text = "".join(chunks).strip()
    if not text:
        raise LLMError("Anthropic response missing text content")
    return text


def _read_api_key(api_key_env: str | None) -> str:
    """Read api key.

    Parameters
    ----------
    api_key_env : str | None
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    env_name = _require_value(api_key_env, "api_key_env")
    value = os.getenv(env_name, "").strip()
    if not value:
        raise LLMError(f"Missing API key in environment variable: {env_name}")
    return value


def _require_value(value: str | None, field_name: str) -> str:
    """Require value.

    Parameters
    ----------
    value : str | None
        Parameter description.
    field_name : str
        Parameter description.

    Returns
    -------
    str
        Return value description.
    """
    if value is None:
        raise LLMError(f"Missing required LLM config field: {field_name}")
    cleaned = value.strip()
    if not cleaned:
        raise LLMError(f"Missing required LLM config field: {field_name}")
    return cleaned
