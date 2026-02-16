"""LLM provider configuration and adapter clients used by authoring agents."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
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
    pi_binary : str
        Executable name/path for the pi-mono coding agent binary.
    pi_provider : str
        Optional underlying provider passed to ``pi --provider``.
    pi_no_session : bool
        Whether to disable on-disk pi session persistence.
    pi_coordination_memory : int
        Number of recent coordination messages injected into prompts.
    pi_extra_args : list[str]
        Additional CLI arguments passed to the pi process.
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
    pi_binary: str = "pi"
    pi_provider: str = ""
    pi_no_session: bool = True
    pi_coordination_memory: int = 24
    pi_extra_args: list[str] = field(default_factory=list)


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


@dataclass(slots=True)
class PiMonoClient:
    """LLM client backed by the pi-mono coding agent RPC mode."""

    config: LLMConfig
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)
    _coordination_history: list[str] = field(default_factory=list, init=False, repr=False)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text by sending a prompt through pi RPC mode."""
        with self._lock:
            process = self._ensure_process()
            request_id = f"colophon-{uuid.uuid4().hex}"
            payload = {
                "id": request_id,
                "type": "prompt",
                "message": _compose_pi_message(
                    prompt=prompt,
                    system_prompt=system_prompt or self.config.system_prompt,
                    coordination_history=self._coordination_history,
                    max_coordination_messages=max(0, int(self.config.pi_coordination_memory)),
                ),
            }
            _write_rpc_command(process=process, payload=payload)

            response_seen = False
            run_complete = False
            chunks: list[str] = []
            assistant_snapshot = ""
            deadline = time.monotonic() + max(1.0, float(self.config.timeout_seconds))

            while time.monotonic() < deadline:
                event = _read_rpc_event(process=process)
                if event is None:
                    continue

                event_type = event.get("type")
                if event_type == "response" and event.get("id") == request_id:
                    response_seen = True
                    if not bool(event.get("success", False)):
                        raise LLMError(f"pi RPC prompt failed: {event.get('error', 'unknown error')}")
                    continue

                if event_type == "message_update":
                    assistant_event = event.get("assistantMessageEvent")
                    if isinstance(assistant_event, dict) and assistant_event.get("type") == "text_delta":
                        delta = assistant_event.get("delta")
                        if isinstance(delta, str):
                            chunks.append(delta)
                    continue

                if event_type == "agent_end":
                    run_complete = True
                    assistant_snapshot = _extract_assistant_text_from_agent_end(event)
                    if response_seen:
                        break

            if not run_complete:
                _try_abort_pi_run(process=process)
                raise LLMError("Timed out waiting for pi RPC response.")
            if not response_seen:
                raise LLMError("pi RPC did not acknowledge prompt request.")

            text = "".join(chunks).strip() or assistant_snapshot.strip()
            if not text:
                raise LLMError("pi RPC run completed but produced no assistant text.")
            return text

    def record_coordination_message(self, message: object) -> None:
        """Record a coordination message so future prompts share agent context."""
        rendered = _render_coordination_message(message)
        if not rendered:
            return
        self._coordination_history.append(rendered)
        cap = max(1, int(self.config.pi_coordination_memory))
        if len(self._coordination_history) > cap:
            self._coordination_history = self._coordination_history[-cap:]

    def close(self) -> None:
        """Terminate the backing pi process if it is running."""
        with self._lock:
            process = self._process
            self._process = None
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _ensure_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process

        binary = (self.config.pi_binary or "pi").strip() or "pi"
        if shutil.which(binary) is None:
            raise LLMError(f"pi binary not found on PATH: {binary}")

        command = [binary, "--mode", "rpc"]
        if self.config.pi_no_session:
            command.append("--no-session")
        provider_arg, model_arg = _resolve_pi_model_args(self.config)
        if provider_arg:
            command.extend(["--provider", provider_arg])
        if model_arg:
            command.extend(["--model", model_arg])
        for extra_arg in self.config.pi_extra_args:
            if isinstance(extra_arg, str) and extra_arg.strip():
                command.append(extra_arg.strip())

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = process
        return process


def create_llm_client(config: LLMConfig) -> LLMClient | None:
    """Create an LLM client from provider config.

    Supported provider aliases:
    - none, off, disabled
    - pi, pi_mono, pi-mono
    - openai, codex, openai_compatible
    - anthropic, claude
    - github, copilot
    """

    provider = config.provider.strip().lower()
    if provider in {"", "none", "off", "disabled"}:
        return None

    resolved = _apply_provider_presets(config)
    normalized_provider = resolved.provider.strip().lower()

    if normalized_provider in {"pi", "pi_mono", "pi-mono"}:
        return PiMonoClient(config=resolved)
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
        pi_binary=config.pi_binary,
        pi_provider=config.pi_provider,
        pi_no_session=config.pi_no_session,
        pi_coordination_memory=config.pi_coordination_memory,
        pi_extra_args=list(config.pi_extra_args),
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


def _compose_pi_message(
    prompt: str,
    system_prompt: str,
    coordination_history: list[str],
    max_coordination_messages: int,
) -> str:
    sections: list[str] = []
    cleaned_system = system_prompt.strip()
    if cleaned_system:
        sections.append("System guidance:\n" + cleaned_system)

    if coordination_history and max_coordination_messages > 0:
        recent = coordination_history[-max_coordination_messages:]
        rendered_recent = "\n".join(f"- {entry}" for entry in recent)
        sections.append("Coordination context:\n" + rendered_recent)

    sections.append("Task:\n" + prompt.strip())
    return "\n\n".join(section for section in sections if section.strip())


def _resolve_pi_model_args(config: LLMConfig) -> tuple[str, str]:
    provider = config.pi_provider.strip()
    model = config.model.strip()
    if provider:
        if model.startswith(f"{provider}/"):
            return provider, model.split("/", 1)[1]
        return provider, model
    return "", model


def _write_rpc_command(process: subprocess.Popen[str], payload: dict) -> None:
    if process.stdin is None:
        raise LLMError("pi RPC process stdin is unavailable.")
    try:
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()
    except BrokenPipeError as exc:
        raise LLMError("pi RPC process is not accepting input.") from exc


def _read_rpc_event(process: subprocess.Popen[str]) -> dict | None:
    if process.stdout is None:
        raise LLMError("pi RPC process stdout is unavailable.")

    line = process.stdout.readline()
    if not line:
        if process.poll() is not None:
            stderr_text = ""
            if process.stderr is not None:
                try:
                    stderr_text = process.stderr.read().strip()
                except Exception:
                    stderr_text = ""
            details = f" (stderr: {stderr_text})" if stderr_text else ""
            raise LLMError(f"pi RPC process exited unexpectedly{details}")
        time.sleep(0.01)
        return None

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _try_abort_pi_run(process: subprocess.Popen[str]) -> None:
    try:
        _write_rpc_command(process=process, payload={"type": "abort"})
    except Exception:
        return


def _extract_assistant_text_from_agent_end(event: dict) -> str:
    messages = event.get("messages", [])
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        text = _extract_assistant_text_from_message(message)
        if text:
            return text
    return ""


def _extract_assistant_text_from_message(message: object) -> str:
    if not isinstance(message, dict):
        return ""

    role = str(message.get("role", "")).lower()
    if role and role != "assistant":
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        rendered = "".join(chunks).strip()
        if rendered:
            return rendered
    return ""


def _render_coordination_message(message: object) -> str:
    if isinstance(message, str):
        return message.strip()

    sender = _safe_attr(message, "sender")
    receiver = _safe_attr(message, "receiver")
    message_type = _safe_attr(message, "message_type")
    priority = _safe_attr(message, "priority")
    content = _safe_attr(message, "content")
    related_id = _safe_attr(message, "related_id")

    if not content:
        return ""

    envelope = f"{sender}->{receiver} [{message_type}] ({priority})"
    if related_id:
        envelope += f" related={related_id}"
    return f"{envelope}: {content}".strip()


def _safe_attr(value: object, name: str) -> str:
    raw = getattr(value, name, "")
    return str(raw).strip() if raw is not None else ""


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
