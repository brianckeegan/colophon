import json
import os
import unittest
from unittest.mock import patch

from colophon.llm import AnthropicClient, LLMConfig, LLMError, OpenAICompatibleClient, PiMonoClient, create_llm_client


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakePipeReader:
    def __init__(self, process: "_FakePiProcess") -> None:
        self._process = process

    def readline(self) -> str:
        if self._process.output_lines:
            return self._process.output_lines.pop(0)
        return ""


class _FakePipeWriter:
    def __init__(self, process: "_FakePiProcess") -> None:
        self._process = process

    def write(self, data: str) -> int:
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            self._process.commands.append(payload)
            command_type = payload.get("type")
            if command_type == "prompt":
                request_id = payload.get("id")
                if self._process.prompt_success:
                    self._process.output_lines.extend(
                        [
                            json.dumps(
                                {
                                    "id": request_id,
                                    "type": "response",
                                    "command": "prompt",
                                    "success": True,
                                }
                            )
                            + "\n",
                            json.dumps(
                                {
                                    "type": "message_update",
                                    "assistantMessageEvent": {"type": "text_delta", "delta": "Hello from pi"},
                                }
                            )
                            + "\n",
                            json.dumps(
                                {
                                    "type": "agent_end",
                                    "messages": [{"role": "assistant", "content": [{"type": "text", "text": "Hello from pi"}]}],
                                }
                            )
                            + "\n",
                        ]
                    )
                else:
                    self._process.output_lines.append(
                        json.dumps(
                            {
                                "id": request_id,
                                "type": "response",
                                "command": "prompt",
                                "success": False,
                                "error": "boom",
                            }
                        )
                        + "\n"
                    )
        return len(data)

    def flush(self) -> None:
        return None


class _FakePipeErr:
    def read(self) -> str:
        return ""


class _FakePiProcess:
    def __init__(self, prompt_success: bool = True) -> None:
        self.prompt_success = prompt_success
        self.output_lines: list[str] = []
        self.commands: list[dict] = []
        self.stdin = _FakePipeWriter(self)
        self.stdout = _FakePipeReader(self)
        self.stderr = _FakePipeErr()

    def poll(self):
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> None:
        return None

    def kill(self) -> None:
        return None


class LLMTests(unittest.TestCase):
    def test_create_llm_client_returns_none_when_disabled(self) -> None:
        config = LLMConfig(provider="none")
        self.assertIsNone(create_llm_client(config))

    def test_create_llm_client_supports_provider_aliases(self) -> None:
        openai_client = create_llm_client(LLMConfig(provider="openai", model="gpt-test"))
        anthropic_client = create_llm_client(LLMConfig(provider="claude", model="claude-test"))
        copilot_client = create_llm_client(LLMConfig(provider="copilot", model="gpt-test"))
        pi_client = create_llm_client(LLMConfig(provider="pi", model="anthropic/claude-test"))

        self.assertIsInstance(openai_client, OpenAICompatibleClient)
        self.assertIsInstance(anthropic_client, AnthropicClient)
        self.assertIsInstance(copilot_client, OpenAICompatibleClient)
        self.assertIsInstance(pi_client, PiMonoClient)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("colophon.llm.request.urlopen")
    def test_openai_compatible_generate_parses_text(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {"choices": [{"message": {"content": "Generated text"}}]}
        )
        client = OpenAICompatibleClient(
            LLMConfig(provider="openai", model="gpt-test", api_base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
        )

        output = client.generate(prompt="Hello")

        self.assertEqual(output, "Generated text")
        self.assertTrue(mock_urlopen.called)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False)
    @patch("colophon.llm.request.urlopen")
    def test_anthropic_generate_parses_text(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {"content": [{"type": "text", "text": "Claude output"}]}
        )
        client = AnthropicClient(
            LLMConfig(
                provider="anthropic",
                model="claude-test",
                api_base_url="https://api.anthropic.com/v1",
                api_key_env="ANTHROPIC_API_KEY",
            )
        )

        output = client.generate(prompt="Hello")

        self.assertEqual(output, "Claude output")
        self.assertTrue(mock_urlopen.called)

    @patch.dict(os.environ, {}, clear=True)
    def test_generate_raises_without_api_key(self) -> None:
        client = OpenAICompatibleClient(
            LLMConfig(provider="openai", model="gpt-test", api_base_url="https://api.openai.com/v1", api_key_env="MISSING_KEY")
        )

        with self.assertRaises(LLMError):
            client.generate(prompt="Hello")

    @patch("colophon.llm.subprocess.Popen")
    @patch("colophon.llm.shutil.which")
    def test_pi_mono_generate_parses_rpc_events(self, mock_which, mock_popen) -> None:
        process = _FakePiProcess(prompt_success=True)
        mock_which.return_value = "/usr/local/bin/pi"
        mock_popen.return_value = process
        client = PiMonoClient(
            LLMConfig(
                provider="pi",
                model="anthropic/claude-test",
                timeout_seconds=2.0,
            )
        )
        client.record_coordination_message("section_coordinator->claim_author_agent [guidance] (normal): stay concise")

        result = client.generate("Draft a paragraph.")

        self.assertEqual(result, "Hello from pi")
        prompt_command = next(command for command in process.commands if command.get("type") == "prompt")
        self.assertIn("Coordination context:", prompt_command.get("message", ""))

    @patch("colophon.llm.shutil.which")
    def test_pi_mono_generate_raises_when_binary_missing(self, mock_which) -> None:
        mock_which.return_value = None
        client = PiMonoClient(LLMConfig(provider="pi", model="anthropic/claude-test"))
        with self.assertRaises(LLMError):
            client.generate("Draft a paragraph.")

    @patch("colophon.llm.subprocess.Popen")
    @patch("colophon.llm.shutil.which")
    def test_pi_mono_generate_raises_on_rpc_error(self, mock_which, mock_popen) -> None:
        process = _FakePiProcess(prompt_success=False)
        mock_which.return_value = "/usr/local/bin/pi"
        mock_popen.return_value = process
        client = PiMonoClient(LLMConfig(provider="pi", model="anthropic/claude-test", timeout_seconds=1.0))

        with self.assertRaises(LLMError):
            client.generate("Draft a paragraph.")


if __name__ == "__main__":
    unittest.main()
