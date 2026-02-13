import json
import os
import unittest
from unittest.mock import patch

from colophon.llm import AnthropicClient, LLMConfig, LLMError, OpenAICompatibleClient, create_llm_client


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class LLMTests(unittest.TestCase):
    def test_create_llm_client_returns_none_when_disabled(self) -> None:
        config = LLMConfig(provider="none")
        self.assertIsNone(create_llm_client(config))

    def test_create_llm_client_supports_provider_aliases(self) -> None:
        openai_client = create_llm_client(LLMConfig(provider="openai", model="gpt-test"))
        anthropic_client = create_llm_client(LLMConfig(provider="claude", model="claude-test"))
        copilot_client = create_llm_client(LLMConfig(provider="copilot", model="gpt-test"))

        self.assertIsInstance(openai_client, OpenAICompatibleClient)
        self.assertIsInstance(anthropic_client, AnthropicClient)
        self.assertIsInstance(copilot_client, OpenAICompatibleClient)

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


if __name__ == "__main__":
    unittest.main()
