"""
Tests for pre-call token counting via provider.count_tokens().

Covers:
- Anthropic exact counting via messages.count_tokens API
- OpenAI/DeepInfra tiktoken estimation
- BaseProvider default (returns None)
- Failure resilience (errors never block the flow)
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.providers.base import BaseProvider, ProviderResponse


# --- Fixtures ---

SAMPLE_MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]
SAMPLE_TOOLS = [
    {
        "name": "search",
        "description": "Search documents",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
]
SAMPLE_SYSTEM = "You are a helpful assistant."


# --- BaseProvider default ---


class ConcreteProvider(BaseProvider):
    """Minimal concrete subclass for testing default count_tokens."""

    def create_message(self, messages, tools, system, max_tokens, temperature, **kwargs):
        raise NotImplementedError

    def stream_message(self, messages, tools, system, max_tokens, temperature, **kwargs):
        raise NotImplementedError

    def supports_feature(self, feature):
        return False

    def get_model_name(self):
        return "test-model"

    def set_model(self, model):
        pass

    def get_provider_name(self):
        return "test"


class TestBaseProviderCountTokens:
    """Test default count_tokens behavior on BaseProvider."""

    def test_base_provider_count_tokens_returns_none(self):
        provider = ConcreteProvider()
        result = provider.count_tokens(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
        assert result is None

    def test_tiktoken_estimate_returns_int(self):
        result = BaseProvider._tiktoken_estimate(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
        assert isinstance(result, int)
        assert result > 0

    def test_tiktoken_estimate_with_structured_system(self):
        structured_system = [
            {"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}}
        ]
        result = BaseProvider._tiktoken_estimate(SAMPLE_MESSAGES, SAMPLE_TOOLS, structured_system)
        assert isinstance(result, int)
        assert result > 0

    def test_tiktoken_estimate_empty_inputs(self):
        result = BaseProvider._tiktoken_estimate([], [], "")
        assert result == 0

    def test_tiktoken_estimate_with_list_content_messages(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "tool_result", "tool_use_id": "123", "content": "result text"},
                ],
            }
        ]
        result = BaseProvider._tiktoken_estimate(messages, [], "System prompt")
        assert isinstance(result, int)
        assert result > 0


# --- Anthropic exact counting ---


class TestAnthropicCountTokens:
    """Test Anthropic provider's exact token counting via API."""

    def test_anthropic_count_tokens_returns_exact_count(self):
        from src.agent.providers.anthropic_provider import AnthropicProvider

        with patch("anthropic.Anthropic") as MockClient:
            mock_instance = MockClient.return_value
            # Mock count_tokens response
            mock_count_result = MagicMock()
            mock_count_result.input_tokens = 42
            mock_instance.messages.count_tokens.return_value = mock_count_result

            # Also need to patch wrap_anthropic to return the mock
            with patch(
                "src.agent.providers.anthropic_provider.wrap_anthropic",
                return_value=mock_instance,
            ):
                provider = AnthropicProvider(api_key="sk-ant-test-key", model="claude-haiku-4-5")

            result = provider.count_tokens(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
            assert result == 42
            mock_instance.messages.count_tokens.assert_called_once()

    def test_anthropic_count_tokens_returns_none_on_error(self):
        from src.agent.providers.anthropic_provider import AnthropicProvider

        with patch("anthropic.Anthropic") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.messages.count_tokens.side_effect = Exception("API error")

            with patch(
                "src.agent.providers.anthropic_provider.wrap_anthropic",
                return_value=mock_instance,
            ):
                provider = AnthropicProvider(api_key="sk-ant-test-key", model="claude-haiku-4-5")

            result = provider.count_tokens(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
            assert result is None

    def test_anthropic_count_tokens_excludes_empty_tools(self):
        """Verify empty tools list is not passed to API (matches create_message pattern)."""
        from src.agent.providers.anthropic_provider import AnthropicProvider

        with patch("anthropic.Anthropic") as MockClient:
            mock_instance = MockClient.return_value
            mock_count_result = MagicMock()
            mock_count_result.input_tokens = 10
            mock_instance.messages.count_tokens.return_value = mock_count_result

            with patch(
                "src.agent.providers.anthropic_provider.wrap_anthropic",
                return_value=mock_instance,
            ):
                provider = AnthropicProvider(api_key="sk-ant-test-key", model="claude-haiku-4-5")

            provider.count_tokens(SAMPLE_MESSAGES, [], SAMPLE_SYSTEM)

            call_kwargs = mock_instance.messages.count_tokens.call_args[1]
            assert "tools" not in call_kwargs


# --- OpenAI tiktoken estimation ---


class TestOpenAICountTokens:
    """Test OpenAI provider's tiktoken-based estimation."""

    def test_openai_count_tokens_returns_estimate(self):
        from src.agent.providers.openai_provider import OpenAIProvider

        with patch("openai.OpenAI"):
            with patch(
                "src.agent.providers.openai_provider.wrap_openai",
                return_value=MagicMock(),
            ):
                provider = OpenAIProvider(api_key="sk-test-key", model="gpt-4o-mini")

        result = provider.count_tokens(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
        assert isinstance(result, int)
        assert result > 0


# --- DeepInfra tiktoken estimation ---


class TestDeepInfraCountTokens:
    """Test DeepInfra provider's tiktoken-based estimation."""

    def test_deepinfra_count_tokens_returns_estimate(self):
        from src.agent.providers.deepinfra_provider import DeepInfraProvider

        with patch("src.agent.providers.deepinfra_provider.wrap_openai", return_value=MagicMock()):
            with patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}):
                provider = DeepInfraProvider(
                    api_key="test-key",
                    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                )

        result = provider.count_tokens(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
        assert isinstance(result, int)
        assert result > 0


# --- Integration: failure does not block tool loop ---


class TestCountTokensResilience:
    """Verify count_tokens failures never block the agent tool loop."""

    def test_count_tokens_failure_does_not_raise(self):
        """Even if count_tokens raises internally, it returns None (not exception)."""
        from src.agent.providers.anthropic_provider import AnthropicProvider

        with patch("anthropic.Anthropic") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.messages.count_tokens.side_effect = RuntimeError("Network timeout")

            with patch(
                "src.agent.providers.anthropic_provider.wrap_anthropic",
                return_value=mock_instance,
            ):
                provider = AnthropicProvider(api_key="sk-ant-test-key", model="claude-haiku-4-5")

            # Should return None, not raise
            result = provider.count_tokens(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
            assert result is None

    def test_tiktoken_import_failure_returns_none(self):
        """If tiktoken is not installed, _tiktoken_estimate returns None."""
        with patch.dict("sys.modules", {"tiktoken": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'tiktoken'")):
                result = BaseProvider._tiktoken_estimate(SAMPLE_MESSAGES, SAMPLE_TOOLS, SAMPLE_SYSTEM)
                assert result is None
