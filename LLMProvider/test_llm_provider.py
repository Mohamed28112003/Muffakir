"""
Test suite for the LLMProvider module (pytest).

All ``langchain_*`` SDK packages are faked via monkeypatched sys.modules, so
no network access or real API keys are required. Validates argument checking,
provider dispatch + kwarg forwarding for all eight providers, get_llm/call
behavior, error messages, and the no-basicConfig logging regression.
"""

import logging
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLMProvider import LLMProvider as PackageReexportedLLMProvider
from LLMProvider.LLMProvider import LLMProvider
from LLMProvider.base import BaseLLMProvider
from Muffakir.Enums import ProviderName
from Muffakir.exceptions import ProviderError, ConfigurationError


# ---------------------------------------------------------------------------
# Fake langchain SDK plumbing
# ---------------------------------------------------------------------------
class FakeChatModel:
    """Records constructor kwargs; supports invoke()/bind(); callable."""

    last_kwargs = None
    last_instance = None

    def __init__(self, **kwargs):
        cls = type(self)
        cls.last_kwargs = dict(kwargs)
        cls.last_instance = self
        self.bound = None

    def bind(self, **kwargs):
        self.bound = kwargs  # HallucinationsCheck-style bind must not fail
        return _BoundFake(self, kwargs)

    def invoke(self, prompt, **kw):
        return f"invoked:{prompt}"

    def __call__(self, *args, **kw):
        return f"called:{args[0] if args else ''}"


class _BoundFake:
    def __init__(self, parent, bound):
        self._parent, self._bound = parent, bound


def _install_fake_langchain(monkeypatch):
    """Register fake SDK classes; returns {key: class}."""
    fakes = {}

    def register(module_name, class_name, key):
        mod = sys.modules.get(module_name)
        if mod is None or getattr(mod, "__faked", False) is not True:
            mod = types.ModuleType(module_name)
            mod.__faked = True
            monkeypatch.setitem(sys.modules, module_name, mod)
        cls = type(class_name, (FakeChatModel,), {})
        setattr(mod, class_name, cls)
        fakes[key] = cls

    register("langchain_groq", "ChatGroq", "groq")
    register("langchain_openai", "ChatOpenAI", "openai")
    register("langchain_openai", "AzureChatOpenAI", "azure")
    register("langchain_anthropic", "ChatAnthropic", "anthropic")
    register("langchain_google_genai", "ChatGoogleGenerativeAI", "gemini")
    register("langchain_ollama", "ChatOllama", "ollama")
    return fakes


def make_provider(fakes, provider=ProviderName.OPENAI, model="test-model",
                  api_key="k", temperature=0.2, max_tokens=123, **extra):
    p = LLMProvider(api_key=api_key, provider=provider, model=model,
                    temperature=temperature, max_tokens=max_tokens, **extra)
    return p


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="API key must be provided"):
        LLMProvider(api_key="", provider=ProviderName.OPENAI, model="m")


def test_ollama_needs_no_api_key(monkeypatch):
    _install_fake_langchain(monkeypatch)
    p = LLMProvider(api_key="", provider=ProviderName.OLLAMA, model="llama3")
    assert p.get_llm() is not None


def test_unsupported_provider_message(monkeypatch):
    _install_fake_langchain(monkeypatch)
    with pytest.raises(ValueError) as excinfo:
        LLMProvider(api_key="k", provider="bogus", model="m")
    msg = str(excinfo.value)
    assert "Unsupported provider" in msg and "GROQ" in msg


# ---------------------------------------------------------------------------
# Provider dispatch + kwarg forwarding
# ---------------------------------------------------------------------------
def test_dispatch_openai(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.OPENAI, model="gpt-test",
                  api_key="sk")
    kw = fakes["openai"].last_kwargs
    assert kw["openai_api_key"] == "sk"
    assert kw["model"] == "gpt-test"
    assert kw["temperature"] == 0.2
    assert kw["max_tokens"] == 123


def test_dispatch_groq(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.GROQ, model="llama-3",
                  api_key="gk")
    kw = fakes["groq"].last_kwargs
    assert kw["api_key"] == "gk" and kw["model"] == "llama-3"


def test_dispatch_anthropic(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.ANTHROPIC, model="claude-x")
    kw = fakes["anthropic"].last_kwargs
    assert kw["api_key"] == "k" and kw["model"] == "claude-x"


def test_together_base_url(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.TOGETHER, model="t-model")
    kw = fakes["openai"].last_kwargs
    assert kw["openai_api_base"] == "https://api.together.xyz/v1"
    assert kw["model"] == "t-model"


def test_openrouter_base_url(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.OPENROUTER, model="o-model")
    kw = fakes["openai"].last_kwargs
    assert kw["openai_api_base"] == "https://openrouter.ai/api/v1"


def test_gemini_uses_max_output_tokens(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.GEMINI, model="gemini-pro")
    kw = fakes["gemini"].last_kwargs
    assert kw["google_api_key"] == "k"
    assert "max_tokens" not in kw
    assert kw["max_output_tokens"] == 123


# ---------------------------------------------------------------------------
# Azure OpenAI specifics
# ---------------------------------------------------------------------------
def test_azure_defaults(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.AZURE_OPENAI,
                  model="my-model", azure_endpoint="https://x.openai.azure.com")
    kw = fakes["azure"].last_kwargs
    # deployment defaults to model; api_version has a pinned default
    assert kw["azure_deployment"] == "my-model"
    assert kw["api_version"] == "2024-02-15-preview"
    assert kw["azure_endpoint"] == "https://x.openai.azure.com"


def test_azure_explicit_deployment_and_version(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.AZURE_OPENAI,
                  model="m", api_version="2024-06-01",
                  azure_endpoint="https://e", azure_deployment="dep-1")
    kw = fakes["azure"].last_kwargs
    assert kw["azure_deployment"] == "dep-1"
    assert kw["api_version"] == "2024-06-01"


def test_azure_missing_endpoint_raises(monkeypatch):
    _install_fake_langchain(monkeypatch)
    with pytest.raises(ValueError, match="azure_endpoint is required"):
        LLMProvider(api_key="k", provider=ProviderName.AZURE_OPENAI, model="m")


def test_azure_missing_endpoint_raises_configuration_error(monkeypatch):
    _install_fake_langchain(monkeypatch)
    with pytest.raises(ConfigurationError, match="azure_endpoint is required"):
        LLMProvider(api_key="k", provider=ProviderName.AZURE_OPENAI, model="m")


def test_azure_accepts_endpoint_alias(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    make_provider(fakes, provider=ProviderName.AZURE_OPENAI,
                  model="m", endpoint="https://alias")
    assert fakes["azure"].last_kwargs["azure_endpoint"] == "https://alias"


# ---------------------------------------------------------------------------
# Ollama specifics
# ---------------------------------------------------------------------------
def test_ollama_default_model_and_url(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    LLMProvider(api_key="", provider=ProviderName.OLLAMA, model="",
                max_tokens=77)
    kw = fakes["ollama"].last_kwargs
    assert kw["model"] == "llama3.2"
    assert kw["base_url"] == "http://localhost:11434"
    # Ollama maps max_tokens -> num_predict
    assert kw["num_predict"] == 77


def test_ollama_base_url_from_extra(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    LLMProvider(api_key="", provider=ProviderName.OLLAMA, model="llama3",
                base_url="http://custom:9999")
    assert fakes["ollama"].last_kwargs["base_url"] == "http://custom:9999"


def test_ollama_alt_base_url_key(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    LLMProvider(api_key="", provider=ProviderName.OLLAMA, model="llama3",
                ollama_base_url="http://alt:11434")
    assert fakes["ollama"].last_kwargs["base_url"] == "http://alt:11434"


# ---------------------------------------------------------------------------
# get_llm / call behavior
# ---------------------------------------------------------------------------
def test_get_llm_returns_constructed_instance(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)
    assert p.get_llm() is fakes["openai"].last_instance


def test_call_prefers_invoke(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)
    assert p.call("hello") == "invoked:hello"


def test_call_falls_back_to_direct_call(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)
    p.llm = lambda *a, **k: "direct-call"  # no .invoke attribute
    assert p.call("hi") == "direct-call"


def test_call_logs_and_reraises(monkeypatch, caplog):
    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("down")

    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)
    p.llm = Boom()
    with caplog.at_level(logging.ERROR), pytest.raises(ProviderError, match="down") as excinfo:
        p.call("x")
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert any("LLM call failed" in r.message for r in caplog.records)


def test_call_redacts_api_key_from_log_and_exception(monkeypatch, caplog):
    secret = "sk-real-secret-789"

    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError(f"AuthenticationError: invalid api key {secret}")

    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes, api_key=secret)
    p.llm = Boom()
    with caplog.at_level(logging.ERROR), pytest.raises(ProviderError) as excinfo:
        p.call("x")
    assert secret not in str(excinfo.value)
    assert secret not in "\n".join(r.message for r in caplog.records)


def test_bind_supported_on_fake_models(monkeypatch):
    """Guard: bind() must work (relied on by the HallucinationsCheck P1 fix)."""
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)
    bound = p.get_llm().bind(temperature=0.0)
    assert bound is not None and p.get_llm().bound == {"temperature": 0.0}


def test_import_error_when_sdk_missing(monkeypatch):
    # A None entry in sys.modules forces ImportError even if the real SDK is
    # installed in site-packages.
    monkeypatch.setitem(sys.modules, "langchain_groq", None)
    with pytest.raises(ImportError, match=r'Muffakir\[groq\]'):
        LLMProvider(api_key="k", provider=ProviderName.GROQ, model="m")


# ---------------------------------------------------------------------------
# Package surface + logging regression
# ---------------------------------------------------------------------------
def test_package_reexports():
    import LLMProvider as pkg
    assert pkg.LLMProvider is LLMProvider
    assert hasattr(pkg, "create_llm_provider")
    assert hasattr(pkg, "CustomOpenAILLMProvider")
    for name in ("BaseLLMProvider", "GroqLLMProvider", "OllamaLLMProvider",
                 "AzureOpenAILLMProvider"):
        assert hasattr(pkg, name)
    assert "LLMProvider" in pkg.__all__
    assert "create_llm_provider" in pkg.__all__


def test_dispatch_custom_openai(monkeypatch):
    from LLMProvider import create_llm_provider
    fakes = _install_fake_langchain(monkeypatch)
    p = create_llm_provider(
        provider="custom",
        api_key="custom_sk",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
        max_tokens=500,
    )
    kw = fakes["openai"].last_kwargs
    assert kw["openai_api_key"] == "custom_sk"
    assert kw["model"] == "deepseek-chat"
    assert kw["base_url"] == "https://api.deepseek.com/v1"
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 500


def test_custom_openai_normalizes_null_role_without_mutating_response():
    from LLMProvider.custom import _normalize_null_message_roles

    response = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": None, "content": "answer"},
                "finish_reason": "stop",
            }
        ],
    }

    normalized = _normalize_null_message_roles(response)

    assert normalized is not response
    assert normalized["choices"][0]["message"]["role"] == "assistant"
    assert response["choices"][0]["message"]["role"] is None


def test_custom_openai_compatibility_model_applies_role_normalization(monkeypatch):
    from LLMProvider import create_llm_provider

    class _ConvertingFakeChatModel(FakeChatModel):
        def _create_chat_result(self, response, generation_info=None):
            return response

    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = _ConvertingFakeChatModel
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    provider = create_llm_provider(
        provider="custom",
        api_key="custom-key",
        model="custom-model",
        base_url="https://example.test/v1",
    )
    result = provider.get_llm()._create_chat_result(
        {"choices": [{"message": {"role": None, "content": "answer"}}]}
    )

    assert result["choices"][0]["message"]["role"] == "assistant"


def test_string_provider_dispatch(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = LLMProvider(api_key="k", provider="groq", model="llama-3")
    assert p.provider == ProviderName.GROQ
    p2 = LLMProvider(api_key="k", provider="deepseek", model="deepseek-v3", base_url="https://api.deepseek.com")
    assert p2.provider == ProviderName.CUSTOM


def test_facade_stores_config(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes, temperature=0.7, max_tokens=55)
    assert (p.temperature, p.max_tokens) == (0.7, 55)
    assert isinstance(p.llm, FakeChatModel)


def test_no_basicconfig_regression():
    """P1 regression: library code must not mutate root logger config."""
    monkey = pytest.MonkeyPatch()
    try:
        root_before = list(logging.getLogger().handlers)
        level_before = logging.getLogger().level
        fakes = _install_fake_langchain(monkey)
        make_provider(fakes, provider=ProviderName.GROQ)
        LLMProvider(api_key="", provider=ProviderName.OLLAMA, model="m2")
        assert list(logging.getLogger().handlers) == root_before
        assert logging.getLogger().level == level_before
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# Usage-tracking callback attachment
# ---------------------------------------------------------------------------
def test_usage_callback_attached_to_constructed_llm(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)
    assert p._usage_callback in p.llm.callbacks


def test_get_usage_totals_reflects_recorded_calls(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)

    class FakeResponse:
        llm_output = {"token_usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}}
        generations = [[]]

    p._usage_callback.on_llm_end(FakeResponse())
    assert p.get_usage_totals() == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


def test_reset_usage_clears_totals(monkeypatch):
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)

    class FakeResponse:
        llm_output = {"token_usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}}
        generations = [[]]

    p._usage_callback.on_llm_end(FakeResponse())
    p.reset_usage()
    assert p.get_usage_totals() == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_get_per_sample_usage_totals_passthrough(monkeypatch):
    from Trace.context import set_current_sample, clear_current_sample

    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)

    class FakeResponse:
        llm_output = {"token_usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
        generations = [[]]

    set_current_sample(0, 0)
    try:
        p._usage_callback.on_llm_end(FakeResponse())
    finally:
        clear_current_sample()

    assert p.get_per_sample_usage_totals()[(0, 0)] == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    assert p.get_sample_usage_totals(0, 0) == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


def test_get_llm_still_returns_same_instance_after_callback_attach(monkeypatch):
    """Regression: attaching the usage callback must not change get_llm()'s identity contract."""
    fakes = _install_fake_langchain(monkeypatch)
    p = make_provider(fakes)
    assert p.get_llm() is fakes["openai"].last_instance


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
