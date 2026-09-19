"""
Unit tests for the WebSearch module.

Covers:
- WebSearchResult contract
- create_web_search_provider factory and alias support
- FirecrawlWebSearchProvider: initialization, search, source parsing, error handling
- TavilyWebSearchProvider: initialization, search, normalization, error handling
- SerpAPIWebSearchProvider: initialization, structured search, fallback search, error handling
"""

import pytest
from unittest.mock import MagicMock, patch

from WebSearch.models import WebSearchResult
from WebSearch.factory import create_web_search_provider
from WebSearch.firecrawl import FirecrawlWebSearchProvider
from WebSearch.tavily import TavilyWebSearchProvider
from WebSearch.serpapi import SerpAPIWebSearchProvider


# ---------------------------------------------------------------------------
# Models & Contract Tests
# ---------------------------------------------------------------------------

def test_web_search_result_model():
    res = WebSearchResult(
        content="محتوى البحث",
        sources=[{"title": "عنوان", "url": "https://example.com"}]
    )
    assert res.content == "محتوى البحث"
    assert len(res.sources) == 1
    assert res.sources[0]["title"] == "عنوان"


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------

def test_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown web search provider"):
        create_web_search_provider("non_existent_provider")


def test_factory_firecrawl_aliases():
    for alias in ("firecrawl", "fire_crawl", "fire-crawl"):
        mock_fc = MagicMock()
        with patch.dict("sys.modules", {"firecrawl": mock_fc}):
            provider = create_web_search_provider(alias, api_key="fc_test_key")
            assert isinstance(provider, FirecrawlWebSearchProvider)


def test_factory_tavily():
    mock_tvly = MagicMock()
    with patch.dict("sys.modules", {"langchain_tavily": mock_tvly}):
        provider = create_web_search_provider("tavily", api_key="tvly_test_key")
        assert isinstance(provider, TavilyWebSearchProvider)


def test_factory_serpapi_aliases():
    for alias in ("serpapi", "serp_api", "serp-api", "google"):
        mock_serp = MagicMock()
        with patch("Muffakir.optional_dependencies.require_optional_dependency"):
            with patch.dict("sys.modules", {"langchain_community.utilities": mock_serp}):
                provider = create_web_search_provider(alias, api_key="serp_test_key")
                assert isinstance(provider, SerpAPIWebSearchProvider)


# ---------------------------------------------------------------------------
# FirecrawlWebSearchProvider Tests
# ---------------------------------------------------------------------------

def test_firecrawl_missing_api_key():
    with pytest.raises(ValueError, match="api_key is required"):
        FirecrawlWebSearchProvider(api_key="")


def test_firecrawl_search_success():
    mock_app_instance = MagicMock()
    mock_app_instance.deep_research.return_value = {
        "data": {
            "finalAnalysis": "التقرير النهائي للأبحاث",
            "sources": [
                {"title": "مصدر أول", "url": "https://source1.com"},
                {"title": "مصدر ثاني", "url": "https://source2.com"},
            ]
        }
    }
    mock_module = MagicMock(FirecrawlApp=MagicMock(return_value=mock_app_instance))

    with patch.dict("sys.modules", {"firecrawl": mock_module}):
        provider = FirecrawlWebSearchProvider(api_key="fc_key", max_depth=1, time_limit=10, max_urls=3)
        res = provider.search("تاريخ الذكاء الاصطناعي")

        assert isinstance(res, WebSearchResult)
        assert res.content == "التقرير النهائي للأبحاث"
        assert len(res.sources) == 2
        assert res.sources[0]["title"] == "مصدر أول"


def test_firecrawl_search_error():
    mock_app_instance = MagicMock()
    mock_app_instance.deep_research.side_effect = RuntimeError("Firecrawl API Error")
    mock_module = MagicMock(FirecrawlApp=MagicMock(return_value=mock_app_instance))

    with patch.dict("sys.modules", {"firecrawl": mock_module}):
        provider = FirecrawlWebSearchProvider(api_key="fc_key")
        with pytest.raises(RuntimeError, match="Firecrawl API Error"):
            provider.search("استعلام فاشل")


# ---------------------------------------------------------------------------
# TavilyWebSearchProvider Tests
# ---------------------------------------------------------------------------

def test_tavily_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="api_key is required"):
            TavilyWebSearchProvider(api_key="")


def test_tavily_search_list_results():
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = [
        {
            "title": "مقال الذكاء الاصطناعي",
            "url": "https://ai.org",
            "content": "الذكاء الاصطناعي يتقدم بسرعة."
        },
        {
            "title": "الروبوتات",
            "url": "https://robots.org",
            "snippet": "الروبوتات تعتمد على الذكاء الاصطناعي."
        }
    ]
    mock_module = MagicMock(TavilySearchResults=MagicMock(return_value=mock_tool))

    with patch.dict("sys.modules", {"langchain_tavily": mock_module}):
        provider = TavilyWebSearchProvider(api_key="tvly_key")
        res = provider.search("ما هو الذكاء الاصطناعي؟")

        assert isinstance(res, WebSearchResult)
        assert "الذكاء الاصطناعي يتقدم بسرعة" in res.content
        assert "الروبوتات تعتمد على الذكاء الاصطناعي" in res.content
        assert len(res.sources) == 2
        assert res.sources[0]["url"] == "https://ai.org"


def test_tavily_search_string_result():
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "إجابة مباشرة كنص بسيط"
    mock_module = MagicMock(TavilySearchResults=MagicMock(return_value=mock_tool))

    with patch.dict("sys.modules", {"langchain_tavily": mock_module}):
        provider = TavilyWebSearchProvider(api_key="tvly_key")
        res = provider.search("سؤال سريع")

        assert res.content == "إجابة مباشرة كنص بسيط"
        assert res.sources == []


def test_tavily_search_error():
    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = ConnectionError("Tavily unreachable")
    mock_module = MagicMock(TavilySearchResults=MagicMock(return_value=mock_tool))

    with patch.dict("sys.modules", {"langchain_tavily": mock_module}):
        provider = TavilyWebSearchProvider(api_key="tvly_key")
        with pytest.raises(ConnectionError, match="Tavily unreachable"):
            provider.search("استعلام فاشل")


# ---------------------------------------------------------------------------
# SerpAPIWebSearchProvider Tests
# ---------------------------------------------------------------------------

def test_serpapi_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="api_key is required"):
            SerpAPIWebSearchProvider(api_key="")


def test_serpapi_search_structured_results():
    mock_wrapper = MagicMock()
    mock_wrapper.results.return_value = {
        "answer_box": {
            "answer": "عاصمة المملكة العربية السعودية هي الرياض."
        },
        "organic_results": [
            {
                "title": "الرياض - ويكيبيديا",
                "link": "https://ar.wikipedia.org/wiki/Riyadh",
                "snippet": "الرياض هي العاصمة الإدارية والسياسية للمملكة العربية السعودية."
            }
        ]
    }
    mock_module = MagicMock(SerpAPIWrapper=MagicMock(return_value=mock_wrapper))

    with patch.dict("sys.modules", {"langchain_community.utilities": mock_module}):
        provider = SerpAPIWebSearchProvider(api_key="serp_key")
        res = provider.search("ما هي عاصمة السعودية؟")

        assert isinstance(res, WebSearchResult)
        assert "عاصمة المملكة العربية السعودية هي الرياض" in res.content
        assert "الرياض هي العاصمة الإدارية" in res.content
        assert len(res.sources) == 1
        assert res.sources[0]["url"] == "https://ar.wikipedia.org/wiki/Riyadh"


def test_serpapi_search_fallback_run():
    mock_wrapper = MagicMock(spec=["run"])
    mock_wrapper.run.return_value = "الرياض هي عاصمة المملكة العربية السعودية."
    mock_module = MagicMock(SerpAPIWrapper=MagicMock(return_value=mock_wrapper))

    with patch.dict("sys.modules", {"langchain_community.utilities": mock_module}):
        provider = SerpAPIWebSearchProvider(api_key="serp_key")
        res = provider.search("عاصمة السعودية")

        assert res.content == "الرياض هي عاصمة المملكة العربية السعودية."
        assert res.sources == []


def test_serpapi_search_error():
    mock_wrapper = MagicMock()
    mock_wrapper.results.side_effect = TimeoutError("SerpAPI timed out")
    mock_module = MagicMock(SerpAPIWrapper=MagicMock(return_value=mock_wrapper))

    with patch.dict("sys.modules", {"langchain_community.utilities": mock_module}):
        provider = SerpAPIWebSearchProvider(api_key="serp_key")
        with pytest.raises(TimeoutError, match="SerpAPI timed out"):
            provider.search("استعلام بطيء")
