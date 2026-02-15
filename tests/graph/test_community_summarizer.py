"""Tests for CommunitySummarizer._parse_response (JSON parsing logic)."""

from unittest.mock import MagicMock, patch

import pytest

from src.graph.community_summarizer import CommunitySummarizer


def _make_summarizer():
    """Create a CommunitySummarizer with mocked provider and prompt."""
    with patch.object(CommunitySummarizer, "__init__", lambda self, *a, **kw: None):
        s = CommunitySummarizer.__new__(CommunitySummarizer)
        s.provider = MagicMock()
        s._prompt = "test prompt"
        return s


class TestParseResponse:
    def test_valid_json(self):
        s = _make_summarizer()
        result = s._parse_response('{"title": "Nuclear Safety", "description": "Group of entities"}')
        assert result == ("Nuclear Safety", "Group of entities")

    def test_json_with_code_fences(self):
        s = _make_summarizer()
        result = s._parse_response('```json\n{"title": "Test", "description": "Desc"}\n```')
        assert result == ("Test", "Desc")

    def test_title_truncated_to_100(self):
        s = _make_summarizer()
        long_title = "A" * 200
        result = s._parse_response(f'{{"title": "{long_title}", "description": "Desc"}}')
        assert result is not None
        assert len(result[0]) == 100

    def test_empty_string(self):
        s = _make_summarizer()
        assert s._parse_response("") is None

    def test_none_text(self):
        s = _make_summarizer()
        assert s._parse_response(None) is None

    def test_invalid_json(self):
        s = _make_summarizer()
        assert s._parse_response("not json at all") is None

    def test_non_dict_json(self):
        s = _make_summarizer()
        assert s._parse_response('["title", "desc"]') is None

    def test_missing_title(self):
        s = _make_summarizer()
        assert s._parse_response('{"description": "Only desc"}') is None

    def test_missing_description(self):
        s = _make_summarizer()
        assert s._parse_response('{"title": "Only title"}') is None

    def test_empty_title(self):
        s = _make_summarizer()
        assert s._parse_response('{"title": "", "description": "Desc"}') is None

    def test_whitespace_title(self):
        s = _make_summarizer()
        assert s._parse_response('{"title": "  ", "description": "Desc"}') is None
