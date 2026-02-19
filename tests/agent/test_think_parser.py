"""Tests for ThinkTagStreamParser — state machine for <think> tag parsing."""

import pytest

from src.agent.providers.think_parser import (
    ChunkType,
    ParsedChunk,
    ThinkTagStreamParser,
)


class TestBasicParsing:
    """Test basic think tag detection."""

    def test_no_think_tags(self):
        """Plain text without think tags passes through as TEXT."""
        parser = ThinkTagStreamParser()
        result = parser.feed("Hello world")
        assert len(result) == 1
        assert result[0].type == ChunkType.TEXT
        assert result[0].content == "Hello world"

    def test_complete_think_block(self):
        """A complete <think>...</think> block in one chunk."""
        parser = ThinkTagStreamParser()
        result = parser.feed("<think>reasoning here</think>answer text")
        assert len(result) == 2
        assert result[0].type == ChunkType.THINKING
        assert result[0].content == "reasoning here"
        assert result[1].type == ChunkType.TEXT
        assert result[1].content == "answer text"

    def test_think_only(self):
        """Only thinking content, no text after."""
        parser = ThinkTagStreamParser()
        result = parser.feed("<think>just thinking</think>")
        assert len(result) == 1
        assert result[0].type == ChunkType.THINKING
        assert result[0].content == "just thinking"

    def test_empty_think_block(self):
        """Empty think block."""
        parser = ThinkTagStreamParser()
        result = parser.feed("<think></think>text after")
        assert len(result) == 1
        assert result[0].type == ChunkType.TEXT
        assert result[0].content == "text after"

    def test_text_before_think(self):
        """Text before think block."""
        parser = ThinkTagStreamParser()
        result = parser.feed("prefix <think>thinking</think>suffix")
        assert len(result) == 3
        assert result[0].type == ChunkType.TEXT
        assert result[0].content == "prefix "
        assert result[1].type == ChunkType.THINKING
        assert result[1].content == "thinking"
        assert result[2].type == ChunkType.TEXT
        assert result[2].content == "suffix"


class TestChunkSplitting:
    """Test think tags split across multiple chunks."""

    def test_tag_split_in_middle(self):
        """<think> tag split as '<thi' + 'nk>'."""
        parser = ThinkTagStreamParser()
        r1 = parser.feed("<thi")
        assert r1 == []  # Buffering partial tag
        r2 = parser.feed("nk>hello")
        assert len(r2) == 1
        assert r2[0].type == ChunkType.THINKING
        assert r2[0].content == "hello"

    def test_close_tag_split(self):
        """</think> split as '</thi' + 'nk>'."""
        parser = ThinkTagStreamParser()
        parser.feed("<think>")
        r1 = parser.feed("content</thi")
        # "content" should be emitted, "</thi" buffered
        assert any(c.type == ChunkType.THINKING and "content" in c.content for c in r1)
        r2 = parser.feed("nk>after")
        assert any(c.type == ChunkType.TEXT and c.content == "after" for c in r2)

    def test_single_char_chunks(self):
        """Feed one character at a time."""
        parser = ThinkTagStreamParser()
        text = "<think>AB</think>CD"
        all_chunks = []
        for char in text:
            all_chunks.extend(parser.feed(char))
        all_chunks.extend(parser.flush())

        thinking = "".join(c.content for c in all_chunks if c.type == ChunkType.THINKING)
        text_out = "".join(c.content for c in all_chunks if c.type == ChunkType.TEXT)
        assert thinking == "AB"
        assert text_out == "CD"

    def test_open_tag_char_by_char(self):
        """Feed <think> character by character."""
        parser = ThinkTagStreamParser()
        for char in "<think>":
            parser.feed(char)
        result = parser.feed("thought")
        assert len(result) == 1
        assert result[0].type == ChunkType.THINKING
        assert result[0].content == "thought"


class TestEdgeCases:
    """Test edge cases and error recovery."""

    def test_partial_tag_not_matching(self):
        """'<th' followed by non-matching char emits as text."""
        parser = ThinkTagStreamParser()
        r1 = parser.feed("<th")
        assert r1 == []
        r2 = parser.feed("x")
        assert len(r2) == 1
        assert r2[0].type == ChunkType.TEXT
        assert r2[0].content == "<thx"

    def test_truncated_open_tag_at_end(self):
        """Stream ends with partial open tag."""
        parser = ThinkTagStreamParser()
        r1 = parser.feed("hello <thi")
        flush = parser.flush()
        all_text = "".join(
            c.content for r in [r1, flush] for c in r if c.type == ChunkType.TEXT
        )
        assert all_text == "hello <thi"

    def test_truncated_think_block(self):
        """Stream ends inside think block (no closing tag)."""
        parser = ThinkTagStreamParser()
        r1 = parser.feed("<think>unclosed reasoning")
        flush = parser.flush()
        all_thinking = "".join(
            c.content for r in [r1, flush] for c in r if c.type == ChunkType.THINKING
        )
        assert all_thinking == "unclosed reasoning"

    def test_truncated_close_tag(self):
        """Stream ends with partial close tag inside think block."""
        parser = ThinkTagStreamParser()
        parser.feed("<think>")
        r1 = parser.feed("content</th")
        flush = parser.flush()
        all_thinking = "".join(
            c.content for r in [r1, flush] for c in r if c.type == ChunkType.THINKING
        )
        assert "content" in all_thinking
        assert "</th" in all_thinking

    def test_angle_bracket_not_tag(self):
        """'<' followed by non-'t' is text, not a tag."""
        parser = ThinkTagStreamParser()
        result = parser.feed("<div>hello</div>")
        text = "".join(c.content for c in result if c.type == ChunkType.TEXT)
        assert text == "<div>hello</div>"

    def test_less_than_in_think(self):
        """'<' inside think block that doesn't start </think>."""
        parser = ThinkTagStreamParser()
        parser.feed("<think>")
        result = parser.feed("a < b and c > d</think>")
        thinking = "".join(c.content for c in result if c.type == ChunkType.THINKING)
        assert "a < b and c > d" == thinking

    def test_multiple_think_blocks(self):
        """Multiple think blocks in one stream."""
        parser = ThinkTagStreamParser()
        result = parser.feed("<think>first</think>middle<think>second</think>end")
        thinking = [c.content for c in result if c.type == ChunkType.THINKING]
        text = [c.content for c in result if c.type == ChunkType.TEXT]
        assert thinking == ["first", "second"]
        assert text == ["middle", "end"]

    def test_empty_input(self):
        """Empty string produces no output."""
        parser = ThinkTagStreamParser()
        result = parser.feed("")
        assert result == []

    def test_flush_with_no_buffered_content(self):
        """Flush on clean state produces nothing."""
        parser = ThinkTagStreamParser()
        parser.feed("hello")
        assert parser.flush() == []


class TestCoalescing:
    """Test that adjacent same-type chunks are merged."""

    def test_adjacent_text_merged(self):
        """Adjacent text chunks should be merged into one."""
        parser = ThinkTagStreamParser()
        # Feed chars that all resolve to TEXT
        result = parser.feed("abc")
        assert len(result) == 1
        assert result[0].content == "abc"

    def test_adjacent_thinking_merged(self):
        """Adjacent thinking chunks should be merged."""
        parser = ThinkTagStreamParser()
        parser.feed("<think>")
        result = parser.feed("abc")
        assert len(result) == 1
        assert result[0].type == ChunkType.THINKING
        assert result[0].content == "abc"


class TestRealWorldPatterns:
    """Test patterns from actual Qwen3 model output."""

    def test_thinking_then_tool_call(self):
        """Model thinks then issues tool call (no text after think)."""
        parser = ThinkTagStreamParser()
        all_chunks = []
        # Simulating chunked output
        for chunk in ["<think>I should", " search for", " this</think>"]:
            all_chunks.extend(parser.feed(chunk))
        all_chunks.extend(parser.flush())

        thinking = "".join(c.content for c in all_chunks if c.type == ChunkType.THINKING)
        text = "".join(c.content for c in all_chunks if c.type == ChunkType.TEXT)
        assert thinking == "I should search for this"
        assert text == ""

    def test_thinking_then_answer(self):
        """Model thinks then provides answer."""
        parser = ThinkTagStreamParser()
        all_chunks = []
        for chunk in [
            "<think>Let me compile",
            " the results</think>",
            "\n\nBased on",
            " the search results...",
        ]:
            all_chunks.extend(parser.feed(chunk))
        all_chunks.extend(parser.flush())

        thinking = "".join(c.content for c in all_chunks if c.type == ChunkType.THINKING)
        text = "".join(c.content for c in all_chunks if c.type == ChunkType.TEXT)
        assert thinking == "Let me compile the results"
        assert text == "\n\nBased on the search results..."

    def test_newlines_in_thinking(self):
        """Thinking content can contain newlines."""
        parser = ThinkTagStreamParser()
        result = parser.feed("<think>line1\nline2\nline3</think>answer")
        thinking = "".join(c.content for c in result if c.type == ChunkType.THINKING)
        assert thinking == "line1\nline2\nline3"


class TestStartThinking:
    """Test start_thinking mode (vLLM Qwen3 chat template strips opening <think> tag)."""

    def test_start_thinking_basic(self):
        """Content before </think> is thinking, content after is text."""
        parser = ThinkTagStreamParser(start_thinking=True)
        result = parser.feed("I should search for this</think>\n\n4")
        thinking = "".join(c.content for c in result if c.type == ChunkType.THINKING)
        text = "".join(c.content for c in result if c.type == ChunkType.TEXT)
        assert thinking == "I should search for this"
        assert text == "\n\n4"

    def test_start_thinking_no_close_tag(self):
        """If no </think>, all content is thinking (tool call iteration)."""
        parser = ThinkTagStreamParser(start_thinking=True)
        r1 = parser.feed("I need to search")
        flush = parser.flush()
        thinking = "".join(
            c.content for r in [r1, flush] for c in r if c.type == ChunkType.THINKING
        )
        assert thinking == "I need to search"

    def test_start_thinking_streamed(self):
        """Streaming chunks with start_thinking mode."""
        parser = ThinkTagStreamParser(start_thinking=True)
        all_chunks = []
        for chunk in ["Let me think", " about this</thi", "nk>\n\nThe answer is 4"]:
            all_chunks.extend(parser.feed(chunk))
        all_chunks.extend(parser.flush())

        thinking = "".join(c.content for c in all_chunks if c.type == ChunkType.THINKING)
        text = "".join(c.content for c in all_chunks if c.type == ChunkType.TEXT)
        assert thinking == "Let me think about this"
        assert text == "\n\nThe answer is 4"

    def test_start_thinking_with_open_tag_present(self):
        """If <think> tag IS present despite start_thinking, handle gracefully."""
        parser = ThinkTagStreamParser(start_thinking=True)
        # In THINKING state, <think> is just thinking content (< doesn't match </think>)
        result = parser.feed("<think>extra tags</think>answer")
        thinking = "".join(c.content for c in result if c.type == ChunkType.THINKING)
        text = "".join(c.content for c in result if c.type == ChunkType.TEXT)
        assert "<think>" in thinking  # <think> treated as thinking content
        assert "extra tags" in thinking
        assert text == "answer"

    def test_start_thinking_empty_thinking(self):
        """Model starts with </think> immediately (no thinking content)."""
        parser = ThinkTagStreamParser(start_thinking=True)
        result = parser.feed("</think>just the answer")
        text = "".join(c.content for c in result if c.type == ChunkType.TEXT)
        assert text == "just the answer"
