"""
State machine parser for <think>...</think> tags in streaming token output.

Qwen3 thinking models emit <think>...</think> reasoning blocks before text/tool calls.
This parser handles tags split arbitrarily across streaming chunks.

NOTE: vLLM's Qwen3 chat template strips the opening <think> tag (it's part of the
generation prefix). Only </think> appears in the stream. Use ``start_thinking=True``
when the model always begins in thinking mode (e.g., vLLM with Qwen3 thinking models).

States:
  TEXT        - Normal text output
  MAYBE_OPEN  - Buffering a potential "<think>" opening tag
  THINKING    - Inside <think> block, emitting thinking content
  MAYBE_CLOSE - Inside <think>, buffering a potential "</think>" closing tag
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class ChunkType(Enum):
    TEXT = auto()
    THINKING = auto()


@dataclass
class ParsedChunk:
    """A parsed chunk with its type and content."""

    type: ChunkType
    content: str


class _State(Enum):
    TEXT = auto()
    MAYBE_OPEN = auto()
    THINKING = auto()
    MAYBE_CLOSE = auto()


_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


class ThinkTagStreamParser:
    """
    Stateful parser for <think>...</think> tags in a chunked token stream.

    Usage:
        parser = ThinkTagStreamParser()
        for chunk in stream:
            for parsed in parser.feed(chunk):
                if parsed.type == ChunkType.THINKING:
                    show_thinking(parsed.content)
                else:
                    buffer_text(parsed.content)
        for parsed in parser.flush():
            ...  # handle remaining content
    """

    def __init__(self, start_thinking: bool = False) -> None:
        self._state = _State.THINKING if start_thinking else _State.TEXT
        self._buf = ""  # Partial tag buffer

    def feed(self, chunk: str) -> List[ParsedChunk]:
        """
        Feed a chunk of streamed text and return parsed results.

        Args:
            chunk: Raw text chunk from LLM stream (may contain partial tags).

        Returns:
            List of ParsedChunk items (may be empty if buffering a partial tag).
        """
        results: List[ParsedChunk] = []
        i = 0

        while i < len(chunk):
            char = chunk[i]

            if self._state == _State.TEXT:
                if char == "<":
                    # Could be start of <think>
                    self._state = _State.MAYBE_OPEN
                    self._buf = "<"
                else:
                    results.append(ParsedChunk(ChunkType.TEXT, char))
                i += 1

            elif self._state == _State.MAYBE_OPEN:
                self._buf += char
                i += 1

                if _OPEN_TAG.startswith(self._buf):
                    # Still matching <think>
                    if self._buf == _OPEN_TAG:
                        # Complete match — enter thinking state
                        self._state = _State.THINKING
                        self._buf = ""
                else:
                    # Not a <think> tag — flush buffer as text
                    results.append(ParsedChunk(ChunkType.TEXT, self._buf))
                    self._buf = ""
                    self._state = _State.TEXT

            elif self._state == _State.THINKING:
                if char == "<":
                    # Could be start of </think>
                    self._state = _State.MAYBE_CLOSE
                    self._buf = "<"
                else:
                    results.append(ParsedChunk(ChunkType.THINKING, char))
                i += 1

            elif self._state == _State.MAYBE_CLOSE:
                self._buf += char
                i += 1

                if _CLOSE_TAG.startswith(self._buf):
                    # Still matching </think>
                    if self._buf == _CLOSE_TAG:
                        # Complete match — exit thinking state
                        self._state = _State.TEXT
                        self._buf = ""
                else:
                    # Not </think> — flush buffer as thinking content
                    results.append(ParsedChunk(ChunkType.THINKING, self._buf))
                    self._buf = ""
                    self._state = _State.THINKING

        return _coalesce(results)

    def flush(self) -> List[ParsedChunk]:
        """
        Flush any remaining buffered content (e.g., truncated tags at stream end).

        Returns:
            List of remaining ParsedChunk items.
        """
        results: List[ParsedChunk] = []

        if self._buf:
            if self._state in (_State.THINKING, _State.MAYBE_CLOSE):
                # Unclosed <think> — emit buffer as thinking
                results.append(ParsedChunk(ChunkType.THINKING, self._buf))
            else:
                # Partial open tag that never completed — emit as text
                results.append(ParsedChunk(ChunkType.TEXT, self._buf))
            self._buf = ""

        self._state = _State.TEXT
        return results


def _coalesce(chunks: List[ParsedChunk]) -> List[ParsedChunk]:
    """Merge adjacent chunks of the same type to reduce event count."""
    if not chunks:
        return chunks

    merged: List[ParsedChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        if chunk.type == merged[-1].type:
            merged[-1] = ParsedChunk(chunk.type, merged[-1].content + chunk.content)
        else:
            merged.append(chunk)
    return merged
