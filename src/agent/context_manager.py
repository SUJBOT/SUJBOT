"""
Context compaction manager for autonomous agent loops.

Implements 3-layer progressive compaction based on actual token usage
from post-call ``response.usage.input_tokens`` (returned by all providers
via ProviderResponse, no additional API call needed):

    Layer 1 (70–85%):  Tool output pruning — only if below compact threshold.
    Layer 2 (85–95%):  LLM summary compaction — only if below emergency threshold.
    Layer 3 (>95%):    Emergency truncation — highest priority, always wins.

Layers are **mutually exclusive per iteration**: only the highest triggered layer
runs (dispatched via ``recommended_action()``).

Design constraints:
    - tool_use_id preservation: Anthropic API validates matching IDs between
      tool_use and tool_result blocks. Pruning MUST keep tool_use_id intact.
    - Same provider for compaction: Layer 2 uses the caller's provider instance.
    - Graceful fallback: If Layer 2 fails, caller detects still-high ratio → Layer 3.
"""

import copy
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Threshold constants (fraction of context window)
PRUNE_THRESHOLD = 0.70
COMPACT_THRESHOLD = 0.85
EMERGENCY_THRESHOLD = 0.95

# Default context window if model lookup fails
DEFAULT_CONTEXT_WINDOW = 128_000

# Emergency truncation: keep first message + this many recent messages
EMERGENCY_KEEP_MESSAGES = 7


class CompactionLayer(Enum):
    """Which compaction action to take (mutually exclusive)."""

    NONE = auto()
    PRUNE = auto()       # Layer 1: tool output pruning
    COMPACT = auto()     # Layer 2: LLM summary compaction
    EMERGENCY = auto()   # Layer 3: emergency truncation


@dataclass(frozen=True)
class ContextBudget:
    """Immutable snapshot of current context window usage."""

    used_tokens: int
    max_tokens: int

    def __post_init__(self):
        if self.used_tokens < 0:
            raise ValueError(f"used_tokens must be non-negative, got {self.used_tokens}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative, got {self.max_tokens}")

    @property
    def ratio(self) -> float:
        """Usage ratio. May exceed 1.0 when context window is overflowed."""
        if self.max_tokens <= 0:
            return 0.0
        return self.used_tokens / self.max_tokens


class ContextBudgetMonitor:
    """
    Tracks context window usage from post-call response.usage.input_tokens.

    Must call ``update_from_response()`` after each LLM call before checking
    thresholds.  Before the first update the monitor reports ratio 0.0 so no
    compaction fires on the very first iteration.
    """

    def __init__(self, context_window: int):
        if context_window <= 0:
            raise ValueError(f"context_window must be positive, got {context_window}")
        self._context_window = context_window
        self._used_tokens: int = 0
        self._initialized: bool = False

    @property
    def context_window(self) -> int:
        return self._context_window

    def update_from_response(self, response: Any) -> None:
        """Extract input_tokens from provider response and update budget."""
        usage = getattr(response, "usage", None)
        if not usage:
            logger.debug("Response has no usage data; context budget not updated")
            return
        if isinstance(usage, dict):
            tokens = usage.get("input_tokens", 0)
        else:
            logger.debug(
                "Response usage is %s (not dict); context budget not updated",
                type(usage).__name__,
            )
            tokens = 0
        if tokens > 0:
            self._used_tokens = tokens
            self._initialized = True

    def check(self) -> ContextBudget:
        """Return current budget snapshot."""
        return ContextBudget(
            used_tokens=self._used_tokens, max_tokens=self._context_window
        )

    def recommended_action(self) -> CompactionLayer:
        """Return the single highest-priority compaction layer to apply."""
        if not self._initialized:
            return CompactionLayer.NONE
        ratio = self.check().ratio
        if ratio >= EMERGENCY_THRESHOLD:
            return CompactionLayer.EMERGENCY
        if ratio >= COMPACT_THRESHOLD:
            return CompactionLayer.COMPACT
        if ratio >= PRUNE_THRESHOLD:
            return CompactionLayer.PRUNE
        return CompactionLayer.NONE

    def needs_pruning(self) -> bool:
        """True when ratio ≥ 70% (Layer 1)."""
        if not self._initialized:
            return False
        return self.check().ratio >= PRUNE_THRESHOLD

    def needs_compaction(self) -> bool:
        """True when ratio ≥ 85% (Layer 2)."""
        if not self._initialized:
            return False
        return self.check().ratio >= COMPACT_THRESHOLD

    def needs_emergency_truncation(self) -> bool:
        """True when ratio ≥ 95% (Layer 3)."""
        if not self._initialized:
            return False
        return self.check().ratio >= EMERGENCY_THRESHOLD


# ---------------------------------------------------------------------------
# Layer 1: Tool output pruning (no LLM call)
# ---------------------------------------------------------------------------

def _prune_content_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace image blocks with text placeholders, truncate long text blocks."""
    pruned: List[Dict[str, Any]] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "image":
            pruned.append({"type": "text", "text": "[image pruned]"})
        elif btype == "text":
            text = block.get("text", "")
            if len(text) > 300:
                text = text[:150] + " [...pruned...] " + text[-100:]
            pruned.append({"type": "text", "text": text})
        else:
            pruned.append(block)
    return pruned


def prune_tool_outputs(
    messages: List[Dict[str, Any]], protect_last_n: int = 2
) -> List[Dict[str, Any]]:
    """
    Layer 1: Replace old tool_result content with compact placeholders.

    Preserves ``tool_use_id`` in every tool_result block (Anthropic API
    validates matching IDs).  The last ``protect_last_n`` user messages
    (tool-result messages) are kept intact so the LLM can still reference
    recent results.

    Args:
        messages: Full message list (mutated copy returned).
        protect_last_n: Number of trailing tool-result user messages to leave
            untouched.

    Returns:
        New message list with pruned older tool results.
    """
    messages = copy.deepcopy(messages)

    # Identify indices of user messages that carry tool_result content
    tool_result_indices: List[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            tool_result_indices.append(i)

    # Protect the last N tool-result user messages
    to_prune = tool_result_indices[: max(0, len(tool_result_indices) - protect_last_n)]

    for idx in to_prune:
        content = messages[idx]["content"]
        if not isinstance(content, list):
            continue
        new_content: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            if block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    pruned_inner = _prune_content_blocks(inner)
                elif isinstance(inner, str) and len(inner) > 300:
                    pruned_inner = inner[:150] + " [...pruned...] " + inner[-100:]
                else:
                    pruned_inner = inner
                new_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id"),
                        "content": pruned_inner,
                    }
                )
            else:
                new_content.append(block)
        messages[idx]["content"] = new_content

    pruned_count = len(to_prune)
    if pruned_count:
        logger.info("Context compaction Layer 1: pruned %d tool-result messages", pruned_count)

    return messages


# ---------------------------------------------------------------------------
# Layer 2: LLM summary compaction (1 extra LLM call)
# ---------------------------------------------------------------------------

_COMPACTION_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "context_compaction.txt"


def _load_compaction_prompt() -> str:
    """Load compaction system prompt from prompts/ directory."""
    if _COMPACTION_PROMPT_PATH.exists():
        return _COMPACTION_PROMPT_PATH.read_text(encoding="utf-8")
    logger.error(
        "Compaction prompt not found at %s — using degraded inline fallback. "
        "This violates CLAUDE.md prompt loading rules. Check volume mounts.",
        _COMPACTION_PROMPT_PATH,
    )
    return (
        "Summarize the preceding conversation into a concise context block. "
        "Preserve: original query, tools called, key facts found, all page_id / chunk_id "
        "citations, and information gaps. Be concise (300-500 words). Do NOT provide a final answer."
    )


def compact_with_summary(
    messages: List[Dict[str, Any]],
    provider: Any,
    protect_last_n_pairs: int = 2,
) -> List[Dict[str, Any]]:
    """
    Layer 2: Summarize older conversation history with one LLM call.

    Keeps the first user message and the last ``protect_last_n_pairs`` message
    pairs (assistant + user) intact.  Everything in between is replaced by a
    single assistant message containing the LLM-generated summary.

    On failure returns messages unchanged (caller should detect still-high
    ratio and fall through to Layer 3).

    Args:
        messages: Full message list.
        provider: LLM provider instance (same one used by the agent).
        protect_last_n_pairs: Number of trailing message pairs (assistant +
            user) to preserve.

    Returns:
        Compacted message list (or original on failure).
    """
    # Need at least first message + something to summarize + protected tail
    min_messages = 1 + protect_last_n_pairs * 2 + 2
    if len(messages) < min_messages:
        logger.debug("Not enough messages (%d) for Layer 2 compaction", len(messages))
        return messages

    first_msg = messages[0]
    tail_count = protect_last_n_pairs * 2  # pairs of (assistant, user)
    middle = messages[1:-tail_count] if tail_count else messages[1:]
    tail = messages[-tail_count:] if tail_count else []

    # Build a text representation of the middle for the compaction LLM
    middle_text_parts: List[str] = []
    for msg in middle:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_pieces = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_pieces.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        text_pieces.append(f"[tool_use: {block.get('name', '?')}]")
                    elif block.get("type") == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, list):
                            for ib in inner:
                                if isinstance(ib, dict) and ib.get("type") == "text":
                                    text_pieces.append(ib.get("text", ""))
                                elif isinstance(ib, dict) and ib.get("type") == "image":
                                    text_pieces.append("[image]")
                        elif isinstance(inner, str):
                            text_pieces.append(inner[:500])
                    elif block.get("type") == "image":
                        text_pieces.append("[image]")
            content = "\n".join(text_pieces)
        middle_text_parts.append(f"{role.upper()}: {content}")

    conversation_text = "\n---\n".join(middle_text_parts)

    compaction_system = _load_compaction_prompt()

    try:
        summary_response = provider.create_message(
            messages=[{"role": "user", "content": conversation_text}],
            tools=[],
            system=compaction_system,
            max_tokens=1024,
            temperature=0.2,
        )
        summary_text = (
            summary_response.text
            if hasattr(summary_response, "text")
            else str(summary_response.content)
        )
    except Exception as e:
        logger.warning(
            "Layer 2 compaction LLM call failed: %s — returning messages unchanged",
            e,
            exc_info=True,
        )
        return messages

    # Rebuild: first_msg + summary as assistant message + tail
    compacted = [
        first_msg,
        {
            "role": "assistant",
            "content": f"[Context summary from earlier conversation]\n\n{summary_text}",
        },
    ] + tail

    logger.info(
        "Context compaction Layer 2: %d messages → %d (summarized %d middle messages)",
        len(messages),
        len(compacted),
        len(middle),
    )

    return compacted


# ---------------------------------------------------------------------------
# Layer 3: Emergency truncation (existing naive logic)
# ---------------------------------------------------------------------------


def emergency_truncate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Layer 3: Last-resort fallback.

    Keeps first message plus the 6 most recent messages, discarding
    everything in between.
    """
    if len(messages) <= EMERGENCY_KEEP_MESSAGES:
        return messages

    truncated_count = len(messages) - EMERGENCY_KEEP_MESSAGES
    result = [messages[0]] + messages[-6:]
    logger.warning(
        "Context compaction Layer 3 (emergency): truncated %d messages (keeping %d)",
        truncated_count,
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Context window lookup
# ---------------------------------------------------------------------------


def get_context_window(model_name: str) -> int:
    """
    Look up context window size for a model from ModelRegistry.

    Falls back to DEFAULT_CONTEXT_WINDOW if model is not found.
    """
    try:
        from ..utils.model_registry import ModelRegistry

        config = ModelRegistry.get_model_config(model_name, "llm")
        return config.context_window
    except (KeyError, ImportError) as e:
        # Expected: model not found in registry, or circular import
        logger.debug(
            "Could not look up context window for '%s': %s — using default %d",
            model_name,
            e,
            DEFAULT_CONTEXT_WINDOW,
        )
        return DEFAULT_CONTEXT_WINDOW
    except Exception as e:
        logger.warning(
            "Unexpected error looking up context window for '%s': %s — using default %d",
            model_name,
            e,
            DEFAULT_CONTEXT_WINDOW,
            exc_info=True,
        )
        return DEFAULT_CONTEXT_WINDOW
