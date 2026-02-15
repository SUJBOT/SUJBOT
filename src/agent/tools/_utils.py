"""
Shared Tool Utilities

Helper functions used across multiple tools.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def create_error_result(error_message: str, tool_name: str = None) -> Dict[str, Any]:
    """
    Create standardized error result.

    Args:
        error_message: Error description
        tool_name: Name of tool that failed

    Returns:
        Error dict
    """
    return {
        "error": error_message,
        "tool": tool_name,
        "success": False,
    }


def estimate_tokens_from_vl_result(num_pages: int, tokens_per_page: int = 1600) -> int:
    """
    Estimate token count for a VL search result including image overhead.

    Anthropic charges ~1600 tokens per document page image.

    Args:
        num_pages: Number of page images in result
        tokens_per_page: Tokens per page image (default: 1600)

    Returns:
        Estimated total tokens (images + text metadata)
    """
    image_tokens = num_pages * tokens_per_page
    # ~50 tokens per page for text metadata (page_id, score, etc.)
    metadata_tokens = num_pages * 50
    return image_tokens + metadata_tokens
