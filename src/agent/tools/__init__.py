"""
RAG Tools

5 VL-adapted tools for retrieval and analysis.
All tools are registered automatically via @register_tool decorator.

Tools:
- search: Jina v4 cosine search -> page images (VL mode)
- expand_context: Adjacent page expansion
- get_document_list: List all indexed documents
- get_document_info: Document metadata/summaries
- get_stats: Corpus/index statistics
"""

import logging
import importlib

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)

# Import all tools to trigger registration
# This uses explicit imports for clarity (no auto-discovery magic)

_failed_imports = []


def _safe_import(module_name: str):
    """Import a tool module with error handling."""
    try:
        return importlib.import_module(f".{module_name}", package=__name__)
    except ImportError as e:
        # Expected: missing optional dependency - log at warning level
        logger.warning(f"Tool module '{module_name}' unavailable (missing dependency): {e}")
        _failed_imports.append((module_name, str(e)))
        return None
    except (SyntaxError, NameError, AttributeError) as e:
        # Code bugs should be fixed, not silently ignored
        logger.error(f"Code error in tool module '{module_name}': {e}", exc_info=True)
        raise
    except Exception as e:
        # Unexpected errors - log with traceback but don't crash
        logger.error(f"Unexpected error importing '{module_name}': {e}", exc_info=True)
        _failed_imports.append((module_name, str(e)))
        return None


# Retrieval tools
_safe_import("search")
_safe_import("expand_context")
_safe_import("get_document_info")
_safe_import("get_document_list")

# Analysis tools
_safe_import("get_stats")

# Report import failures
if _failed_imports:
    logger.warning(
        f"Tool import failures ({len(_failed_imports)}): {', '.join(name for name, _ in _failed_imports)}"
    )

__all__ = ["BaseTool", "ToolInput", "ToolResult", "ToolRegistry", "get_registry"]
