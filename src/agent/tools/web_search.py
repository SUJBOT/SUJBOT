"""
Web Search Tool — internet search via Gemini Google Search grounding.

Uses Google's Gemini model with native Google Search grounding to answer
questions requiring current or external information not in the document corpus.
Returns grounded text with source URLs for citation.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import requests
from google import genai
from google.genai import types
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)

_REDIRECT_TIMEOUT = 4  # seconds per redirect resolution


class WebSearchInput(ToolInput):
    """Input for web search tool."""

    query: str = Field(
        ...,
        description="Search query for finding current information on the web",
        min_length=1,
        max_length=500,
    )


@register_tool
class WebSearchTool(BaseTool):
    """Search the internet for current information not available in the document corpus."""

    name = "web_search"
    description = (
        "Search the internet for current information not in the document corpus. "
        "Use ONLY as a last resort when internal search yields no results AND "
        "the question requires external/current information. "
        "Returns grounded text with source URLs."
    )
    input_schema = WebSearchInput

    def execute_impl(self, query: str) -> ToolResult:
        # Check if web search is enabled in config
        if self.config and not getattr(self.config, "web_search_enabled", True):
            return ToolResult(
                success=False,
                data=None,
                error=(
                    "Web search is currently disabled. "
                    "Tell the user you cannot search the internet right now. "
                    "IMPORTANT: Respond in the SAME language as the user's query."
                ),
            )

        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return ToolResult(
                success=False,
                data=None,
                error="GOOGLE_API_KEY not set — web search requires a Google API key",
            )

        model_name = getattr(self.config, "web_search_model", "gemini-2.0-flash")

        # Call Gemini API — catch API/network errors here;
        # programming errors (KeyError, TypeError, etc.) propagate to BaseTool.execute()
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
        except (ValueError, RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=(
                    f"Web search failed: {type(e).__name__}: {e}. "
                    "Tell the user you could not complete the web search. "
                    "IMPORTANT: Respond in the SAME language as the user's query."
                ),
            )

        # Parse grounded response
        answer = response.text or ""

        sources: List[Dict[str, Any]] = []
        search_queries: List[str] = []

        candidate = response.candidates[0] if response.candidates else None
        grounding_meta = getattr(candidate, "grounding_metadata", None) if candidate else None

        if grounding_meta:
            # Search queries used
            search_queries = list(getattr(grounding_meta, "web_search_queries", []) or [])

            # Grounding chunks → source URLs (Gemini returns redirect URLs)
            chunks = getattr(grounding_meta, "grounding_chunks", []) or []
            raw_sources: List[Dict[str, str]] = []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    url = getattr(web, "uri", "") or ""
                    title = getattr(web, "title", "") or ""
                    if url:
                        raw_sources.append({"url": url, "title": title})

            # Resolve redirect URLs to actual page URLs in parallel
            resolved = self._resolve_redirect_urls(raw_sources)

            # Deduplicate by resolved URL
            seen_urls: set = set()
            for src in resolved:
                url = src["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "url": url,
                        "title": src["title"],
                        "index": len(sources) + 1,
                    })
                    logger.debug(f"Grounding source: {url} ({src['title']})")

        # Build citation instruction for the agent
        source_lines = []
        for s in sources:
            source_lines.append(f"[{s['index']}] {s['title']} - {s['url']}")

        citation_note = ""
        if source_lines:
            citation_note = (
                "\n\nSources (cite with \\webcite{url}{title}):\n"
                + "\n".join(source_lines)
            )

        return ToolResult(
            success=True,
            data={
                "answer": answer + citation_note,
                "sources": sources,
            },
            metadata={
                "query": query,
                "model": model_name,
                "search_queries": search_queries,
                "source_count": len(sources),
            },
        )

    @staticmethod
    def _resolve_redirect_urls(
        raw_sources: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Resolve Gemini grounding redirect URLs to actual page URLs.

        Gemini returns opaque redirect URIs via vertexaisearch.cloud.google.com.
        Following the redirect with HEAD gives the real page URL.
        """
        if not raw_sources:
            return []

        def _resolve_one(src: Dict[str, str]) -> Dict[str, str]:
            url = src["url"]
            title = src["title"]
            try:
                resp = requests.head(url, allow_redirects=True, timeout=_REDIRECT_TIMEOUT)
                resolved_url = resp.url
                # Build a better title from the resolved URL if title is just a domain
                parsed = urlparse(resolved_url)
                if title == parsed.netloc or not title:
                    # Use last meaningful path segment as title hint
                    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
                    if path_parts:
                        last = unquote(path_parts[-1])
                        # Clean up file extensions
                        if "." in last:
                            last = last.rsplit(".", 1)[0]
                        last = last.replace("-", " ").replace("_", " ")
                        title = f"{parsed.netloc}: {last[:60]}"
                    else:
                        title = parsed.netloc
                return {"url": resolved_url, "title": title}
            except requests.RequestException as e:
                logger.debug(f"Failed to resolve redirect URL: {e}")
                return src  # Fall back to original redirect URL

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_resolve_one, src): i for i, src in enumerate(raw_sources)}
            results: List[Optional[Dict[str, str]]] = [None] * len(raw_sources)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return [r for r in results if r is not None]
