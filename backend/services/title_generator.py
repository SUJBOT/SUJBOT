"""
Conversation Title Generator using local 8B vLLM model.

Generates concise, descriptive titles from the first user message.
Uses async OpenAI-compatible API for non-blocking operation.
"""

import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class TitleGenerator:
    """
    Generate conversation titles using local Qwen3-VL-8B via vLLM.

    Uses the lighter 8B model for fast, zero-cost title generation.
    Thread-safe and designed for concurrent usage across multiple workers.
    """

    def __init__(self):
        """Initialize the title generator with local 8B vLLM client."""
        base_url = os.getenv("LOCAL_LLM_8B_BASE_URL", "http://localhost:18082/v1")
        self.client = AsyncOpenAI(
            api_key="local-no-key",
            base_url=base_url,
            timeout=15.0,
            max_retries=2,
        )
        self.model = "Qwen/Qwen3-VL-8B-Instruct"
        logger.info("TitleGenerator initialized with local 8B vLLM at %s", base_url)

    async def generate_title(self, first_message: str) -> Optional[str]:
        """
        Generate a conversation title from the first user message.

        Args:
            first_message: The user's first message in the conversation

        Returns:
            Generated title (max 60 chars), or fallback to truncated message
        """
        if not first_message or not first_message.strip():
            return None

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a title generator. Your ONLY job is to create a short title "
                            "(2-5 words) that summarizes the user's message topic.\n\n"
                            "Rules:\n"
                            "- Output ONLY the title, nothing else\n"
                            "- No quotes, no punctuation at the end\n"
                            "- No explanations or preamble\n"
                            "- If the message is a greeting, output: Přátelský pozdrav\n"
                            "- If unclear, summarize the main topic in 2-5 words\n\n"
                            "Examples:\n"
                            "User: 'Ahoj, jak se máš?' → Přátelský pozdrav\n"
                            "User: 'Jak napsat rekurzi v Pythonu?' → Rekurze v Pythonu\n"
                            "User: 'Vysvětli mi kvantovou mechaniku' → Kvantová mechanika"
                        ),
                    },
                    {"role": "user", "content": first_message[:500]},
                ],
                max_tokens=20,
                temperature=0.2,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            raw_title = response.choices[0].message.content
            logger.info(f"LLM raw title response: '{raw_title}' for message: '{first_message[:50]}...'")

            if raw_title:
                title = self._clean_title(raw_title)
                logger.info(f"Cleaned title: '{title}'")

                # If LLM returned the original message or something too long, use fallback
                if title and len(title) <= 60 and title.lower() != first_message.lower().strip():
                    logger.info(f"Using LLM title: '{title}'")
                    return title
                else:
                    logger.warning("Title rejected (too long or same as message), using fallback")

            fallback = self._fallback_title(first_message)
            logger.info(f"Using fallback title: '{fallback}'")
            return fallback

        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            return self._fallback_title(first_message)

    def _clean_title(self, title: str) -> str:
        """Clean up LLM-generated title."""
        if not title:
            return ""
        # Strip <think>...</think> reasoning blocks (Qwen3 models)
        cleaned = re.sub(r"<think>.*?</think>\s*", "", title, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        # Remove common prefixes LLMs add
        prefixes_to_remove = [
            "Title:", "title:", "Název:", "název:",
            "Here is", "Here's", "The title is",
        ]
        cleaned = cleaned.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        # Remove quotes and trailing punctuation
        cleaned = cleaned.strip('"\'').rstrip('.!?:')
        return cleaned

    def _fallback_title(self, message: str) -> str:
        """Generate fallback title by truncating the message."""
        # Clean up whitespace and truncate
        clean = " ".join(message.split())
        if len(clean) <= 50:
            return clean
        return clean[:47] + "..."


# Singleton instance for use across the application
title_generator = TitleGenerator()
