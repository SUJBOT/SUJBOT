"""
Shared text processing utilities.
"""

import re


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from text."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text
