"""
Abbreviation Detector — heuristics for Czech legal/regulatory abbreviations.

Provides static detection of abbreviations, Czech abbreviation patterns in text,
and initial-letter matching between abbreviations and full names.
"""

import re
import unicodedata
from typing import List, Optional, Tuple

# Domain-specific known abbreviations (abbreviation → full Czech name)
KNOWN_ABBREVIATIONS = {
    "SÚJB": "Státní úřad pro jadernou bezpečnost",
    "IAEA": "International Atomic Energy Agency",
    "ČVUT": "České vysoké učení technické",
    "JE": "jaderná elektrárna",
    "AZ": "aktivní zóna",
    "BZ": "bezpečnostní zpráva",
    "INES": "International Nuclear and Radiological Event Scale",
    "NEA": "Nuclear Energy Agency",
    "ČEZ": "České energetické závody",
    "MAAE": "Mezinárodní agentura pro atomovou energii",
    "SONS": "Státní úřad pro jadernou bezpečnost",  # former abbreviation
    "TLD": "termoluminiscenční dozimetr",
    "OPB": "obecné požadavky na bezpečnost",
    "PpBJZ": "předprovozní bezpečnostní zpráva",
    "JBHV": "jaderná bezpečnost, havarijní vzorek",
    "RO": "radiační ochrana",
    "NJZ": "nové jaderné zdroje",
}

# Czech patterns for abbreviation introduction
# Matches: "dále jen „XYZ"", "(dále jen „XYZ")", "dále jen ‚XYZ'",
#           "dále jen „XYZ" nebo „ABC")"
_CZECH_ABBR_PATTERNS = [
    # "dále jen „ABBR"" — Czech „…" (U+201E…U+201C) and other double-quote styles
    re.compile(
        r'dále\s+jen\s+[„\u201c\u201d"»]([^"„\u201c\u201d"«»]+)[\u201c\u201d"«]', re.IGNORECASE
    ),
    # (dále jen „ABBR") with parens
    re.compile(
        r'\(dále\s+jen\s+[„\u201c\u201d"»]([^"„\u201c\u201d"«»]+)[\u201c\u201d"«]\)', re.IGNORECASE
    ),
    # "dále jen ‚ABBR'"
    re.compile(r"dále\s+jen\s+[‚']([^'']+)[''']", re.IGNORECASE),
]


def _strip_diacritics(s: str) -> str:
    """Remove diacritics for initial-letter matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def is_likely_abbreviation(name: str) -> bool:
    """Check if a name looks like an abbreviation.

    Heuristics:
    - 2-10 characters, all uppercase (with possible diacritics)
    - Known abbreviation from dictionary
    """
    stripped = name.strip()
    if stripped in KNOWN_ABBREVIATIONS:
        return True
    if (
        2 <= len(stripped) <= 10
        and stripped == stripped.upper()
        and stripped.replace(".", "").isalpha()
    ):
        return True
    return False


def detect_abbreviations_in_text(text: str) -> List[str]:
    """Extract abbreviations introduced in Czech text.

    Detects patterns like:
    - "dále jen „SÚJB""
    - "(dále jen „AZ")"

    Returns list of abbreviation strings found.
    """
    results = []
    seen: set = set()
    for pattern in _CZECH_ABBR_PATTERNS:
        for match in pattern.finditer(text):
            abbr = match.group(1).strip()
            if abbr and len(abbr) <= 50 and abbr not in seen:
                results.append(abbr)
                seen.add(abbr)
    return results


def find_abbreviation_match(abbr: str, full_name: str) -> bool:
    """Check if abbreviation matches initials of full name.

    Uses initial-letter matching with diacritics stripping and
    Czech preposition skipping.

    Examples:
        - "SÚJB" matches "Státní Úřad pro Jadernou Bezpečnost"
        - "ČVUT" matches "České Vysoké Učení Technické"
        - "JE" matches "Jaderná Elektrárna"
    """
    abbr_clean = _strip_diacritics(abbr.upper().strip())

    # Skip common Czech prepositions/articles
    skip_words = {"a", "v", "na", "pro", "ze", "do", "od", "za", "po", "se", "s", "k", "o", "i"}
    words = [w for w in full_name.split() if w.lower() not in skip_words and len(w) > 1]

    if len(words) < 2 or len(abbr_clean) < 2:
        return False

    initials = "".join(_strip_diacritics(w[0]).upper() for w in words)

    return initials == abbr_clean or initials.startswith(abbr_clean)


def lookup_known_abbreviation(name: str) -> Optional[str]:
    """If name is a known abbreviation, return its full form."""
    return KNOWN_ABBREVIATIONS.get(name.strip())


def find_abbreviation_pairs(names: List[str]) -> List[Tuple[str, str]]:
    """Given a list of entity names, find (abbreviation, full_name) pairs.

    Returns list of (abbreviation, full_name) tuples.
    """
    abbrs = [n for n in names if is_likely_abbreviation(n)]
    fulls = [n for n in names if not is_likely_abbreviation(n) and len(n) > 10]

    pairs = []
    for abbr in abbrs:
        # Check known dictionary first
        known = lookup_known_abbreviation(abbr)
        if known:
            for full in fulls:
                if full.lower() == known.lower():
                    pairs.append((abbr, full))
                    break
            else:
                # Known match not in the list, but still useful info
                pass
            continue

        # Try initial-letter matching
        for full in fulls:
            if find_abbreviation_match(abbr, full):
                pairs.append((abbr, full))
                break

    return pairs
