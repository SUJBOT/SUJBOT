"""Tests for enhanced entity deduplication: aliases, batch LLM, abbreviation detector."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.abbreviation_detector import (
    detect_abbreviations_in_text,
    find_abbreviation_match,
    find_abbreviation_pairs,
    is_likely_abbreviation,
    lookup_known_abbreviation,
)
from src.graph.storage import (
    GraphStorageAdapter,
    _infer_alias_type,
    _parse_batch_verdicts,
    _parse_dedup_verdict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter_with_mock_pool():
    """Create a GraphStorageAdapter with a mocked pool."""
    adapter = GraphStorageAdapter.__new__(GraphStorageAdapter)
    adapter.pool = MagicMock()
    adapter._connection_string = None
    adapter._owns_pool = False
    adapter._embedder = None
    return adapter


# ---------------------------------------------------------------------------
# Phase 1: _infer_alias_type
# ---------------------------------------------------------------------------


class TestInferAliasType:
    def test_short_uppercase_abbreviation(self):
        assert _infer_alias_type("SÚJB") == "abbreviation"

    def test_ascii_abbreviation(self):
        assert _infer_alias_type("IAEA") == "abbreviation"

    def test_two_letter_abbreviation(self):
        assert _infer_alias_type("JE") == "abbreviation"

    def test_full_name_is_variant(self):
        assert _infer_alias_type("Státní úřad pro jadernou bezpečnost") == "variant"

    def test_mixed_case_is_variant(self):
        assert _infer_alias_type("AtomAct") == "variant"

    def test_single_char_is_variant(self):
        assert _infer_alias_type("A") == "variant"

    def test_long_uppercase_is_variant(self):
        assert _infer_alias_type("VERYLONGACRONYM") == "variant"

    def test_has_numbers_is_variant(self):
        assert _infer_alias_type("VR1") == "variant"

    def test_whitespace_stripped(self):
        assert _infer_alias_type("  IAEA  ") == "abbreviation"


# ---------------------------------------------------------------------------
# Phase 1: async_add_alias / async_lookup_alias
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_add_alias_calls_insert():
    """async_add_alias executes INSERT with ON CONFLICT DO NOTHING."""
    adapter = _make_adapter_with_mock_pool()
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")

    result = await adapter.async_add_alias(
        conn,
        entity_id=42,
        alias="SÚJB",
        alias_type="abbreviation",
        source="extraction",
    )
    assert result is True
    conn.execute.assert_called_once()


@pytest.mark.anyio
async def test_add_alias_duplicate_returns_false():
    """Duplicate alias (already exists) → returns False."""
    adapter = _make_adapter_with_mock_pool()
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 0")

    result = await adapter.async_add_alias(conn, entity_id=42, alias="SÚJB")
    assert result is False


@pytest.mark.anyio
async def test_lookup_alias_with_type():
    """async_lookup_alias with entity_type filter."""
    adapter = _make_adapter_with_mock_pool()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=42)

    result = await adapter.async_lookup_alias(conn, "SÚJB", "ORGANIZATION")
    assert result == 42
    # Should have used the type-filtered query
    call_args = conn.fetchval.call_args
    assert "entity_type" in call_args[0][0]


@pytest.mark.anyio
async def test_lookup_alias_without_type():
    """async_lookup_alias without entity_type returns from alias table."""
    adapter = _make_adapter_with_mock_pool()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=42)

    result = await adapter.async_lookup_alias(conn, "SÚJB")
    assert result == 42


@pytest.mark.anyio
async def test_lookup_alias_not_found():
    """async_lookup_alias returns None when no match."""
    adapter = _make_adapter_with_mock_pool()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=None)

    result = await adapter.async_lookup_alias(conn, "unknown_entity")
    assert result is None


@pytest.mark.anyio
async def test_migrate_aliases():
    """async_migrate_aliases copies aliases from source to target entity."""
    adapter = _make_adapter_with_mock_pool()
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 3")

    count = await adapter.async_migrate_aliases(conn, from_entity_id=2, to_entity_id=1)
    assert count == 3
    conn.execute.assert_called_once()


@pytest.mark.anyio
async def test_get_aliases():
    """async_get_aliases returns list of alias strings."""
    adapter = _make_adapter_with_mock_pool()
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[
            {"alias": "SÚJB"},
            {"alias": "State Office"},
        ]
    )

    aliases = await adapter.async_get_aliases(conn, entity_id=42)
    assert aliases == ["SÚJB", "State Office"]


# ---------------------------------------------------------------------------
# Phase 2: _parse_batch_verdicts
# ---------------------------------------------------------------------------


class TestParseBatchVerdicts:
    def test_valid_json(self):
        result = _parse_batch_verdicts('{"verdicts": ["YES", "NO", "YES"]}', 3)
        assert result == [True, False, True]

    def test_json_with_code_fences(self):
        result = _parse_batch_verdicts('```json\n{"verdicts": ["YES", "NO"]}\n```', 2)
        assert result == [True, False]

    def test_count_mismatch_returns_none(self):
        result = _parse_batch_verdicts('{"verdicts": ["YES"]}', 3)
        assert result is None

    def test_invalid_json_with_regex_fallback(self):
        text = 'Some text before {"verdicts": ["YES", "NO"]} extra'
        result = _parse_batch_verdicts(text, 2)
        assert result == [True, False]

    def test_completely_invalid_returns_none(self):
        result = _parse_batch_verdicts("random text", 2)
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_batch_verdicts("", 2)
        assert result is None

    def test_array_format(self):
        result = _parse_batch_verdicts('["YES", "NO", "YES"]', 3)
        assert result == [True, False, True]

    def test_case_insensitive(self):
        result = _parse_batch_verdicts('{"verdicts": ["yes", "No", "YES"]}', 3)
        assert result == [True, False, True]


# ---------------------------------------------------------------------------
# Phase 2: _batch_llm_canonicalize
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_batch_llm_canonicalize_groups_by_type():
    """Batch LLM groups candidates by entity_type and uses batch prompt for >1 pairs."""
    adapter = _make_adapter_with_mock_pool()

    # Both candidates in same type group so batch mode triggers (len(batch) > 1)
    candidates = [
        MagicMock(
            **{
                "__getitem__": lambda self, k: {
                    "id1": 1,
                    "id2": 2,
                    "name1": "SÚJB",
                    "name2": "Státní úřad",
                    "type1": "ORGANIZATION",
                    "type2": "ORGANIZATION",
                    "desc1": "desc1",
                    "desc2": "desc2",
                    "similarity": 0.90,
                }[k]
            }
        ),
        MagicMock(
            **{
                "__getitem__": lambda self, k: {
                    "id1": 3,
                    "id2": 4,
                    "name1": "IAEA",
                    "name2": "Mezinárodní agentura",
                    "type1": "ORGANIZATION",
                    "type2": "ORGANIZATION",
                    "desc1": "agency",
                    "desc2": "agency2",
                    "similarity": 0.88,
                }[k]
            }
        ),
    ]

    @dataclass
    class Response:
        text: str

    provider = MagicMock()
    # Batch call returns verdicts for 2 pairs
    provider.create_message = MagicMock(return_value=Response(text='{"verdicts": ["YES", "YES"]}'))

    single_template = (
        "Entity 1: {name1} ({type1}) — {desc1}\n" "Entity 2: {name2} ({type2}) — {desc2}"
    )

    confirmed = await adapter._batch_llm_canonicalize(
        candidates, provider, single_template, batch_size=20
    )

    assert (1, 2) in confirmed
    assert (3, 4) in confirmed


@pytest.mark.anyio
async def test_batch_llm_fallback_on_parse_failure():
    """When batch parsing fails, falls back to sequential arbitration."""
    adapter = _make_adapter_with_mock_pool()

    # Need 2+ candidates in same type for batch mode to trigger
    cand1 = MagicMock()
    cand1.__getitem__ = lambda self, k: {
        "id1": 1,
        "id2": 2,
        "name1": "A",
        "name2": "B",
        "type1": "CONCEPT",
        "type2": "CONCEPT",
        "desc1": "d1",
        "desc2": "d2",
        "similarity": 0.90,
    }[k]
    cand2 = MagicMock()
    cand2.__getitem__ = lambda self, k: {
        "id1": 3,
        "id2": 4,
        "name1": "C",
        "name2": "D",
        "type1": "CONCEPT",
        "type2": "CONCEPT",
        "desc1": "d3",
        "desc2": "d4",
        "similarity": 0.85,
    }[k]

    @dataclass
    class Response:
        text: str

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call (batch) returns unparseable
            return Response(text="INVALID RESPONSE")
        # Subsequent calls (sequential fallback) return YES
        return Response(text='{"verdict": "YES"}')

    provider = MagicMock()
    provider.create_message = mock_create

    single_template = "{name1} vs {name2} ({type1}/{type2}): {desc1}/{desc2}"

    confirmed = await adapter._batch_llm_canonicalize(
        [cand1, cand2], provider, single_template, batch_size=20
    )

    assert (1, 2) in confirmed
    assert (3, 4) in confirmed
    assert call_count == 3  # 1 batch failed, then 2 sequential


# ---------------------------------------------------------------------------
# Phase 4: Abbreviation Detector
# ---------------------------------------------------------------------------


class TestIsLikelyAbbreviation:
    def test_known_abbreviation(self):
        assert is_likely_abbreviation("SÚJB") is True

    def test_unknown_short_uppercase(self):
        assert is_likely_abbreviation("XYZ") is True

    def test_lowercase_word(self):
        assert is_likely_abbreviation("reaktor") is False

    def test_mixed_case(self):
        assert is_likely_abbreviation("AtomAct") is False

    def test_single_char(self):
        assert is_likely_abbreviation("A") is False

    def test_too_long(self):
        assert is_likely_abbreviation("VERYLONGABBREV") is False

    def test_with_dot(self):
        assert is_likely_abbreviation("ČSN.") is True


class TestDetectAbbreviationsInText:
    def test_czech_pattern_dale_jen(self):
        text = "Státní úřad pro jadernou bezpečnost (dále jen \u201eSÚJB\u201c) je regulátor."
        result = detect_abbreviations_in_text(text)
        assert "SÚJB" in result

    def test_no_abbreviation_in_text(self):
        result = detect_abbreviations_in_text("Běžný text bez zkratek.")
        assert result == []

    def test_multiple_abbreviations(self):
        text = (
            "Aktivní zóna (dále jen \u201eAZ\u201c) a bezpečnostní zpráva "
            "(dále jen \u201eBZ\u201c) jsou klíčové."
        )
        result = detect_abbreviations_in_text(text)
        assert "AZ" in result
        assert "BZ" in result


class TestFindAbbreviationMatch:
    def test_sujb_matches(self):
        assert find_abbreviation_match("SÚJB", "Státní úřad pro jadernou bezpečnost") is True

    def test_cvut_matches(self):
        assert find_abbreviation_match("ČVUT", "České vysoké učení technické") is True

    def test_je_matches(self):
        assert find_abbreviation_match("JE", "jaderná elektrárna") is True

    def test_no_match(self):
        assert find_abbreviation_match("XYZ", "Státní úřad pro jadernou bezpečnost") is False

    def test_single_char_no_match(self):
        assert find_abbreviation_match("S", "Státní") is False


class TestLookupKnownAbbreviation:
    def test_known(self):
        assert lookup_known_abbreviation("SÚJB") == "Státní úřad pro jadernou bezpečnost"

    def test_unknown(self):
        assert lookup_known_abbreviation("UNKNOWN") is None


class TestFindAbbreviationPairs:
    def test_finds_pair(self):
        names = ["SÚJB", "Státní úřad pro jadernou bezpečnost", "reaktor VR-1"]
        pairs = find_abbreviation_pairs(names)
        assert ("SÚJB", "Státní úřad pro jadernou bezpečnost") in pairs

    def test_no_pairs(self):
        names = ["reaktor VR-1", "jaderná elektrárna"]
        pairs = find_abbreviation_pairs(names)
        assert pairs == []


# ---------------------------------------------------------------------------
# Phase 4: Entity Extractor — abbreviation/name_en parsing
# ---------------------------------------------------------------------------


class TestEntityExtractorAbbreviations:
    def test_parse_abbreviations_and_name_en(self):
        from src.graph.entity_extractor import EntityExtractor

        # Create extractor without provider (we'll call _parse_response directly)
        extractor = EntityExtractor.__new__(EntityExtractor)

        response_text = """{
            "entities": [
                {
                    "name": "Státní úřad pro jadernou bezpečnost",
                    "type": "ORGANIZATION",
                    "description": "Czech nuclear regulatory authority",
                    "abbreviations": ["SÚJB"],
                    "name_en": "State Office for Nuclear Safety"
                }
            ],
            "relationships": []
        }"""

        result = extractor._parse_response(response_text, "test_page")
        assert len(result["entities"]) == 1
        entity = result["entities"][0]
        assert entity["abbreviations"] == ["SÚJB"]
        assert entity["name_en"] == "State Office for Nuclear Safety"

    def test_parse_without_optional_fields(self):
        from src.graph.entity_extractor import EntityExtractor

        extractor = EntityExtractor.__new__(EntityExtractor)

        response_text = """{
            "entities": [
                {"name": "SÚJB", "type": "ORGANIZATION", "description": "regulator"}
            ],
            "relationships": []
        }"""

        result = extractor._parse_response(response_text, "test_page")
        assert len(result["entities"]) == 1
        entity = result["entities"][0]
        # Optional fields should not be present (or empty list)
        assert entity.get("abbreviations", []) == []
        assert "name_en" not in entity

    def test_parse_empty_abbreviations(self):
        from src.graph.entity_extractor import EntityExtractor

        extractor = EntityExtractor.__new__(EntityExtractor)

        response_text = """{
            "entities": [
                {
                    "name": "SÚJB",
                    "type": "ORGANIZATION",
                    "description": "desc",
                    "abbreviations": [],
                    "name_en": ""
                }
            ],
            "relationships": []
        }"""

        result = extractor._parse_response(response_text, "test_page")
        entity = result["entities"][0]
        assert entity.get("abbreviations", []) == []
        assert "name_en" not in entity


# ---------------------------------------------------------------------------
# Phase 3: Cross-type compatibility in candidate SQL
# ---------------------------------------------------------------------------


class TestCrossTypeCompatibility:
    """Verify the candidate SQL allows cross-type matching for compatible types."""

    def test_regulation_document_standard_compatible(self):
        """REGULATION, DOCUMENT, STANDARD should be cross-matchable in dedup SQL."""
        import inspect

        from src.graph.storage import GraphStorageAdapter

        source = inspect.getsource(GraphStorageAdapter.async_deduplicate_semantic)
        # The SQL should contain cross-type matching
        assert "REGULATION" in source
        assert "DOCUMENT" in source
        assert "STANDARD" in source

    def test_concept_requirement_compatible(self):
        """CONCEPT and REQUIREMENT should be cross-matchable."""
        import inspect

        from src.graph.storage import GraphStorageAdapter

        source = inspect.getsource(GraphStorageAdapter.async_deduplicate_semantic)
        assert "CONCEPT" in source
        assert "REQUIREMENT" in source
