"""Tests for graph storage dedup methods (SQL merge logic)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.storage import GraphStorageAdapter, _parse_command_count, _parse_dedup_verdict

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
# _parse_command_count
# ---------------------------------------------------------------------------


class TestParseCommandCount:
    def test_delete_result(self):
        assert _parse_command_count("DELETE 5") == 5

    def test_update_result(self):
        assert _parse_command_count("UPDATE 3") == 3

    def test_zero(self):
        assert _parse_command_count("DELETE 0") == 0

    def test_invalid(self):
        assert _parse_command_count("INVALID") == 0

    def test_none(self):
        assert _parse_command_count(None) == 0


# ---------------------------------------------------------------------------
# async_deduplicate_exact
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_exact_dedup_no_duplicates():
    """When no duplicates exist, returns zero stats."""
    adapter = _make_adapter_with_mock_pool()

    conn_mock = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=[])  # no groups found
    acq = AsyncMock()
    acq.__aenter__ = AsyncMock(return_value=conn_mock)
    acq.__aexit__ = AsyncMock(return_value=False)
    adapter.pool.acquire = MagicMock(return_value=acq)

    result = await adapter.async_deduplicate_exact()
    assert result["groups_merged"] == 0
    assert result["entities_removed"] == 0


@pytest.mark.anyio
async def test_exact_dedup_merges_duplicates():
    """Two entities with same (case-insensitive) name+type → merged."""
    adapter = _make_adapter_with_mock_pool()

    # First call: find groups; subsequent calls: merge operations
    group_row = MagicMock()
    group_row.__getitem__ = lambda self, key: {
        "name_key": "sújb",
        "entity_type": "ORGANIZATION",
        "entity_ids": [1, 2],  # 1=canonical (longer desc), 2=duplicate
        "descriptions": ["Státní úřad pro jadernou bezpečnost", "SÚJB"],
        "canonical_name": "Státní úřad pro jadernou bezpečnost",
    }[key]
    group_row.get = lambda key, default=None: {
        "canonical_name": "Státní úřad pro jadernou bezpečnost",
    }.get(key, default)

    call_count = 0

    def make_acq():
        nonlocal call_count
        conn = AsyncMock()
        if call_count == 0:
            conn.fetch = AsyncMock(return_value=[group_row])
        else:
            conn.execute = AsyncMock(return_value="DELETE 1")
        conn.transaction = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False))
        )
        call_count += 1
        acq = AsyncMock()
        acq.__aenter__ = AsyncMock(return_value=conn)
        acq.__aexit__ = AsyncMock(return_value=False)
        return acq

    adapter.pool.acquire = make_acq

    result = await adapter.async_deduplicate_exact()
    assert result["groups_merged"] == 1
    assert result["entities_removed"] == 1


@pytest.mark.anyio
async def test_exact_dedup_handles_transaction_error():
    """If one group's transaction fails, others still processed."""
    adapter = _make_adapter_with_mock_pool()

    group1 = MagicMock()
    group1.__getitem__ = lambda self, key: {
        "name_key": "a",
        "entity_type": "CONCEPT",
        "entity_ids": [1, 2],
        "descriptions": ["desc a", ""],
        "canonical_name": "A",
    }[key]
    group1.get = lambda key, default=None: {"canonical_name": "A"}.get(key, default)

    group2 = MagicMock()
    group2.__getitem__ = lambda self, key: {
        "name_key": "b",
        "entity_type": "CONCEPT",
        "entity_ids": [3, 4],
        "descriptions": ["desc b", ""],
        "canonical_name": "B",
    }[key]
    group2.get = lambda key, default=None: {"canonical_name": "B"}.get(key, default)

    call_count = 0
    import asyncpg

    def make_acq():
        nonlocal call_count
        conn = AsyncMock()
        if call_count == 0:
            conn.fetch = AsyncMock(return_value=[group1, group2])
        elif call_count == 1:
            # First group fails
            conn.execute = AsyncMock(side_effect=asyncpg.PostgresError("boom"))
            conn.transaction = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False)
                )
            )
        else:
            # Second group succeeds
            conn.execute = AsyncMock(return_value="DELETE 1")
            conn.transaction = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False)
                )
            )
        call_count += 1
        acq = AsyncMock()
        acq.__aenter__ = AsyncMock(return_value=conn)
        acq.__aexit__ = AsyncMock(return_value=False)
        return acq

    adapter.pool.acquire = make_acq

    # Should not raise — partial success
    result = await adapter.async_deduplicate_exact()
    assert result["groups_merged"] == 1  # only second group succeeded


# ---------------------------------------------------------------------------
# async_deduplicate_semantic
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_semantic_dedup_no_provider():
    """Without LLM provider, semantic dedup is skipped."""
    adapter = _make_adapter_with_mock_pool()

    result = await adapter.async_deduplicate_semantic(
        similarity_threshold=0.85,
        llm_provider=None,
    )
    assert result["candidates_found"] == 0


@pytest.mark.anyio
async def test_semantic_dedup_no_candidates():
    """When no candidates above threshold, returns zero stats."""
    adapter = _make_adapter_with_mock_pool()

    conn_mock = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=[])
    acq = AsyncMock()
    acq.__aenter__ = AsyncMock(return_value=conn_mock)
    acq.__aexit__ = AsyncMock(return_value=False)
    adapter.pool.acquire = MagicMock(return_value=acq)

    provider = MagicMock()
    result = await adapter.async_deduplicate_semantic(
        similarity_threshold=0.85,
        llm_provider=provider,
    )
    assert result["candidates_found"] == 0
    assert result["llm_confirmed"] == 0


@pytest.mark.anyio
async def test_semantic_dedup_llm_confirms():
    """LLM confirms merge → entities merged."""
    adapter = _make_adapter_with_mock_pool()

    candidate = MagicMock()
    candidate.__getitem__ = lambda self, key: {
        "id1": 1,
        "id2": 2,
        "name1": "SÚJB",
        "name2": "Státní úřad pro jadernou bezpečnost",
        "type1": "ORGANIZATION",
        "type2": "ORGANIZATION",
        "desc1": "Nuclear authority",
        "desc2": "State office for nuclear safety",
        "similarity": 0.92,
    }[key]

    # Entity info rows for canonical name selection
    entity_info_rows = [
        {"entity_id": 1, "name": "SÚJB", "description": "Nuclear authority"},
        {"entity_id": 2, "name": "Státní úřad pro jadernou bezpečnost", "description": "State office for nuclear safety"},
    ]

    call_count = 0

    def make_acq():
        nonlocal call_count
        conn = AsyncMock()
        if call_count == 0:
            conn.fetch = AsyncMock(return_value=[candidate])
        elif call_count == 1:
            # Entity info fetch for canonical name selection
            conn.fetch = AsyncMock(return_value=entity_info_rows)
        else:
            conn.execute = AsyncMock(return_value="DELETE 1")
            conn.transaction = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False)
                )
            )
        call_count += 1
        acq = AsyncMock()
        acq.__aenter__ = AsyncMock(return_value=conn)
        acq.__aexit__ = AsyncMock(return_value=False)
        return acq

    adapter.pool.acquire = make_acq

    # Mock LLM saying YES
    from dataclasses import dataclass

    @dataclass
    class Response:
        text: str = "YES\nSame entity"

    provider = MagicMock()
    provider.create_message = MagicMock(return_value=Response())

    # The prompt file exists at prompts/graph_entity_dedup.txt in the project root
    result = await adapter.async_deduplicate_semantic(
        similarity_threshold=0.85,
        llm_provider=provider,
    )

    assert result["llm_confirmed"] == 1
    assert result["entities_removed"] == 1


@pytest.mark.anyio
async def test_semantic_dedup_llm_rejects():
    """LLM rejects merge → no entities merged."""
    adapter = _make_adapter_with_mock_pool()

    candidate = MagicMock()
    candidate.__getitem__ = lambda self, key: {
        "id1": 1,
        "id2": 2,
        "name1": "vyhl. č. 307/2002 Sb.",
        "name2": "vyhl. č. 308/2002 Sb.",
        "type1": "REGULATION",
        "type2": "REGULATION",
        "desc1": "Regulation 307",
        "desc2": "Regulation 308",
        "similarity": 0.88,
    }[key]

    conn_mock = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=[candidate])
    acq = AsyncMock()
    acq.__aenter__ = AsyncMock(return_value=conn_mock)
    acq.__aexit__ = AsyncMock(return_value=False)
    adapter.pool.acquire = MagicMock(return_value=acq)

    from dataclasses import dataclass

    @dataclass
    class Response:
        text: str = "NO\nDifferent regulations"

    provider = MagicMock()
    provider.create_message = MagicMock(return_value=Response())

    result = await adapter.async_deduplicate_semantic(
        similarity_threshold=0.85,
        llm_provider=provider,
    )

    assert result["candidates_found"] == 1
    assert result["llm_confirmed"] == 0
    assert result["entities_removed"] == 0


class TestParseCommandCountEdgeCases:
    def test_empty_string(self):
        assert _parse_command_count("") == 0

    def test_single_word(self):
        assert _parse_command_count("DELETE") == 0


@pytest.mark.anyio
async def test_semantic_dedup_transitive_closure():
    """Three entities forming a chain: A≈B and B≈C → all merge into one group."""
    adapter = _make_adapter_with_mock_pool()

    cand1 = MagicMock()
    cand1.__getitem__ = lambda self, key: {
        "id1": 1,
        "id2": 2,
        "name1": "Entity A",
        "name2": "Entity B",
        "type1": "CONCEPT",
        "type2": "CONCEPT",
        "desc1": "desc a",
        "desc2": "desc b",
        "similarity": 0.90,
    }[key]

    cand2 = MagicMock()
    cand2.__getitem__ = lambda self, key: {
        "id1": 2,
        "id2": 3,
        "name1": "Entity B",
        "name2": "Entity C",
        "type1": "CONCEPT",
        "type2": "CONCEPT",
        "desc1": "desc b",
        "desc2": "desc c",
        "similarity": 0.88,
    }[key]

    # Entity info rows for canonical name selection
    entity_info_rows = [
        {"entity_id": 1, "name": "Entity A", "description": "desc a"},
        {"entity_id": 2, "name": "Entity B", "description": "desc b"},
        {"entity_id": 3, "name": "Entity C", "description": "desc c"},
    ]

    call_count = 0

    def make_acq():
        nonlocal call_count
        conn = AsyncMock()
        if call_count == 0:
            conn.fetch = AsyncMock(return_value=[cand1, cand2])
        elif call_count == 1:
            # Entity info fetch for canonical name selection
            conn.fetch = AsyncMock(return_value=entity_info_rows)
        else:
            conn.execute = AsyncMock(return_value="DELETE 2")
            conn.transaction = MagicMock(
                return_value=AsyncMock(
                    __aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False)
                )
            )
        call_count += 1
        acq = AsyncMock()
        acq.__aenter__ = AsyncMock(return_value=conn)
        acq.__aexit__ = AsyncMock(return_value=False)
        return acq

    adapter.pool.acquire = make_acq

    from dataclasses import dataclass

    @dataclass
    class Response:
        text: str = "YES\nSame entity"

    provider = MagicMock()
    provider.create_message = MagicMock(return_value=Response())

    result = await adapter.async_deduplicate_semantic(
        similarity_threshold=0.85,
        llm_provider=provider,
    )

    # Both pairs confirmed, but transitive closure merges into 1 group
    assert result["llm_confirmed"] == 2
    assert result["groups_merged"] == 1  # all three → one group


# ---------------------------------------------------------------------------
# _parse_dedup_verdict
# ---------------------------------------------------------------------------


class TestParseDedupVerdict:
    def test_json_yes(self):
        assert _parse_dedup_verdict('{"verdict": "YES", "reason": "Same entity"}') is True

    def test_json_no(self):
        assert _parse_dedup_verdict('{"verdict": "NO", "reason": "Different"}') is False

    def test_json_case_insensitive(self):
        assert _parse_dedup_verdict('{"verdict": "yes"}') is True

    def test_json_with_code_fences(self):
        assert _parse_dedup_verdict('```json\n{"verdict": "YES"}\n```') is True

    def test_json_missing_verdict_key(self):
        assert _parse_dedup_verdict('{"answer": "YES"}') is False

    def test_json_non_dict_falls_back_to_text(self):
        """Non-dict JSON (e.g. array) falls back to plain text parsing."""
        assert _parse_dedup_verdict('["YES"]') is False  # starts with '[', not 'YES'

    def test_plain_text_yes(self):
        assert _parse_dedup_verdict("YES\nSame entity") is True

    def test_plain_text_no(self):
        assert _parse_dedup_verdict("NO\nDifferent regulations") is False

    def test_plain_text_whitespace(self):
        assert _parse_dedup_verdict("  YES  ") is True

    def test_empty_string(self):
        assert _parse_dedup_verdict("") is False
