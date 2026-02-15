"""Tests for graph storage dedup methods (SQL merge logic)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.storage import GraphStorageAdapter, _parse_command_count


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
    """When no cross-document duplicates exist, returns zero stats."""
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
async def test_exact_dedup_merges_cross_document():
    """Two entities with same name+type in different documents → merged."""
    adapter = _make_adapter_with_mock_pool()

    # First call: find groups; subsequent calls: merge operations
    group_row = MagicMock()
    group_row.__getitem__ = lambda self, key: {
        "name": "SÚJB",
        "entity_type": "ORGANIZATION",
        "entity_ids": [1, 2],  # 1=canonical (longer desc), 2=duplicate
        "descriptions": ["Státní úřad pro jadernou bezpečnost", "SÚJB"],
    }[key]

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
        "name": "A",
        "entity_type": "CONCEPT",
        "entity_ids": [1, 2],
        "descriptions": ["desc a", ""],
    }[key]

    group2 = MagicMock()
    group2.__getitem__ = lambda self, key: {
        "name": "B",
        "entity_type": "CONCEPT",
        "entity_ids": [3, 4],
        "descriptions": ["desc b", ""],
    }[key]

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
    assert result["groups_merged"] == 2  # both attempted


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

    call_count = 0

    def make_acq():
        nonlocal call_count
        conn = AsyncMock()
        if call_count == 0:
            conn.fetch = AsyncMock(return_value=[candidate])
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

    # Need the prompt file to exist
    with patch("src.graph.storage.Path") as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.read_text.return_value = (
            "Entity 1: {name1} (type: {type1}) — {desc1}\n"
            "Entity 2: {name2} (type: {type2}) — {desc2}"
        )
        mock_path.return_value = mock_path_instance
        # Also mock the resolve().parent chain
        mock_path.__truediv__ = MagicMock(return_value=mock_path_instance)
        file_path = MagicMock()
        file_path.resolve.return_value.parent.parent.parent.__truediv__ = MagicMock(
            return_value=MagicMock(__truediv__=MagicMock(return_value=mock_path_instance))
        )

        # Simpler approach: just put a real prompt file
        import tempfile
        import os

        prompt_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        prompt_file = os.path.join(prompt_dir, "graph_entity_dedup.txt")

        # The prompt file should already exist from our earlier step
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
