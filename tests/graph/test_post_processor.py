"""Tests for graph post-processor: rebuild + dedup pipeline."""

import asyncio
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.graph.post_processor import (
    _embed_new_communities,
    _embed_new_entities,
    rebuild_graph_communities,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    text: str


class FakeProvider:
    """Minimal LLM provider stub."""

    def __init__(self, answer: str = "YES\nSame entity"):
        self._answer = answer

    def get_model_name(self) -> str:
        return "fake-model"

    def create_message(self, messages, tools, system, max_tokens, temperature):
        return _FakeResponse(text=self._answer)


class FakeEmbedder:
    """Returns deterministic embeddings."""

    def encode_passages(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        return np.random.RandomState(42).randn(len(texts), 384).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.random.RandomState(42).randn(384).astype(np.float32)


class FakeDetector:
    """Returns a single community containing all entities."""

    def detect(self, entities, relationships, max_levels=3, **kwargs):
        if len(entities) < 3:
            return []
        return [
            {
                "level": 0,
                "entity_ids": [e["entity_id"] for e in entities],
                "title": None,
                "summary": None,
            }
        ]


class FakeSummarizer:
    """Returns a fixed summary."""

    def __init__(self):
        self.provider = FakeProvider()

    def summarize(self, community_entities, community_relationships):
        return "Test summary. With details."


def _make_storage_mock(
    entities=None,
    relationships=None,
    dedup_exact_result=None,
    dedup_semantic_result=None,
):
    """Build a mock GraphStorageAdapter with sensible defaults."""
    storage = AsyncMock()

    entities = entities or [
        {
            "entity_id": 1,
            "name": "A",
            "entity_type": "CONCEPT",
            "description": "desc A",
            "document_id": "doc1",
        },
        {
            "entity_id": 2,
            "name": "B",
            "entity_type": "CONCEPT",
            "description": "desc B",
            "document_id": "doc1",
        },
        {
            "entity_id": 3,
            "name": "C",
            "entity_type": "CONCEPT",
            "description": "desc C",
            "document_id": "doc2",
        },
    ]
    relationships = relationships or [
        {
            "source_entity_id": 1,
            "target_entity_id": 2,
            "relationship_type": "RELATED",
            "weight": 1.0,
            "relationship_id": 1,
        },
        {
            "source_entity_id": 2,
            "target_entity_id": 3,
            "relationship_type": "RELATED",
            "weight": 1.0,
            "relationship_id": 2,
        },
    ]

    storage.async_get_all = AsyncMock(return_value=(entities, relationships))
    storage.async_deduplicate_exact = AsyncMock(
        return_value=dedup_exact_result
        or {"groups_merged": 0, "entities_removed": 0, "relationships_remapped": 0}
    )
    storage.async_deduplicate_semantic = AsyncMock(
        return_value=dedup_semantic_result
        or {"candidates_found": 0, "llm_confirmed": 0, "entities_removed": 0}
    )
    storage.async_save_communities = AsyncMock(return_value=1)
    storage._ensure_pool = AsyncMock()
    storage.pool = MagicMock()

    # Mock pool.acquire() context manager for embedding
    conn_mock = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=[])  # no entities need embedding
    conn_mock.executemany = AsyncMock()
    acq = AsyncMock()
    acq.__aenter__ = AsyncMock(return_value=conn_mock)
    acq.__aexit__ = AsyncMock(return_value=False)
    storage.pool.acquire = MagicMock(return_value=acq)

    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rebuild_full_pipeline():
    """Full pipeline: dedup → detect → summarize → save."""
    storage = _make_storage_mock()
    detector = FakeDetector()
    summarizer = FakeSummarizer()
    provider = FakeProvider()

    stats = await rebuild_graph_communities(
        graph_storage=storage,
        community_detector=detector,
        community_summarizer=summarizer,
        graph_embedder=None,  # skip embedding for speed
        llm_provider=provider,
        document_id="test-doc",
        enable_dedup=True,
    )

    assert storage.async_deduplicate_exact.called
    assert stats["communities_detected"] == 1
    assert stats["communities_saved"] == 1
    assert stats["communities_summarized"] == 1


@pytest.mark.anyio
async def test_rebuild_with_few_entities():
    """<3 entities → skip community detection."""
    entities = [
        {
            "entity_id": 1,
            "name": "A",
            "entity_type": "CONCEPT",
            "description": "x",
            "document_id": "d1",
        },
    ]
    storage = _make_storage_mock(entities=entities, relationships=[])
    detector = FakeDetector()

    stats = await rebuild_graph_communities(
        graph_storage=storage,
        community_detector=detector,
        enable_dedup=False,
    )

    assert stats["communities_detected"] == 0


@pytest.mark.anyio
async def test_rebuild_skips_dedup_when_disabled():
    """enable_dedup=False → no dedup methods called."""
    storage = _make_storage_mock()
    detector = FakeDetector()

    await rebuild_graph_communities(
        graph_storage=storage,
        community_detector=detector,
        enable_dedup=False,
    )

    assert not storage.async_deduplicate_exact.called
    assert not storage.async_deduplicate_semantic.called


@pytest.mark.anyio
async def test_rebuild_continues_on_dedup_failure():
    """If dedup fails, community detection still runs."""
    storage = _make_storage_mock()
    storage.async_deduplicate_exact = AsyncMock(side_effect=RuntimeError("boom"))
    detector = FakeDetector()

    stats = await rebuild_graph_communities(
        graph_storage=storage,
        community_detector=detector,
        enable_dedup=True,
    )

    # Dedup failed but communities still detected
    assert "error" in stats.get("exact_dedup", {})
    assert stats["communities_detected"] == 1


@pytest.mark.anyio
async def test_rebuild_no_summarizer():
    """Without summarizer, communities get fallback titles."""
    storage = _make_storage_mock()
    detector = FakeDetector()

    stats = await rebuild_graph_communities(
        graph_storage=storage,
        community_detector=detector,
        community_summarizer=None,
        enable_dedup=False,
    )

    assert stats["communities_detected"] == 1
    assert "communities_summarized" not in stats


@pytest.mark.anyio
async def test_embed_new_entities():
    """Test embedding entities with NULL search_embedding."""
    storage = AsyncMock()
    storage._ensure_pool = AsyncMock()

    # Mock pool.acquire() returning entities that need embedding
    entity_rows = [
        {
            "entity_id": 1,
            "name": "SÚJB",
            "entity_type": "ORGANIZATION",
            "description": "Nuclear authority",
        },
        {"entity_id": 2, "name": "AtomAct", "entity_type": "REGULATION", "description": ""},
    ]

    conn_mock = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=entity_rows)
    conn_mock.executemany = AsyncMock()
    acq = AsyncMock()
    acq.__aenter__ = AsyncMock(return_value=conn_mock)
    acq.__aexit__ = AsyncMock(return_value=False)
    storage.pool = MagicMock()
    storage.pool.acquire = MagicMock(return_value=acq)

    embedder = FakeEmbedder()
    count = await _embed_new_entities(storage, embedder)
    assert count == 2


@pytest.mark.anyio
async def test_embed_new_communities():
    """Test embedding communities with NULL search_embedding."""
    storage = AsyncMock()
    storage._ensure_pool = AsyncMock()

    community_rows = [
        {
            "community_id": 1,
            "title": "Nuclear Safety",
            "summary": "Overview of nuclear safety regulations",
        },
    ]

    conn_mock = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=community_rows)
    conn_mock.executemany = AsyncMock()
    acq = AsyncMock()
    acq.__aenter__ = AsyncMock(return_value=conn_mock)
    acq.__aexit__ = AsyncMock(return_value=False)
    storage.pool = MagicMock()
    storage.pool.acquire = MagicMock(return_value=acq)

    embedder = FakeEmbedder()
    count = await _embed_new_communities(storage, embedder)
    assert count == 1


@pytest.mark.anyio
async def test_debounce_cancels_previous():
    """Two rapid calls → only last rebuild should actually fire."""
    pytest.importorskip("fastapi", reason="fastapi required for backend route tests")
    from backend.routes.documents import (
        _REBUILD_DELAY_SECONDS,
        _schedule_graph_rebuild,
        _vl_components,
    )

    # Setup minimal components
    mock_storage = MagicMock()
    mock_detector = MagicMock()
    _vl_components["graph_storage"] = mock_storage
    _vl_components["community_detector"] = mock_detector

    try:
        # First call — sets timer
        _schedule_graph_rebuild("doc1")

        # Access the module-level timer
        from backend.routes import documents

        timer1 = documents._rebuild_timer
        assert timer1 is not None

        # Second call — should cancel first timer
        _schedule_graph_rebuild("doc2")
        timer2 = documents._rebuild_timer
        assert timer2 is not None

        # timer1 should have been cancelled
        assert timer1.cancelled()
        assert not timer2.cancelled()

        # Cleanup
        timer2.cancel()
    finally:
        _vl_components.pop("graph_storage", None)
        _vl_components.pop("community_detector", None)
