"""Tests for entity extractor type validation with compliance types."""

import pytest
from src.graph.entity_extractor import ENTITY_TYPES, RELATIONSHIP_TYPES


class TestEntityTypes:
    """Verify compliance entity types are registered."""

    def test_compliance_entity_types_exist(self):
        """All 5 compliance entity types must be in ENTITY_TYPES."""
        compliance_types = {"OBLIGATION", "PROHIBITION", "PERMISSION", "EVIDENCE", "CONTROL"}
        for etype in compliance_types:
            assert etype in ENTITY_TYPES, f"Missing compliance entity type: {etype}"

    def test_original_entity_types_preserved(self):
        """Original 10 types must still exist after extension."""
        original_types = {
            "REGULATION",
            "STANDARD",
            "SECTION",
            "ORGANIZATION",
            "PERSON",
            "CONCEPT",
            "REQUIREMENT",
            "FACILITY",
            "ROLE",
            "DOCUMENT",
        }
        for etype in original_types:
            assert etype in ENTITY_TYPES, f"Missing original entity type: {etype}"

    def test_total_entity_type_count(self):
        """Should have exactly 15 entity types (10 original + 5 compliance)."""
        assert len(ENTITY_TYPES) == 15

    def test_relationship_types_unchanged(self):
        """Relationship types should not be modified."""
        assert len(RELATIONSHIP_TYPES) == 9
