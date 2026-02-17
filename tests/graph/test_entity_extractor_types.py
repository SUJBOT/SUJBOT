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

    def test_legislation_entity_types_exist(self):
        """All 4 legislation entity types must be in ENTITY_TYPES."""
        legislation_types = {"DEFINITION", "SANCTION", "DEADLINE", "AMENDMENT"}
        for etype in legislation_types:
            assert etype in ENTITY_TYPES, f"Missing legislation entity type: {etype}"

    def test_total_entity_type_count(self):
        """Should have exactly 19 entity types (10 original + 5 compliance + 4 legislation)."""
        assert len(ENTITY_TYPES) == 19

    def test_legislation_relationship_types_exist(self):
        """All 5 legislation relationship types must be in RELATIONSHIP_TYPES."""
        legislation_rels = {"SUPERSEDES", "DERIVED_FROM", "HAS_SANCTION", "HAS_DEADLINE", "COMPLIES_WITH"}
        for rtype in legislation_rels:
            assert rtype in RELATIONSHIP_TYPES, f"Missing legislation relationship type: {rtype}"

    def test_total_relationship_type_count(self):
        """Should have exactly 14 relationship types (9 original + 5 legislation)."""
        assert len(RELATIONSHIP_TYPES) == 14
