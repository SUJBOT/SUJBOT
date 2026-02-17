"""
Graph entity and relationship type constants.

Single source of truth for valid types used in entity extraction and validation.
"""

ENTITY_TYPES = {
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
    "OBLIGATION",
    "PROHIBITION",
    "PERMISSION",
    "EVIDENCE",
    "CONTROL",
    "DEFINITION",
    "SANCTION",
    "DEADLINE",
    "AMENDMENT",
}

RELATIONSHIP_TYPES = {
    "DEFINES",
    "REFERENCES",
    "AMENDS",
    "REQUIRES",
    "REGULATES",
    "PART_OF",
    "APPLIES_TO",
    "SUPERVISES",
    "AUTHORED_BY",
    "SUPERSEDES",
    "DERIVED_FROM",
    "HAS_SANCTION",
    "HAS_DEADLINE",
    "COMPLIES_WITH",
}
