"""
Tests for VL PageStore — page_id parsing, make_page_id, round-trip.
"""

import pytest

from src.vl.page_store import PageStore


class TestPageIdParsing:
    """Tests for page_id_to_components and make_page_id."""

    def test_standard_case(self):
        doc_id, page_num = PageStore.page_id_to_components("BZ_VR1_p001")
        assert doc_id == "BZ_VR1"
        assert page_num == 1

    def test_multi_digit_page_number(self):
        doc_id, page_num = PageStore.page_id_to_components("DOC_p123")
        assert doc_id == "DOC"
        assert page_num == 123

    def test_document_id_with_underscore_p(self):
        """Document IDs containing _p should split on the LAST occurrence."""
        doc_id, page_num = PageStore.page_id_to_components("NP_protocol_p005")
        assert doc_id == "NP_protocol"
        assert page_num == 5

    def test_document_id_with_multiple_p_segments(self):
        doc_id, page_num = PageStore.page_id_to_components("abc_p2_doc_p010")
        assert doc_id == "abc_p2_doc"
        assert page_num == 10

    def test_invalid_format_no_p_separator(self):
        with pytest.raises(ValueError, match="Invalid page_id format"):
            PageStore.page_id_to_components("no_page_suffix")

    def test_invalid_format_non_numeric_page(self):
        with pytest.raises(ValueError, match="Invalid page_id format"):
            PageStore.page_id_to_components("BZ_VR1_pabc")

    def test_invalid_format_empty_string(self):
        with pytest.raises(ValueError, match="Invalid page_id format"):
            PageStore.page_id_to_components("")

    def test_invalid_format_trailing_p(self):
        with pytest.raises(ValueError, match="Invalid page_id format"):
            PageStore.page_id_to_components("BZ_VR1_p")

    def test_make_page_id_standard(self):
        assert PageStore.make_page_id("BZ_VR1", 1) == "BZ_VR1_p001"

    def test_make_page_id_large_number(self):
        assert PageStore.make_page_id("DOC", 123) == "DOC_p123"

    def test_make_page_id_single_digit(self):
        assert PageStore.make_page_id("DOC", 5) == "DOC_p005"

    def test_round_trip(self):
        """make_page_id → page_id_to_components should round-trip correctly."""
        original_doc = "BZ_VR1"
        original_page = 42
        page_id = PageStore.make_page_id(original_doc, original_page)
        doc_id, page_num = PageStore.page_id_to_components(page_id)
        assert doc_id == original_doc
        assert page_num == original_page

    def test_round_trip_with_p_in_doc_id(self):
        """Round-trip with _p in document_id."""
        original_doc = "NP_protocol"
        original_page = 7
        page_id = PageStore.make_page_id(original_doc, original_page)
        doc_id, page_num = PageStore.page_id_to_components(page_id)
        assert doc_id == original_doc
        assert page_num == original_page
