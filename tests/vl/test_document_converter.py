"""Tests for document format conversion (TXT, MD, HTML, DOCX, LaTeX)."""

import fitz
import pytest

from src.vl.document_converter import DocumentConverter, IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS, _decode_text
from src.exceptions import ConversionError


@pytest.fixture
def converter():
    return DocumentConverter()


class TestSupportedExtensions:
    def test_pdf_included(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_all_expected_extensions(self):
        expected = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".tex", ".latex"}
        assert SUPPORTED_EXTENSIONS == expected


class TestDecodeText:
    def test_utf8(self):
        assert _decode_text("hello".encode("utf-8")) == "hello"

    def test_utf8_bom(self):
        assert _decode_text("hello".encode("utf-8-sig")) == "hello"

    def test_czech_windows1250(self):
        text = "Příliš žluťoučký kůň"
        encoded = text.encode("windows-1250")
        assert _decode_text(encoded) == text

    def test_latin1_fallback(self):
        # latin-1 can decode anything
        result = _decode_text(bytes(range(128, 256)))
        assert len(result) == 128


class TestTextToPdf:
    def test_basic_conversion(self, converter):
        pdf_bytes = converter._text_to_pdf(b"Hello World\nLine 2")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) == 1
        text = doc[0].get_text()
        assert "Hello" in text
        doc.close()

    def test_multipage(self, converter):
        # Create enough lines to force multiple pages
        content = "\n".join(f"Line {i}" for i in range(200)).encode()
        pdf_bytes = converter._text_to_pdf(content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) > 1
        doc.close()

    def test_empty_text(self, converter):
        pdf_bytes = converter._text_to_pdf(b"")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Empty text should still produce valid PDF
        assert len(doc) >= 1
        doc.close()


class TestMarkdownToPdf:
    def test_basic_conversion(self, converter):
        md = b"# Title\n\nParagraph with **bold** and *italic*."
        pdf_bytes = converter._markdown_to_pdf(md)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        text = doc[0].get_text()
        assert "Title" in text
        doc.close()


class TestHtmlToPdf:
    def test_basic_conversion(self, converter):
        html = b"<h1>Hello</h1><p>World</p>"
        pdf_bytes = converter._html_to_pdf(html)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        doc.close()

    def test_full_html_document(self, converter):
        html = b"<html><body><h1>Test</h1></body></html>"
        pdf_bytes = converter._html_to_pdf(html)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        doc.close()


class TestConvertToPdf:
    @pytest.mark.anyio
    async def test_txt_conversion(self, converter):
        pdf = await converter.convert_to_pdf(b"Test", ".txt")
        doc = fitz.open(stream=pdf, filetype="pdf")
        assert len(doc) >= 1
        doc.close()

    @pytest.mark.anyio
    async def test_md_conversion(self, converter):
        pdf = await converter.convert_to_pdf(b"# Title", ".md")
        doc = fitz.open(stream=pdf, filetype="pdf")
        assert len(doc) >= 1
        doc.close()

    @pytest.mark.anyio
    async def test_html_conversion(self, converter):
        pdf = await converter.convert_to_pdf(b"<p>Hello</p>", ".html")
        doc = fitz.open(stream=pdf, filetype="pdf")
        assert len(doc) >= 1
        doc.close()

    @pytest.mark.anyio
    async def test_pdf_passthrough(self, converter):
        # PDF should be returned unchanged
        original = fitz.open()
        original.new_page()
        pdf_bytes = original.tobytes()
        original.close()

        result = await converter.convert_to_pdf(pdf_bytes, ".pdf")
        assert result == pdf_bytes

    @pytest.mark.anyio
    async def test_unsupported_extension(self, converter):
        from src.exceptions import ConversionError

        with pytest.raises(ConversionError, match="Unsupported format"):
            await converter.convert_to_pdf(b"data", ".xyz")

    @pytest.mark.anyio
    async def test_docx_without_libreoffice(self, converter):
        """DOCX conversion should raise if LibreOffice is not installed."""
        import shutil

        if shutil.which("libreoffice"):
            pytest.skip("LibreOffice is installed")

        from src.exceptions import ConversionError

        with pytest.raises(ConversionError, match="LibreOffice"):
            await converter.convert_to_pdf(b"fake docx", ".docx")

    @pytest.mark.anyio
    async def test_latex_without_pdflatex(self, converter):
        """LaTeX conversion should raise if pdflatex is not installed."""
        import shutil

        if shutil.which("pdflatex"):
            pytest.skip("pdflatex is installed")

        from src.exceptions import ConversionError

        with pytest.raises(ConversionError, match="pdflatex"):
            await converter.convert_to_pdf(b"\\documentclass{article}", ".tex")


class TestExtractText:
    def test_txt_extraction(self, converter):
        text = converter.extract_text(b"Hello World", ".txt")
        assert text == "Hello World"

    def test_md_extraction(self, converter):
        text = converter.extract_text(b"# Title\n**bold**", ".md")
        assert "Title" in text
        assert "bold" in text

    def test_html_extraction(self, converter):
        text = converter.extract_text(b"<p>Hello <b>World</b></p>", ".html")
        assert "Hello" in text
        assert "World" in text
        assert "<p>" not in text  # Tags should be stripped

    def test_latex_extraction(self, converter):
        latex = rb"\documentclass{article}\begin{document}Hello World\end{document}"
        text = converter.extract_text(latex, ".tex")
        assert "Hello World" in text

    def test_pdf_extraction(self, converter):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF Content")
        pdf_bytes = doc.tobytes()
        doc.close()

        text = converter.extract_text(pdf_bytes, ".pdf")
        assert "Test PDF Content" in text


class TestCheckDependencies:
    def test_returns_dict(self):
        deps = DocumentConverter.check_dependencies()
        assert "libreoffice" in deps
        assert "pdflatex" in deps
        assert isinstance(deps["libreoffice"], bool)
        assert isinstance(deps["pdflatex"], bool)


def _make_png(width=100, height=80, color=(1, 0, 0)):
    """Create a minimal PNG image in memory using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(fitz.Rect(0, 0, width, height), color=color, fill=color)
    pix = page.get_pixmap()
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


class TestImagesToPdf:
    def test_single_image(self):
        png = _make_png()
        result = DocumentConverter.images_to_pdf([png], ["test.png"])
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) == 1
        doc.close()

    def test_multiple_images_become_pages(self):
        images = [_make_png(color=(1, 0, 0)), _make_png(color=(0, 1, 0)), _make_png(color=(0, 0, 1))]
        result = DocumentConverter.images_to_pdf(images, ["a.png", "b.png", "c.png"])
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) == 3
        doc.close()

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No images"):
            DocumentConverter.images_to_pdf([], [])

    def test_corrupt_image_raises_conversion_error(self):
        with pytest.raises(ConversionError):
            DocumentConverter.images_to_pdf([b"not an image"], ["bad.png"])

    def test_image_extensions_constant(self):
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".tif" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS
