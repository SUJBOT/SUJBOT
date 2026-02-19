"""
Document format converter — converts non-PDF documents to PDF for VL pipeline.

Supported formats: DOCX, TXT, Markdown, HTML, LaTeX.
Strategy: convert everything to PDF, then run the unchanged VL indexing pipeline.

Also provides text extraction for chat attachments (no conversion needed).
"""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import fitz  # PyMuPDF

from ..exceptions import ConversionError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".tex", ".latex"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# A4 dimensions in points (72 dpi)
_A4_WIDTH = 595
_A4_HEIGHT = 842
_MARGIN = 54  # ~0.75 inch


def _decode_text(content: bytes) -> str:
    """Decode text bytes with Central European fallback cascade."""
    for encoding in ("utf-8-sig", "utf-8", "windows-1250", "iso-8859-2", "latin-1"):
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return content.decode("latin-1")  # latin-1 never fails


class DocumentConverter:
    """Converts document formats to PDF and extracts text."""

    async def convert_to_pdf(self, content: bytes, extension: str) -> bytes:
        """
        Convert document content to PDF bytes.

        Args:
            content: Raw file bytes
            extension: File extension (e.g., '.docx', '.txt')

        Returns:
            PDF bytes

        Raises:
            ConversionError: If conversion fails
        """
        ext = extension.lower()
        if ext == ".pdf":
            return content

        try:
            if ext == ".docx":
                return await self._docx_to_pdf(content)
            elif ext == ".txt":
                return self._text_to_pdf(content)
            elif ext == ".md":
                return self._markdown_to_pdf(content)
            elif ext in (".html", ".htm"):
                return self._html_to_pdf(content)
            elif ext in (".tex", ".latex"):
                return await self._latex_to_pdf(content)
            else:
                raise ConversionError(
                    f"Unsupported format: {ext}",
                    details={"extension": ext},
                )
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(
                f"Failed to convert {ext} to PDF: {e}",
                details={"extension": ext},
                cause=e,
            )

    def extract_text(self, content: bytes, extension: str) -> str:
        """
        Extract plain text from document for chat attachments.

        Args:
            content: Raw file bytes
            extension: File extension

        Returns:
            Extracted text content
        """
        ext = extension.lower()
        try:
            if ext in (".txt", ".md"):
                return _decode_text(content)
            elif ext in (".html", ".htm"):
                return self._extract_html_text(content)
            elif ext == ".docx":
                return self._extract_docx_text(content)
            elif ext in (".tex", ".latex"):
                return self._extract_latex_text(content)
            elif ext == ".pdf":
                return self._extract_pdf_text(content)
            else:
                return _decode_text(content)
        except Exception as e:
            logger.warning(f"Text extraction failed for {ext}: {e}", exc_info=True)
            # Binary formats produce garbled text when raw-decoded — return error message
            if ext in (".docx", ".pdf"):
                return f"[Error: Could not extract text from {ext} file: {e}]"
            return _decode_text(content)

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check availability of external tools for conversion."""
        return {
            "libreoffice": shutil.which("libreoffice") is not None,
            "pdflatex": shutil.which("pdflatex") is not None,
        }

    @staticmethod
    def images_to_pdf(image_buffers: list[bytes], filenames: list[str]) -> bytes:
        """Combine multiple images into a single PDF (one image per page).

        Each image is inserted as a full page preserving its aspect ratio.

        Args:
            image_buffers: Raw image bytes (PNG, JPG, TIFF, BMP, WebP)
            filenames: Corresponding filenames (for error messages)

        Returns:
            PDF bytes

        Raises:
            ValueError: If image_buffers is empty
            ConversionError: If any image cannot be processed
        """
        if not image_buffers:
            raise ValueError("No images provided")

        doc = fitz.open()
        try:
            for i, (img_bytes, fname) in enumerate(zip(image_buffers, filenames)):
                try:
                    img_doc = fitz.open(stream=img_bytes, filetype="png")  # fitz auto-detects
                    if len(img_doc) == 0:
                        raise ConversionError(
                            f"Image has no pages: {fname}",
                            details={"filename": fname, "index": i},
                        )
                    img_rect = img_doc[0].rect
                    page = doc.new_page(width=img_rect.width, height=img_rect.height)
                    page.insert_image(page.rect, stream=img_bytes)
                    img_doc.close()
                except ConversionError:
                    raise
                except Exception as e:
                    raise ConversionError(
                        f"Failed to process image {fname}: {e}",
                        details={"filename": fname, "index": i},
                        cause=e,
                    )
            return doc.tobytes()
        finally:
            doc.close()

    # ─── Conversion methods ───────────────────────────────────────────

    async def _docx_to_pdf(self, content: bytes) -> bytes:
        """Convert DOCX to PDF via LibreOffice headless."""
        if not shutil.which("libreoffice"):
            raise ConversionError(
                "LibreOffice not installed — cannot convert DOCX to PDF",
                details={"dependency": "libreoffice"},
            )

        with tempfile.TemporaryDirectory(prefix="sujbot_docx_") as tmpdir:
            input_path = Path(tmpdir) / "input.docx"
            input_path.write_bytes(content)

            proc = await asyncio.create_subprocess_exec(
                "libreoffice",
                "--headless",
                "--norestore",
                "--convert-to",
                "pdf",
                "--outdir",
                tmpdir,
                str(input_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                proc.kill()
                raise ConversionError(
                    "DOCX conversion timed out (60s limit)",
                    details={"timeout": 60},
                )

            if proc.returncode != 0:
                raise ConversionError(
                    f"LibreOffice conversion failed (exit {proc.returncode})",
                    details={"stderr": stderr.decode(errors="replace")[:500]},
                )

            output_path = Path(tmpdir) / "input.pdf"
            if not output_path.exists():
                raise ConversionError("LibreOffice produced no output PDF")

            return output_path.read_bytes()

    def _text_to_pdf(self, content: bytes) -> bytes:
        """Convert plain text to PDF via PyMuPDF."""
        text = _decode_text(content)
        doc = fitz.open()

        try:
            fontsize = 10
            line_height = fontsize * 1.4
            usable_height = _A4_HEIGHT - 2 * _MARGIN
            lines_per_page = int(usable_height / line_height)

            lines = text.split("\n")
            for i in range(0, len(lines), lines_per_page):
                page = doc.new_page(width=_A4_WIDTH, height=_A4_HEIGHT)
                y = _MARGIN
                for line in lines[i : i + lines_per_page]:
                    # Truncate very long lines
                    display_line = line[:200] if len(line) > 200 else line
                    page.insert_text(
                        (_MARGIN, y + fontsize),
                        display_line,
                        fontsize=fontsize,
                        fontname="helv",
                    )
                    y += line_height

            return doc.tobytes()
        finally:
            doc.close()

    def _markdown_to_pdf(self, content: bytes) -> bytes:
        """Convert Markdown to PDF via marko → HTML → fitz.Story."""
        import marko

        text = _decode_text(content)
        html = marko.convert(text)
        return self._html_content_to_pdf(html)

    def _html_to_pdf(self, content: bytes) -> bytes:
        """Convert HTML to PDF via fitz.Story."""
        html = _decode_text(content)
        return self._html_content_to_pdf(html)

    def _html_content_to_pdf(self, html: str) -> bytes:
        """Render HTML string to PDF pages via fitz.Story + DocumentWriter."""
        import io

        # Wrap in basic HTML structure with styling if not already wrapped
        if "<html" not in html.lower():
            html = (
                '<html><body style="font-family: sans-serif; font-size: 10pt;'
                f' line-height: 1.4; margin: 0; padding: 0;">{html}</body></html>'
            )

        bio = io.BytesIO()
        writer = fitz.DocumentWriter(bio)
        story = fitz.Story(html)
        content_rect = fitz.Rect(
            _MARGIN, _MARGIN, _A4_WIDTH - _MARGIN, _A4_HEIGHT - _MARGIN
        )
        page_rect = fitz.Rect(0, 0, _A4_WIDTH, _A4_HEIGHT)

        try:
            more = True
            while more:
                dev = writer.begin_page(page_rect)
                more, _ = story.place(content_rect)
                story.draw(dev)
                writer.end_page()

            writer.close()
            return bio.getvalue()
        except Exception:
            try:
                writer.close()
            except Exception as cleanup_err:
                logger.debug("Failed to close DocumentWriter during cleanup: %s", cleanup_err)
            raise

    async def _latex_to_pdf(self, content: bytes) -> bytes:
        """Convert LaTeX to PDF via pdflatex (two passes)."""
        if not shutil.which("pdflatex"):
            raise ConversionError(
                "pdflatex not installed — cannot convert LaTeX to PDF",
                details={"dependency": "pdflatex"},
            )

        with tempfile.TemporaryDirectory(prefix="sujbot_latex_") as tmpdir:
            input_path = Path(tmpdir) / "input.tex"
            input_path.write_bytes(content)

            # Two passes for cross-references
            output_path = Path(tmpdir) / "input.pdf"
            for pass_num in range(2):
                proc = await asyncio.create_subprocess_exec(
                    "pdflatex",
                    "--no-shell-escape",
                    "-no-parse-first-line",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    tmpdir,
                    str(input_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    proc.kill()
                    raise ConversionError(
                        f"LaTeX compilation timed out on pass {pass_num + 1} (120s limit)",
                        details={"timeout": 120, "pass": pass_num + 1},
                    )

                # After pass 1, skip pass 2 if no output was produced
                if pass_num == 0 and proc.returncode != 0 and not output_path.exists():
                    break
            if not output_path.exists():
                # Try to extract a useful error from the log
                log_path = Path(tmpdir) / "input.log"
                error_detail = ""
                if log_path.exists():
                    log_text = log_path.read_text(errors="replace")
                    # Extract first error line
                    for line in log_text.split("\n"):
                        if line.startswith("!"):
                            error_detail = line[:200]
                            break

                raise ConversionError(
                    f"pdflatex produced no output PDF{': ' + error_detail if error_detail else ''}",
                    details={"returncode": proc.returncode},
                )

            return output_path.read_bytes()

    # ─── Text extraction methods ──────────────────────────────────────

    def _extract_html_text(self, content: bytes) -> str:
        """Extract text from HTML via BeautifulSoup."""
        from bs4 import BeautifulSoup

        html = _decode_text(content)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX via python-docx."""
        import io

        from docx import Document

        doc = Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs if para.text)

    def _extract_latex_text(self, content: bytes) -> str:
        """Extract text from LaTeX source."""
        try:
            from pylatexenc.latex2text import LatexNodes2Text

            text = _decode_text(content)
            return LatexNodes2Text().latex_to_text(text)
        except ImportError:
            # Fallback: return raw LaTeX source
            return _decode_text(content)

    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF via PyMuPDF."""
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            pages = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append(text)
            return "\n\n".join(pages)
        finally:
            doc.close()
