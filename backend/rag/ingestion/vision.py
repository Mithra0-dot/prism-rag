"""
PRISM — Precision Retrieval with Intelligent Semantic Multimodal
----------------------------------------------------------------
backend/rag/ingestion/vision.py

Computer Vision pipeline for extracting text from images and charts
embedded in PDF documents.

Why this matters:
  A financial report PDF may have a bar chart showing quarterly revenue.
  Standard text extraction (pypdf) sees nothing — the chart is an image.
  This module uses OpenCV to detect image regions and EasyOCR to read
  text from them, making that data available for RAG retrieval.

Pipeline per PDF page:
  1. Render PDF page as high-resolution image (via pdf2image/fitz)
  2. Detect image/chart regions with OpenCV (contour detection)
  3. For each detected region, run EasyOCR text extraction
  4. Merge extracted text with the page's regular text content
  5. Return enriched Document objects

Phase 2 upgrade:
  The DocumentLoader.load() method checks if vision=True is passed
  and calls this module after standard text extraction.

Usage:
    from backend.rag.ingestion.vision import VisionExtractor

    extractor = VisionExtractor()
    enriched_docs = extractor.extract(pdf_path=Path("report.pdf"))
"""

import time
from pathlib import Path
from typing import Optional

from langchain.schema import Document

from backend.core.config import get_settings
from backend.core.logger import get_logger
from backend.core.exceptions import VisionProcessingError, DocumentIngestionError

settings = get_settings()
log = get_logger("vision")


class VisionExtractor:
    """
    Extracts text from images and charts embedded in PDF pages.

    Uses:
      - PyMuPDF (fitz)  : renders PDF pages as images
      - OpenCV          : detects image/chart regions via contour analysis
      - EasyOCR         : reads text from detected regions

    Args:
        languages      : List of language codes for OCR (default: English)
        min_confidence : Minimum OCR confidence score to include text (0-1)
        gpu            : Use GPU for EasyOCR if available (faster)
        min_region_area: Minimum pixel area for a region to be processed
    """

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        min_confidence: float = 0.3,
        gpu: bool = False,
        min_region_area: int = 5000,
    ) -> None:
        self.languages = languages or ["en"]
        self.min_confidence = min_confidence
        self.gpu = gpu
        self.min_region_area = min_region_area
        self._reader = None   # lazy-loaded EasyOCR reader

        log.info(
            f"VisionExtractor configured | "
            f"languages={self.languages} | "
            f"min_confidence={self.min_confidence} | "
            f"gpu={self.gpu}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def extract(
        self,
        pdf_path: Path,
        filename: Optional[str] = None,
        dpi: int = 200,
    ) -> list[Document]:
        """
        Extract text from images/charts in a PDF file.

        Args:
            pdf_path : Path to the PDF file
            filename : Original filename for metadata
            dpi      : Render resolution (higher = better OCR, slower)

        Returns:
            List of Document objects — one per page with enriched content
            that includes both regular text AND OCR-extracted text.

        Raises:
            VisionProcessingError : if image processing fails
            DocumentIngestionError: if PDF cannot be opened
        """
        filename = filename or pdf_path.name
        log.info(f"Starting vision extraction for '{filename}'")
        start = time.perf_counter()

        try:
            import fitz   # PyMuPDF
        except ImportError:
            raise VisionProcessingError(
                message="PyMuPDF not installed. Run: pip install pymupdf",
                detail="Required for PDF-to-image rendering",
            )

        try:
            pdf_doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise DocumentIngestionError(
                message=f"Could not open PDF: {filename}",
                detail=str(e),
            )

        documents: list[Document] = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]

            # ── 1. Extract regular text ───────────────────────────────
            regular_text = page.get_text("text").strip()

            # ── 2. Render page as image ───────────────────────────────
            try:
                page_image = self._render_page(page, dpi=dpi)
            except Exception as e:
                log.warning(f"Failed to render page {page_num + 1}: {e}")
                # Fall back to text-only for this page
                if regular_text:
                    documents.append(self._build_document(
                        text=regular_text,
                        ocr_text="",
                        filename=filename,
                        page_num=page_num + 1,
                        total_pages=len(pdf_doc),
                    ))
                continue

            # ── 3. Detect image/chart regions ─────────────────────────
            regions = self._detect_regions(page_image)
            log.debug(
                f"Page {page_num + 1}: {len(regions)} visual regions detected"
            )

            # ── 4. OCR each region ────────────────────────────────────
            ocr_texts: list[str] = []
            for region in regions:
                ocr_text = self._ocr_region(page_image, region)
                if ocr_text:
                    ocr_texts.append(ocr_text)

            combined_ocr = "\n".join(ocr_texts)

            # ── 5. Build enriched document ────────────────────────────
            if regular_text or combined_ocr:
                documents.append(self._build_document(
                    text=regular_text,
                    ocr_text=combined_ocr,
                    filename=filename,
                    page_num=page_num + 1,
                    total_pages=len(pdf_doc),
                ))

        pdf_doc.close()

        elapsed = time.perf_counter() - start
        log.info(
            f"Vision extraction complete for '{filename}' | "
            f"pages={len(documents)} | "
            f"time={elapsed:.2f}s"
        )

        return documents

    def extract_from_image(self, image_path: Path) -> str:
        """
        Run OCR on a standalone image file (PNG, JPG, etc).

        Used when a user uploads an image directly rather than a PDF.

        Args:
            image_path : Path to the image file

        Returns:
            Extracted text string

        Raises:
            VisionProcessingError : if OCR fails
        """
        try:
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                raise VisionProcessingError(
                    message=f"Could not read image: {image_path.name}",
                    detail="OpenCV returned None — check file format",
                )
            return self._run_ocr(image)
        except VisionProcessingError:
            raise
        except Exception as e:
            raise VisionProcessingError(
                message=f"Image OCR failed for '{image_path.name}'",
                detail=str(e),
            ) from e

    # ── Private — Rendering ───────────────────────────────────────────

    def _render_page(self, page, dpi: int = 200):
        """
        Render a PDF page as a numpy array (OpenCV-compatible image).

        PyMuPDF renders to a Pixmap, which we convert to numpy for OpenCV.
        DPI of 200 gives good OCR quality without excessive memory use.
        """
        import numpy as np

        matrix = page.get_transformation_matrix(dpi / 72)   # 72 is PDF base DPI
        pixmap = page.get_pixmap(matrix=matrix, colorspace="rgb")
        image_bytes = pixmap.tobytes("png")

        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    # ── Private — Region Detection ────────────────────────────────────

    def _detect_regions(self, image) -> list[tuple]:
        """
        Detect image/chart regions using OpenCV contour detection.

        Algorithm:
          1. Convert to grayscale
          2. Apply binary threshold (separates foreground from background)
          3. Find contours (closed boundaries)
          4. Filter contours by minimum area (ignore tiny noise regions)
          5. Return bounding boxes of candidate regions

        This approach works well for charts, diagrams, tables with borders,
        and any image embedded in the PDF with a visible boundary.

        Returns:
            List of (x, y, w, h) bounding box tuples
        """
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Binary threshold — pixels below 200 become black (foreground)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # Find contours — RETR_EXTERNAL gets only outermost boundaries
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Filter: must be large enough to be a real chart/image
                # and not so large it's the whole page
                page_area = image.shape[0] * image.shape[1]
                if self.min_region_area < area < (page_area * 0.9):
                    regions.append((x, y, w, h))

            return regions

        except Exception as e:
            log.warning(f"Region detection failed: {e}")
            return []

    # ── Private — OCR ────────────────────────────────────────────────

    def _ocr_region(self, image, region: tuple) -> str:
        """
        Crop a region from the page image and run EasyOCR on it.

        Args:
            image  : Full page image as numpy array
            region : (x, y, w, h) bounding box

        Returns:
            Extracted text string, or empty string if nothing found
        """
        try:

            x, y, w, h = region
            # Add padding around the region for better OCR accuracy
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            cropped = image[y1:y2, x1:x2]
            return self._run_ocr(cropped)

        except Exception as e:
            log.debug(f"OCR failed for region {region}: {e}")
            return ""

    def _run_ocr(self, image) -> str:
        """
        Run EasyOCR on an image and return the extracted text.

        Filters results by confidence score to reduce noise.
        """
        reader = self._get_reader()

        try:
            results = reader.readtext(image)
            texts = [
                text
                for (_, text, confidence) in results
                if confidence >= self.min_confidence and text.strip()
            ]
            return " ".join(texts)
        except Exception as e:
            log.debug(f"EasyOCR failed: {e}")
            return ""

    def _get_reader(self):
        """Lazy-load the EasyOCR reader (takes ~5 seconds first time)."""
        if self._reader is None:
            log.info(f"Loading EasyOCR reader | languages={self.languages}")
            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    verbose=False,
                )
                log.info("EasyOCR reader loaded")
            except Exception as e:
                raise VisionProcessingError(
                    message="Failed to load EasyOCR",
                    detail=str(e),
                )
        return self._reader

    # ── Private — Document Builder ────────────────────────────────────

    def _build_document(
        self,
        text: str,
        ocr_text: str,
        filename: str,
        page_num: int,
        total_pages: int,
    ) -> Document:
        """
        Merge regular text and OCR text into a single enriched Document.

        The OCR text is appended with a clear label so the LLM knows
        it came from a visual element (chart, diagram, image).
        """
        from datetime import datetime

        content_parts = []

        if text:
            content_parts.append(text)

        if ocr_text:
            content_parts.append(
                f"\n[VISUAL CONTENT EXTRACTED VIA OCR]\n{ocr_text}"
            )

        full_content = "\n".join(content_parts).strip()

        return Document(
            page_content=full_content,
            metadata={
                "source":       filename,
                "file_type":    "pdf",
                "page":         page_num,
                "total_pages":  total_pages,
                "has_ocr":      bool(ocr_text),
                "ocr_chars":    len(ocr_text),
                "ingested_at":  datetime.utcnow().isoformat(),
                "pipeline":     "vision",
            },
        )
