"""
Document loader service for the AI Career Assistant.
"""

import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and processes professional documents"""

    @staticmethod
    def load_pdf(path: str) -> str:
        """Load text content from a PDF file"""
        try:
            reader = PdfReader(path)
            content = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    content += text

            # Debug logging for PDF content
            content_length = len(content)
            webrtc_found = "WebRTC" in content
            websocket_found = "WebSocket" in content

            logger.info(f"Loaded PDF: {path} - Length: {content_length} chars")
            logger.info(f"PDF Debug - WebRTC found: {webrtc_found}, WebSocket found: {websocket_found}")

            # Log a snippet around WebRTC if found
            if webrtc_found:
                webrtc_index = content.find("WebRTC")
                snippet = content[max(0, webrtc_index-50):webrtc_index+50]
                logger.info(f"WebRTC context: ...{snippet}...")

            return content
        except Exception as e:
            logger.error(f"Failed to load PDF {path}: {e}")
            return ""

    @staticmethod
    def load_text(path: str) -> str:
        """Load content from a text file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Loaded text file: {path}")
            return content
        except Exception as e:
            logger.error(f"Failed to load text file {path}: {e}")
            return ""