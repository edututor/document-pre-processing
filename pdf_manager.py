from loguru import logger
from typing import Union, BinaryIO
from io import BytesIO
import pdfplumber
from docx import Document
import magic
import PyPDF2

class PDFManagerError(Exception):
    """Custom exception for PDF manager errors"""
    pass

class PdfManager:
    SUPPORTED_TYPES = {
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
    }

    def __init__(self):
        self.file_handlers = {
            'pdf': self._handle_pdf,
            'docx': self._handle_docx
        }

    def detect_file_type(self, file_content: bytes) -> str:
        """Detect file type using python-magic"""
        mime = magic.from_buffer(file_content, mime=True)
        if mime not in self.SUPPORTED_TYPES:
            raise PDFManagerError(f"Unsupported file type: {mime}")
        return self.SUPPORTED_TYPES[mime]

    def validate_pdf(self, file_content: bytes) -> bool:
        """Validate PDF file integrity"""
        try:
            PyPDF2.PdfReader(BytesIO(file_content))
            return True
        except Exception as e:
            raise PDFManagerError(f"Invalid PDF file: {str(e)}")

    def _handle_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            self.validate_pdf(file_content)
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise PDFManagerError(f"PDF processing failed: {str(e)}")

    def _handle_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            document = Document(BytesIO(file_content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as e:
            raise PDFManagerError(f"DOCX processing failed: {str(e)}")

    def extract_text(self, file_content: bytes) -> str:
        """
        Extract text from supported file types
        
        Args:
            file_content (bytes): File content
            
        Returns:
            str: Extracted text
            
        Raises:
            PDFManagerError: If text extraction fails
        """
        try:
            logger.info("Starting text extraction")
            file_type = self.detect_file_type(file_content)
            
            handler = self.file_handlers.get(file_type)
            if not handler:
                raise PDFManagerError(f"No handler for file type: {file_type}")
            
            text = handler(file_content)
            logger.success(f"Text extracted successfully: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise PDFManagerError(f"Failed to extract text: {str(e)}")
