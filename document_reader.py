"""
Document Reader Module

This module provides functionality to read and extract text from various document formats
including PDF files and text files.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union
import chardet
from pypdf import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentReader:
    """A class to read and extract text from various document formats."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', '.json'}
    
    def __init__(self):
        """Initialize the DocumentReader."""
        pass
    
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file type is supported, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def read_document(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Read and extract text from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted text content, or None if reading failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if not self.is_supported_file(file_path):
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return None
        
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._read_pdf(file_path)
            else:
                return self._read_text_file(file_path)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _read_pdf(self, file_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            reader = PdfReader(str(file_path))
            text_content = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                    continue
            
            if not text_content:
                logger.warning(f"No text content extracted from PDF: {file_path}")
                return ""
            
            return "\n\n".join(text_content)
        
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            raise
    
    def _read_text_file(self, file_path: Path) -> str:
        """
        Read text from a text file with automatic encoding detection.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            str: File content as text
        """
        try:
            # First, try to detect encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                content = file.read()
            
            return content
        
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {str(e)}")
            # Fallback: try with utf-8 and ignore errors
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    return file.read()
            except Exception as fallback_error:
                logger.error(f"Fallback read also failed: {str(fallback_error)}")
                raise
    
    def get_file_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get basic information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            dict: File information including size, type, etc.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        try:
            stat_info = file_path.stat()
            return {
                "name": file_path.name,
                "size": stat_info.st_size,
                "extension": file_path.suffix,
                "is_supported": self.is_supported_file(file_path),
                "modified_time": stat_info.st_mtime
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """Test the DocumentReader functionality."""
    reader = DocumentReader()
    
    # Test with a sample file (you can modify this path)
    test_file = "sample.txt"
    
    if Path(test_file).exists():
        print(f"Reading {test_file}...")
        content = reader.read_document(test_file)
        if content:
            print(f"Content length: {len(content)} characters")
            print(f"Preview: {content[:200]}...")
        else:
            print("Failed to read content")
    else:
        print(f"Test file {test_file} not found. Create one to test the reader.")


if __name__ == "__main__":
    main()