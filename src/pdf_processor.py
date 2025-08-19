"""PDF processing utilities for extracting text and metadata."""

import fitz  # PyMuPDF
import streamlit as st
from typing import Dict, List, Tuple


class PDFProcessor:
    """Handles PDF text extraction with metadata."""
    
    def __init__(self):
        self.supported_formats = ['pdf']
    
    def extract_text_from_pdf(self, file) -> Tuple[str, Dict]:
        """
        Extract raw text from a PDF file with metadata.
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        text = ""
        metadata = {}
        
        try:
            # Read file content
            file_content = file.read()
            doc = fitz.open(stream=file_content, filetype="pdf")
            
            # Extract metadata
            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', 'Unknown'),
                'author': doc.metadata.get('author', 'Unknown'),
                'file_name': file.name,
                'file_size': len(file_content)
            }
            
            # Extract text from each page
            page_texts = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                text += page_text + "\n"
                
                page_texts.append({
                    'page_num': page_num + 1,
                    'text': page_text,
                    'word_count': len(page_text.split()),
                    'char_count': len(page_text)
                })
            
            metadata['pages'] = page_texts
            metadata['total_words'] = sum(page['word_count'] for page in page_texts)
            metadata['total_chars'] = len(text)
            
            doc.close()
            
        except Exception as e:
            st.error(f"Error extracting from {file.name}: {str(e)}")
            return "", {}
        
        return text, metadata
    
    def extract_from_multiple_files(self, files: List) -> Tuple[List[str], List[Dict]]:
        """
        Extract text from multiple PDF files.
        
        Args:
            files: List of Streamlit uploaded file objects
            
        Returns:
            Tuple of (list_of_texts, list_of_metadata)
        """
        texts = []
        metadata_list = []
        
        for file in files:
            text, metadata = self.extract_text_from_pdf(file)
            if text:  # Only add if extraction was successful
                texts.append(text)
                metadata_list.append(metadata)
        
        return texts, metadata_list
    
    def validate_file(self, file) -> bool:
        """
        Validate if the uploaded file is supported.
        
        Args:
            file: Streamlit uploaded file object
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        if not file:
            return False
        
        # Check file extension
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in self.supported_formats:
            st.error(f"Unsupported file format: {file_extension}")
            return False
        
        # Check file size (optional)
        if hasattr(file, 'size') and file.size > 50 * 1024 * 1024:  # 50MB
            st.error("File size too large. Maximum size is 50MB.")
            return False
        
        return True
    
    @staticmethod
    def get_file_summary(metadata_list: List[Dict]) -> Dict:
        """
        Generate summary statistics from file metadata.
        
        Args:
            metadata_list: List of metadata dictionaries
            
        Returns:
            Dict with summary statistics
        """
        if not metadata_list:
            return {}
        
        total_pages = sum(meta['page_count'] for meta in metadata_list)
        total_words = sum(meta['total_words'] for meta in metadata_list)
        total_files = len(metadata_list)
        
        avg_pages = total_pages / total_files if total_files > 0 else 0
        avg_words = total_words / total_files if total_files > 0 else 0
        
        return {
            'total_files': total_files,
            'total_pages': total_pages,
            'total_words': total_words,
            'avg_pages_per_file': round(avg_pages, 1),
            'avg_words_per_file': round(avg_words, 1),
            'file_names': [meta['file_name'] for meta in metadata_list]
        }