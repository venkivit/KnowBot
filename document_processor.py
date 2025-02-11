import PyPDF2
from typing import List, Optional
import io

class DocumentProcessor:
    @staticmethod
    def process_text_file(content: bytes) -> List[str]:
        """Process text file content into documents"""
        try:
            text_content = content.decode()
            # Split by double newlines and filter empty strings
            documents = [p.strip() for p in text_content.split('\n\n') if p.strip()]
            return documents
        except Exception as e:
            print(f"Error processing text file: {str(e)}")
            return []

    @staticmethod
    def process_pdf_file(content: bytes) -> List[str]:
        """Process PDF file content into documents"""
        try:
            # Create a PDF file reader object
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            documents = []
            for page in pdf_reader.pages:
                text = page.extract_text().strip()
                if text:  # Only add non-empty pages
                    # Split long pages into smaller chunks
                    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
                    documents.extend(chunks)
            
            return documents
        except Exception as e:
            print(f"Error processing PDF file: {str(e)}")
            return []

    @staticmethod
    def process_file(content: bytes, file_type: str) -> List[str]:
        """Process file content based on file type"""
        if file_type.lower() == 'pdf':
            return DocumentProcessor.process_pdf_file(content)
        else:  # Default to text processing
            return DocumentProcessor.process_text_file(content)
