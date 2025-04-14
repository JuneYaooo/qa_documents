import os
import re
import zipfile
import docx
import chardet
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional
from ..utils.logger import BeijingLogger

# Set up logger
beijing_logger = BeijingLogger()
logger = beijing_logger.get_logger()

class DocumentProcessor:
    def __init__(self, max_chunk_size: int = 1000):
        """
        Initialize the document processor with a maximum chunk size.
        
        Args:
            max_chunk_size (int): Maximum number of tokens in a chunk
        """
        self.max_chunk_size = max_chunk_size
    
    def process_uploaded_files(self, files_list) -> List[Dict[str, Any]]:
        """
        Process a list of uploaded files from Gradio.
        
        Args:
            files_list (list): List of file paths from Gradio uploads
            
        Returns:
            List of dictionaries containing extracted content
        """
        all_documents = []
        
        for file_path in files_list:
            try:
                processed_doc = self.process_single_file(file_path)
                if processed_doc:
                    all_documents.append(processed_doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
                
        return all_documents
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file based on its extension.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Dictionary containing extracted content
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.read_pdf(file_path)
        elif file_extension == '.docx':
            return self.read_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return self.read_text_file(file_path)
        elif file_extension == '.zip':
            extract_path = self.unzip_file(file_path)
            # Handle the extracted files here if needed
            return {"file_name": os.path.basename(file_path), "message": "Zip file extracted"}
        else:
            return self.read_text_file(file_path)  # Attempt to read as text file
    
    def read_pdf(self, filepath: str) -> Dict[str, Any]:
        """
        Extract content from PDF files using multiple methods.
        """
        filename = os.path.basename(filepath)
        result = {'file_extension': 'pdf', 'file_name': filename}

        def clean_text(text):
            try:
                return text.encode('utf-8', 'ignore').decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return text.encode('utf-8', 'ignore').decode('gbk')
                except UnicodeDecodeError:
                    return text
        
        # Try PyMuPDF (fitz) first
        try:
            document = fitz.open(filepath)
            content = []
            for page in document:
                content.append(clean_text(page.get_text()))
            combined_text = clean_text("".join(content))
            logger.info(f"PyMuPDF extracted: {combined_text[:50]}...")
            if combined_text and not self.is_text_garbled(combined_text):
                result['file_content'] = combined_text
                result['chunks'] = self.split_content_to_chunks(combined_text)
                return result
        except Exception as e:
            logger.error(f"PyMuPDF (fitz) failed for {filename}: {e}")

        # Try pdfplumber
        try:
            with pdfplumber.open(filepath) as pdf:
                content = []
                for page in pdf.pages:
                    content.append(clean_text(page.extract_text() or ""))
            combined_text = clean_text("".join(content))
            logger.info(f"pdfplumber extracted: {combined_text[:50]}...")
            if combined_text and not self.is_text_garbled(combined_text):
                result['file_content'] = combined_text
                result['chunks'] = self.split_content_to_chunks(combined_text)
                return result
        except Exception as e:
            logger.error(f"pdfplumber failed for {filename}: {e}")

        # Try PyPDF2 as last resort
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = []
                for page in reader.pages:
                    content.append(clean_text(page.extract_text() or ""))
            combined_text = clean_text("".join(content))
            logger.info(f"PyPDF2 extracted: {combined_text[:50]}...")
            if combined_text and not self.is_text_garbled(combined_text):
                result['file_content'] = combined_text
                result['chunks'] = self.split_content_to_chunks(combined_text)
                return result
        except Exception as e:
            logger.error(f"PyPDF2 failed for {filename}: {e}")

        logger.error(f"All extraction methods failed for {filename}")
        return {}

    def read_docx(self, filepath: str) -> Dict[str, Any]:
        """
        Extract content from DOCX files.
        """
        try:
            filename = os.path.basename(filepath)
            result = {'file_extension': 'docx', 'file_name': filename}
            doc = docx.Document(filepath)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            result['file_content'] = '\n'.join(full_text)
            result['chunks'] = self.split_content_to_chunks(result['file_content'])
            return result
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return {}

    def read_text_file(self, filepath: str) -> Dict[str, Any]:
        """
        Extract content from text files (TXT, MD, etc).
        """
        try:
            filename = os.path.basename(filepath)
            file_extension = os.path.splitext(filename)[1].lower()
            result = {'file_extension': file_extension, 'file_name': filename}
            
            # Try UTF-8 first
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except UnicodeDecodeError:
                # Fall back to GBK
                try:
                    with open(filepath, 'r', encoding='gbk') as file:
                        file_content = file.read()
                except Exception:
                    # Use chardet to detect encoding
                    with open(filepath, 'rb') as file:
                        raw_data = file.read()
                        encoding = chardet.detect(raw_data)['encoding']
                        file_content = raw_data.decode(encoding or 'utf-8', errors='ignore')
            
            result['file_content'] = file_content
            result['chunks'] = self.split_content_to_chunks(file_content)
            return result
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return {}

    def unzip_file(self, zip_file_path: str) -> str:
        """
        Extract a ZIP file and return the path to the extracted directory.
        """
        extract_to_path = os.path.dirname(zip_file_path)
        extract_path = zip_file_path.rsplit('.', 1)[0]
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_info_list = zip_ref.infolist()
            for zip_info in zip_info_list:
                try:
                    filename = zip_info.filename.encode('cp437').decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        filename = zip_info.filename.encode('cp437').decode('gbk')
                    except UnicodeDecodeError:
                        try:
                            detected_encoding = chardet.detect(zip_info.filename.encode('utf-8'))
                            encoding = detected_encoding['encoding']
                            filename = zip_info.filename.encode('utf-8').decode(encoding)
                        except (UnicodeDecodeError, TypeError):
                            logger.error(f"Error decoding filename: {zip_info.filename}")
                            continue

                zip_info.filename = filename
                zip_ref.extract(zip_info, extract_to_path)

        return extract_path

    def is_text_garbled(self, text: str) -> bool:
        """
        Check if the extracted text is garbled.
        """
        chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
        symbol_characters = re.findall(r'[\u0000-\u0020\u3000\uFFFD]', text)

        if len(chinese_characters) > 0:
            chinese_ratio = len(chinese_characters) / max(len(text), 1)
            symbol_ratio = len(symbol_characters) / max(len(text), 1)
            return chinese_ratio < 0.2 or symbol_ratio > 0.3

        non_ascii_ratio = sum(1 for char in text if ord(char) > 127) / max(len(text), 1)
        return non_ascii_ratio > 0.3
    
    def split_content_to_chunks(self, content: str) -> List[str]:
        """
        Split the content into chunks based on max_chunk_size.
        
        This function implements a simple splitting strategy based on paragraphs and sentences.
        More sophisticated methods could be used in a production environment.
        """
        # Split content by paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph can fit in the current chunk, add it
            if len(current_chunk) + len(paragraph) <= self.max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph is smaller than max_chunk_size, start a new chunk with it
                if len(paragraph) <= self.max_chunk_size:
                    current_chunk = paragraph + "\n\n"
                else:
                    # Split large paragraphs into sentences
                    sentences = re.split(r'(?<=[.!?。！？])\s+', paragraph)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            
                            # If sentence is too long, split it further
                            if len(sentence) > self.max_chunk_size:
                                sentence_chunks = [sentence[i:i+self.max_chunk_size] 
                                                 for i in range(0, len(sentence), self.max_chunk_size)]
                                chunks.extend(sentence_chunks[:-1])
                                current_chunk = sentence_chunks[-1] + " "
                            else:
                                current_chunk = sentence + " "
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_markdown_by_headings(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        Split markdown text by headings for better document structure.
        """
        def split_paragraphs(text, max_length):
            sentences = re.split(r'(?<=[。！？])', text)
            paragraphs = []
            current_paragraph = ""
            
            for sentence in sentences:
                if len(current_paragraph) + len(sentence) > max_length:
                    if current_paragraph.strip():
                        paragraphs.append(current_paragraph.strip())
                    current_paragraph = sentence
                else:
                    current_paragraph += sentence
            
            if current_paragraph.strip():
                paragraphs.append(current_paragraph.strip())
            
            return paragraphs

        # Use regex to split markdown by headings
        headings = re.split(r'\n\s*(?=#)', markdown_text.strip())
        result = []
        
        for i, section in enumerate(headings):
            if not section.strip():
                continue
                
            # Parse the heading level and content
            match = re.match(r'(#+)\s+(.+)', section)
            if match:
                heading_level = len(match.group(1))
                heading_content = match.group(2).strip()
                section_content = section[len(match.group(0)):].strip()
                
                # Add to result with proper structure
                result.append({
                    'heading_level': heading_level,
                    'heading': heading_content,
                    'content': section_content
                })
            else:
                # Handle sections without proper headings
                result.append({
                    'heading_level': 0,
                    'heading': f"Section {i+1}",
                    'content': section.strip()
                })
        
        return result
    
