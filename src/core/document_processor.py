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

# 设置日志记录器
beijing_logger = BeijingLogger()
logger = beijing_logger.get_logger()

class DocumentProcessor:
    def __init__(self, max_chunk_size: int = 1000):
        """
        初始化文档处理器，设置最大分块大小。
        
        参数:
            max_chunk_size (int): 每个文本块的最大token数量，默认为1000
        """
        self.max_chunk_size = max_chunk_size
    
    def process_uploaded_files(self, files_list) -> List[Dict[str, Any]]:
        """
        处理从Gradio上传的文件列表。
        
        参数:
            files_list (list): 从Gradio上传的文件路径列表
            
        返回:
            包含提取内容的字典列表，每个字典对应一个文件
        """
        all_documents = []
        
        for file_path in files_list:
            try:
                processed_doc = self.process_single_file(file_path)
                if processed_doc:
                    all_documents.append(processed_doc)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
                continue
                
        return all_documents
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        根据文件扩展名处理单个文件。
        
        参数:
            file_path (str): 文件路径
            
        返回:
            包含提取内容的字典，包括文件名、文件内容和分块信息
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
            # 如果需要处理解压后的文件，可以在这里添加相关逻辑
            return {"file_name": os.path.basename(file_path), "message": "ZIP文件已解压"}
        else:
            return self.read_text_file(file_path)  # 尝试作为文本文件读取
    
    def read_pdf(self, filepath: str) -> Dict[str, Any]:
        """
        使用多种方法从PDF文件中提取内容。
        依次尝试PyMuPDF、pdfplumber和PyPDF2三种方法，直到成功提取内容。
        """
        filename = os.path.basename(filepath)
        result = {'file_extension': 'pdf', 'file_name': filename}

        def clean_text(text):
            """
            清理文本，处理可能的编码问题。
            依次尝试UTF-8、GBK编码，确保文本可读。
            """
            try:
                return text.encode('utf-8', 'ignore').decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return text.encode('utf-8', 'ignore').decode('gbk')
                except UnicodeDecodeError:
                    return text
        
        # 首先尝试使用PyMuPDF (fitz)
        try:
            document = fitz.open(filepath)
            content = []
            for page in document:
                content.append(clean_text(page.get_text()))
            combined_text = clean_text("".join(content))
            logger.info(f"PyMuPDF提取内容: {combined_text[:50]}...")
            if combined_text and not self.is_text_garbled(combined_text):
                result['file_content'] = combined_text
                result['chunks'] = self.split_content_to_chunks(combined_text)
                return result
        except Exception as e:
            logger.error(f"PyMuPDF (fitz) 处理 {filename} 失败: {e}")

        # 如果PyMuPDF失败，尝试使用pdfplumber
        try:
            with pdfplumber.open(filepath) as pdf:
                content = []
                for page in pdf.pages:
                    content.append(clean_text(page.extract_text() or ""))
            combined_text = clean_text("".join(content))
            logger.info(f"pdfplumber提取内容: {combined_text[:50]}...")
            if combined_text and not self.is_text_garbled(combined_text):
                result['file_content'] = combined_text
                result['chunks'] = self.split_content_to_chunks(combined_text)
                return result
        except Exception as e:
            logger.error(f"pdfplumber 处理 {filename} 失败: {e}")

        # 如果前两种方法都失败，最后尝试使用PyPDF2
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = []
                for page in reader.pages:
                    content.append(clean_text(page.extract_text() or ""))
            combined_text = clean_text("".join(content))
            logger.info(f"PyPDF2提取内容: {combined_text[:50]}...")
            if combined_text and not self.is_text_garbled(combined_text):
                result['file_content'] = combined_text
                result['chunks'] = self.split_content_to_chunks(combined_text)
                return result
        except Exception as e:
            logger.error(f"PyPDF2 处理 {filename} 失败: {e}")

        logger.error(f"所有提取方法对 {filename} 都失败了")
        return {}

    def read_docx(self, filepath: str) -> Dict[str, Any]:
        """
        从DOCX文件中提取内容。
        提取文档中的所有段落文本，并保持段落结构。
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
            logger.error(f"读取 {filepath} 时出错: {e}")
            return {}

    def read_text_file(self, filepath: str) -> Dict[str, Any]:
        """
        从文本文件（TXT、MD等）中提取内容。
        支持多种编码格式，包括UTF-8、GBK等。
        """
        try:
            filename = os.path.basename(filepath)
            file_extension = os.path.splitext(filename)[1].lower()
            result = {'file_extension': file_extension, 'file_name': filename}
            
            # 首先尝试使用UTF-8编码
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    file_content = file.read()
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试使用GBK编码
                try:
                    with open(filepath, 'r', encoding='gbk') as file:
                        file_content = file.read()
                except Exception:
                    # 如果GBK也失败，使用chardet检测编码
                    with open(filepath, 'rb') as file:
                        raw_data = file.read()
                        encoding = chardet.detect(raw_data)['encoding']
                        file_content = raw_data.decode(encoding or 'utf-8', errors='ignore')
            
            result['file_content'] = file_content
            result['chunks'] = self.split_content_to_chunks(file_content)
            return result
        except Exception as e:
            logger.error(f"读取 {filepath} 时出错: {e}")
            return {}

    def unzip_file(self, zip_file_path: str) -> str:
        """
        解压ZIP文件并返回解压目录的路径。
        处理可能的文件名编码问题，支持多种编码格式。
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
                            logger.error(f"解码文件名时出错: {zip_info.filename}")
                            continue

                zip_info.filename = filename
                zip_ref.extract(zip_info, extract_to_path)

        return extract_path

    def is_text_garbled(self, text: str) -> bool:
        """
        检查提取的文本是否乱码。
        通过分析中文字符比例和特殊符号比例来判断文本质量。
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
        根据max_chunk_size将内容分割成多个块。
        
        实现了一个基于段落和句子的简单分割策略。
        在生产环境中可以使用更复杂的分割方法。
        """
        # 按段落分割内容
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果段落可以放入当前块，则添加
            if len(current_chunk) + len(paragraph) <= self.max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # 如果当前块不为空，将其添加到块列表中
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果段落小于max_chunk_size，用它开始新的块
                if len(paragraph) <= self.max_chunk_size:
                    current_chunk = paragraph + "\n\n"
                else:
                    # 将大段落分割成句子
                    sentences = re.split(r'(?<=[.!?。！？])\s+', paragraph)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            
                            # 如果句子太长，进一步分割
                            if len(sentence) > self.max_chunk_size:
                                sentence_chunks = [sentence[i:i+self.max_chunk_size] 
                                                 for i in range(0, len(sentence), self.max_chunk_size)]
                                chunks.extend(sentence_chunks[:-1])
                                current_chunk = sentence_chunks[-1] + " "
                            else:
                                current_chunk = sentence + " "
        
        # 如果最后一个块不为空，添加它
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_markdown_by_headings(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        按标题分割markdown文本，以获得更好的文档结构。
        支持多级标题，并保持文档的层次结构。
        """
        def split_paragraphs(text, max_length):
            """
            将文本分割成段落，确保每个段落不超过最大长度。
            基于句子进行分割，保持语义完整性。
            """
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

        # 使用正则表达式按标题分割markdown
        headings = re.split(r'\n\s*(?=#)', markdown_text.strip())
        result = []
        
        for i, section in enumerate(headings):
            if not section.strip():
                continue
                
            # 解析标题级别和内容
            match = re.match(r'(#+)\s+(.+)', section)
            if match:
                heading_level = len(match.group(1))
                heading_content = match.group(2).strip()
                section_content = section[len(match.group(0)):].strip()
                
                # 添加带结构的标题和内容到结果中
                result.append({
                    'heading_level': heading_level,
                    'heading': heading_content,
                    'content': section_content
                })
            else:
                # 处理没有正确标题的部分
                result.append({
                    'heading_level': 0,
                    'heading': f"Section {i+1}",
                    'content': section.strip()
                })
        
        return result
    
