import os
import re
import zipfile
import docx
import chardet
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import io
import requests
from typing import Dict, List, Any, Optional
from ..utils.logger import BeijingLogger
from dotenv import load_dotenv
import time
from io import BytesIO

# 加载环境变量
load_dotenv()

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
        # 从环境变量获取MinerU API URL
        self.ocr_api_url = os.getenv('MINERU_API_URL', '')
    
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
        首先尝试OCR API，然后依次尝试pymupdf4llm、PyMuPDF、pdfplumber和PyPDF2，直到成功提取内容。
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
                    
        # 0. 首先尝试使用Mineru API处理
        mineru_mode = os.getenv('MINERU_MODE', '')
        if mineru_mode:
            try:
                logger.info(f"尝试使用Mineru API ({mineru_mode})提取文件 {filename}...")
                with open(filepath, 'rb') as file:
                    pdf = PyPDF2.PdfReader(file)
                    num_pages = len(pdf.pages)
                    all_markdown_content = []
                    
                    for i in range(0, num_pages, 20):
                        logger.info(f"正在处理第 {i//20+1} 页到第 {min(i + 20, num_pages)//20+1} 页")
                        end_page = min(i + 20, num_pages)
                        
                        # 创建一个新的 PDF 写入器
                        pdf_writer = PyPDF2.PdfWriter()
                        for page_num in range(i, end_page):
                            pdf_writer.add_page(pdf.pages[page_num])
                        
                        # 将切分后的 PDF 保存到内存中
                        temp_pdf = io.BytesIO()
                        pdf_writer.write(temp_pdf)
                        temp_pdf.seek(0)
                        
                        # 根据MINERU_MODE处理该部分PDF
                        temp_file_path = f"/tmp/{filename}_part_{i//20+1}.pdf"
                        with open(temp_file_path, 'wb') as tmp_file:
                            tmp_file.write(temp_pdf.getvalue())
                        
                        try:
                            markdown_content = ""
                            if mineru_mode == 'web_api':
                                markdown_content = self.parse_pdf_to_markdown_mineru_web_api(temp_file_path)
                            elif mineru_mode == 'local_api':
                                markdown_content = self.parse_pdf_to_markdown_mineru_local_api(temp_file_path)
                            else:
                                logger.info(f"未知的MINERU_MODE值: {mineru_mode}，跳过Mineru API处理")
                                
                            if markdown_content:
                                all_markdown_content.append(markdown_content)
                            
                            # 删除临时文件
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
                                
                        except Exception as e:
                            logger.error(f"处理PDF部分 {i//20+1} 失败: {e}")
                            # 继续处理其他部分
                    
                    if all_markdown_content:
                        combined_ocr_text = clean_text("".join(all_markdown_content))
                        logger.info(f"Mineru API提取内容: {combined_ocr_text[:50]}...")
                        if combined_ocr_text and not self.is_text_garbled(combined_ocr_text):
                            result['file_content'] = combined_ocr_text
                            result['chunks'] = self.split_content_to_chunks(combined_ocr_text)
                            return result
                    logger.info(f"Mineru API结果为空或乱码，文件: {filename}")
                
            except Exception as e:
                logger.error(f"Mineru API ({mineru_mode})处理 {filename} 失败: {e}")
        else:
            logger.info(f"MINERU_MODE环境变量未设置，跳过Mineru API处理步骤")

        # 0.5. 尝试使用pymupdf4llm
        try:
            import pymupdf4llm
            logger.info(f"尝试使用pymupdf4llm提取文件 {filename}...")
            
            # 使用pymupdf4llm提取PDF内容为Markdown格式
            md_text = pymupdf4llm.to_markdown(
                filepath,
                force_text=True,
                show_progress=False,  # 不显示进度条
                write_images=False,   # 不写出图片
                embed_images=False    # 不嵌入图片
            )
            
            if md_text:
                # 清理文本
                md_text = clean_text(md_text)
                logger.info(f"pymupdf4llm提取内容: {md_text[:50]}...")
                
                if md_text and not self.is_text_garbled(md_text):
                    result['file_content'] = md_text
                    result['chunks'] = self.split_content_to_chunks(md_text)
                    return result
                logger.info(f"pymupdf4llm结果为空或乱码，文件: {filename}")
        except ImportError:
            logger.info("pymupdf4llm未安装，跳过此提取方法")
        except Exception as e:
            logger.error(f"pymupdf4llm处理 {filename} 失败: {e}")
        
        # 1. 尝试使用 PyMuPDF (fitz)
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

        # 2. 尝试使用 pdfplumber
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

        # 3. 尝试使用 PyPDF2
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
    
    def parse_pdf_to_markdown_mineru_web_api(self, pdf_path, is_ocr=False, enable_formula=True, enable_table=True, save_to_file=False, output_dir="output/mineru"):
        """
        使用Mineru Web API将本地PDF文件解析为Markdown内容
        
        参数:
            pdf_path (str): 本地PDF文件的路径
            is_ocr (bool, optional): 是否使用OCR。默认为False
            enable_formula (bool, optional): 是否启用公式识别。默认为True
            enable_table (bool, optional): 是否启用表格识别。默认为True
            save_to_file (bool, optional): 是否将Markdown保存到文件。默认为False
            output_dir (str, optional): 保存Markdown文件的目录。默认为"output/mineru"
            
        返回:
            str: PDF的Markdown内容
        """
        # 步骤1: 获取文件上传URL
        upload_url = f"{os.getenv('MINERU_API_URL')}/file-urls/batch"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {os.getenv('MINERU_API_KEY')}"
        }
        
        filename = os.path.basename(pdf_path)
        data = {
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "files": [{"name": filename, "is_ocr": is_ocr}]
        }
        
        try:
            response = requests.post(upload_url, headers=headers, json=data)
            if response.status_code != 200:
                raise Exception(f"获取上传URL失败: {response.text}")
            
            result = response.json()
            if result["code"] != 0:
                raise Exception(f"API错误: {result['msg']}")
            
            batch_id = result["data"]["batch_id"]
            file_url = result["data"]["file_urls"][0]
            
            # 步骤2: 上传文件
            with open(pdf_path, 'rb') as f:
                upload_response = requests.put(file_url, data=f)
                if upload_response.status_code != 200:
                    raise Exception(f"上传文件失败: {upload_response.text}")
            
            # 步骤3: 轮询结果
            status_url = f"{os.getenv('MINERU_API_URL')}/extract-results/batch/{batch_id}"
            max_retries = 60  # 最大等待时间: 10分钟
            wait_time = 10  # 秒
            
            for _ in range(max_retries):
                time.sleep(wait_time)
                status_response = requests.get(status_url, headers=headers)
                
                if status_response.status_code != 200:
                    raise Exception(f"获取任务状态失败: {status_response.text}")
                
                status_result = status_response.json()
                if status_result["code"] != 0:
                    raise Exception(f"API错误: {status_result['msg']}")
                
                extract_results = status_result["data"]["extract_result"]
                for result in extract_results:
                    if result["file_name"] == filename:
                        if result["state"] == "done":
                            # 步骤4: 下载结果
                            zip_url = result["full_zip_url"]
                            zip_response = requests.get(zip_url)
                            
                            if zip_response.status_code != 200:
                                raise Exception(f"下载结果失败: {zip_response.text}")
                            
                            # 步骤5: 提取Markdown内容
                            with zipfile.ZipFile(BytesIO(zip_response.content)) as z:
                                markdown_files = [f for f in z.namelist() if f.endswith('.md')]
                                if not markdown_files:
                                    raise Exception("在结果中未找到Markdown文件")
                                
                                # 获取Markdown内容
                                markdown_content = z.read(markdown_files[0]).decode('utf-8')
                                
                                # 如果需要，将Markdown保存到文件
                                if save_to_file:
                                    # 如果输出目录不存在，创建它
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    # 生成输出文件名
                                    base_filename = os.path.splitext(filename)[0]
                                    output_path = os.path.join(output_dir, f"{base_filename}.md")
                                    
                                    # 将Markdown内容保存到文件
                                    with open(output_path, 'w', encoding='utf-8') as file:
                                        file.write(markdown_content)
                                    print(f"Markdown内容已保存至: {output_path}")
                                
                                return markdown_content
                        
                        elif result["state"] == "failed":
                            raise Exception(f"任务失败: {result.get('err_msg', '未知错误')}")
            
            raise Exception("任务处理超时")
        
        except Exception as e:
            raise Exception(f"解析PDF出错: {str(e)}")

    def parse_pdf_to_markdown_mineru_local_api(self, pdf_path, api_url=None, save_to_file=False, output_dir="output/mineru"):
        """
        使用Mineru本地API将PDF文件解析为Markdown内容
        
        参数:
            pdf_path (str): 本地PDF文件的路径
            api_url (str, optional): 本地Mineru端点的API URL。如果为None，则使用环境变量中的MINERU_API_URL
            save_to_file (bool, optional): 是否将Markdown保存到文件。默认为False
            output_dir (str, optional): 保存Markdown文件的目录。默认为"output/mineru"
            
        返回:
            str: PDF的Markdown内容
        """
        try:
            filename = os.path.basename(pdf_path)
            
            # 使用提供的API URL或从环境变量获取
            url = api_url or os.getenv('MINERU_API_URL', 'http://localhost:8000/pdf_parse?parse_method=auto')
            
            logger.info(f"使用Mineru本地API解析PDF: {filename}")
            
            # 准备请求
            payload = {}
            files = [
                ('pdf_file', (filename, open(pdf_path, 'rb'), 'application/pdf'))
            ]
            headers = {
                'accept': 'application/json'
            }
            
            # 发送请求
            response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=600)
            
            # 检查响应状态
            if response.status_code != 200:
                raise Exception(f"API请求失败，状态码 {response.status_code}: {response.text}")
            
            # 解析响应
            response_data = response.json()
            content_list = response_data.get('content_list', [])
            md_content = response_data.get('md_content', '')
            
            # 如果需要，将Markdown保存到文件
            if save_to_file and md_content:
                # 如果输出目录不存在，创建它
                os.makedirs(output_dir, exist_ok=True)
                
                # 生成输出文件名
                base_filename = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_filename}.md")
                
                # 将Markdown内容保存到文件
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(md_content)
                logger.info(f"Markdown内容已保存至: {output_path}")
            
            return md_content
            
        except Exception as e:
            logger.error(f"使用Mineru本地API解析PDF时出错: {str(e)}")
            raise Exception(f"解析PDF出错: {str(e)}")
