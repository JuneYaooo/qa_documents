import os
import json
import logging
import time
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from ..utils.logger import BeijingLogger
from ..utils.json_utils import JsonUtils

# 加载环境变量
load_dotenv()

# 设置日志记录器
beijing_logger = BeijingLogger()
logger = beijing_logger.get_logger()

class QAExtractor:
    def __init__(self):
        """
        初始化QA提取器，配置OpenAI API凭证。
        设置API密钥、基础URL和模型名称等关键参数。
        如果环境变量中没有API密钥，将抛出异常。
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        
        if not self.api_key:
            raise ValueError("在环境变量中未找到OpenAI API密钥")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"QA提取器已初始化，使用模型: {self.model_name}")
    
    def extract_qa_pairs(self, document: Dict[str, Any], prompt: str) -> List[Dict[str, Any]]:
        """
        使用OpenAI从文档中提取问答对。
        
        参数:
            document: 包含文档内容和元数据的字典
                     必须包含'chunks'或'file_content'字段
            prompt: 自定义提示词，用于指导AI生成问答对
            
        返回:
            问答对列表，每个问答对为字典格式，包含问题和答案
        """
        all_qa_pairs = []
        chunks = document.get('chunks', [])
        
        if not chunks and 'file_content' in document:
            # 如果没有分块但存在文件内容，则使用整个文档作为一个块
            chunks = [document['file_content']]
        
        if not chunks:
            logger.error(f"在文档中未找到内容: {document.get('file_name', 'unknown')}")
            return []
        
        # 处理每个文本块
        for i, chunk in enumerate(chunks):
            logger.info(f"正在处理 {document.get('file_name', 'unknown')} 的第 {i+1}/{len(chunks)} 个文本块")
            try:
                qa_pairs = self._generate_qa_from_chunk(
                    chunk=chunk, 
                    prompt=prompt,
                    document_metadata={
                        'file_name': document.get('file_name', ''),
                        'file_extension': document.get('file_extension', '')
                    }
                )
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                logger.error(f"从第 {i+1} 个文本块提取问答对时出错: {e}")
        
        return all_qa_pairs
    
    def _generate_qa_from_chunk(self, chunk: str, prompt: str, document_metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        从单个文本块生成问答对。
        
        参数:
            chunk: 文本块内容
            prompt: 自定义提示词
            document_metadata: 文档的额外元数据，包含文件名和扩展名等信息
            
        返回:
            问答对列表，每个问答对包含问题、答案和原文本块
        """
        # 如果提示词中没有指定JSON格式要求，添加默认的格式说明
        if "JSON format" not in prompt and "json format" not in prompt:
            system_prompt = """
            您是一位专门从文档中生成问答对的专家。
            请从提供的文档块中提取有意义的问答对。
            重点关注关键概念、事实和重要细节。
            
            请按以下JSON格式返回问答对:
            [
                {
                    "question": "基于文档内容的问题",
                    "answer": "仅基于文档的答案"
                },
                ...
            ]
            
            指导原则:
            1. 生成涵盖内容不同方面的多样化问题
            2. 确保问题清晰，答案全面
            3. 包含事实性和概念性问题
            4. 每个答案都必须直接来自文档支持
            5. 不要编造信息或添加文档中没有的知识
            """
        else:
            system_prompt = "您是一位专门从文档中生成问答对的专家。"
        
        # 构建完整的提示词
        if not prompt.strip().endswith(":"):
            user_prompt = f"{prompt}:\n\n{chunk}"
        else:
            user_prompt = f"{prompt}\n\n{chunk}"
        
        # 使用重试机制调用API
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                content = response.choices[0].message.content
                
                # 从响应中提取JSON
                qa_pairs = self._extract_json_from_response(content)
                
                # 将原始文本块添加到每个问答对中
                for qa_pair in qa_pairs:
                    qa_pair["chunk"] = chunk
                
                return qa_pairs
                
            except Exception as e:
                logger.error(f"第 {attempt+1}/{max_retries} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避策略
                else:
                    logger.error(f"在 {max_retries} 次尝试后仍无法提取问答对")
                    raise
        
        return []  # 由于上面的raise语句，正常情况下不会执行到这里
    
    def _extract_json_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        从模型响应中提取并解析JSON。
        
        参数:
            response_text: 模型的原始文本响应
            
        返回:
            解析后的问答对列表
        """
        # 使用JsonUtils.safe_parse_json进行解析
        parsed_data = JsonUtils.safe_parse_json(response_text, debug_prefix="QA提取器")
        
        # 处理结果为空的情况
        if not parsed_data:
            logger.error(f"无法从响应中解析JSON: {response_text[:200]}..." if len(response_text) > 200 else response_text)
            
            # 回退到正则表达式提取单个QA对
            qa_pairs = []
            pattern = r'"question":\s*"([^"]*)",\s*"answer":\s*"([^"]*)"'
            matches = re.findall(pattern, response_text)
            
            for question, answer in matches:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
            
            if qa_pairs:
                return qa_pairs
            return []
        
        # 确保返回的是列表
        if isinstance(parsed_data, list):
            return parsed_data
        elif isinstance(parsed_data, dict) and any(key in parsed_data for key in ["qa", "qa_pairs", "qas", "pairs"]):
            # 处理模型可能返回 {"qa_pairs": [...]} 格式的情况
            for key in ["qa", "qa_pairs", "qas", "pairs"]:
                if key in parsed_data and isinstance(parsed_data[key], list):
                    return parsed_data[key]
        
        # 如果是单个QA对而不是列表，包装为列表
        if isinstance(parsed_data, dict) and "question" in parsed_data and "answer" in parsed_data:
            return [parsed_data]
            
        # 如果解析出的JSON不符合预期格式，返回空列表
        logger.error(f"解析的JSON不是QA对列表格式: {parsed_data}")
        return []
    
    def batch_process_documents(self, documents: List[Dict[str, Any]], prompt: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量处理多个文档。
        
        参数:
            documents: 文档字典列表，每个字典包含文档内容和元数据
            prompt: 用于QA提取的自定义提示词
            
        返回:
            字典，键为文档名，值为对应的问答对列表
        """
        results = {}
        
        for document in documents:
            file_name = document.get('file_name', 'unnamed_document')
            try:
                qa_pairs = self.extract_qa_pairs(document, prompt)
                results[file_name] = qa_pairs
                logger.info(f"已为 {file_name} 生成 {len(qa_pairs)} 个问答对")
            except Exception as e:
                logger.error(f"处理文档 {file_name} 时出错: {e}")
                results[file_name] = []
        
        return results
    
    def save_qa_pairs_to_json(self, qa_pairs: Dict[str, List[Dict[str, Any]]], output_dir: str) -> List[str]:
        """
        将问答对保存为JSON文件。
        
        参数:
            qa_pairs: 字典，键为文档名，值为问答对列表
            output_dir: 保存JSON文件的目录路径
            
        返回:
            已创建的JSON文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        for doc_name, pairs in qa_pairs.items():
            if not pairs:
                continue
                
            # 创建安全的文件名（只保留字母数字、连字符、下划线和点）
            safe_name = "".join([c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in doc_name])
            output_file = os.path.join(output_dir, f"qa_{safe_name}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            
            created_files.append(output_file)
            logger.info(f"已将 {len(pairs)} 个问答对保存到 {output_file}")
        
        return created_files 