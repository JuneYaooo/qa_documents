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

# Load environment variables
load_dotenv()

# Set up logger
beijing_logger = BeijingLogger()
logger = beijing_logger.get_logger()

class QAExtractor:
    def __init__(self):
        """
        Initialize the QA Extractor with OpenAI API credentials.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"QAExtractor initialized with model: {self.model_name}")
    
    def extract_qa_pairs(self, document: Dict[str, Any], prompt: str) -> List[Dict[str, Any]]:
        """
        Extract QA pairs from a document using OpenAI.
        
        Args:
            document: Dictionary containing document content and metadata
            prompt: Custom prompt for QA extraction
            
        Returns:
            List of QA pairs in dictionary format
        """
        all_qa_pairs = []
        chunks = document.get('chunks', [])
        
        if not chunks and 'file_content' in document:
            # If no chunks but file content exists, use the whole document
            chunks = [document['file_content']]
        
        if not chunks:
            logger.error(f"No content found in document: {document.get('file_name', 'unknown')}")
            return []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} for {document.get('file_name', 'unknown')}")
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
                logger.error(f"Error extracting QA pairs from chunk {i+1}: {e}")
        
        return all_qa_pairs
    
    def _generate_qa_from_chunk(self, chunk: str, prompt: str, document_metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generate QA pairs from a single chunk of text.
        
        Args:
            chunk: Text chunk
            prompt: Custom prompt
            document_metadata: Additional metadata about the document
            
        Returns:
            List of QA pairs
        """
        # If prompt doesn't contain specific instructions for QA format, add them
        if "JSON format" not in prompt and "json format" not in prompt:
            system_prompt = """
            You are an expert at generating question-answer pairs from documents.
            Extract meaningful QA pairs from the provided document chunk.
            Focus on key concepts, facts, and important details.
            
            Return the QA pairs in the following JSON format:
            [
                {
                    "question": "The question based on document content",
                    "answer": "The answer to the question, based solely on the document"
                },
                ...
            ]
            
            Guidelines:
            1. Generate diverse questions covering different aspects of the content
            2. Ensure questions are clear and answers are comprehensive
            3. Include both factual and conceptual questions
            4. Each answer should be directly supported by the document
            5. Do not make up information or add knowledge not in the document
            """
        else:
            system_prompt = "You are an expert at generating question-answer pairs from documents."
        
        # Build the complete prompt
        if not prompt.strip().endswith(":"):
            user_prompt = f"{prompt}:\n\n{chunk}"
        else:
            user_prompt = f"{prompt}\n\n{chunk}"
        
        # Make API call with retry mechanism
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
                
                # Extract JSON from response
                qa_pairs = self._extract_json_from_response(content)
                
                # Add the original chunk to each QA pair
                for qa_pair in qa_pairs:
                    qa_pair["chunk"] = chunk
                
                return qa_pairs
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to extract QA pairs after {max_retries} attempts")
                    raise
        
        return []  # Should not reach here due to the raise above
    
    def _extract_json_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extract and parse JSON from the model's response.
        
        Args:
            response_text: Raw text response from the model
            
        Returns:
            Parsed list of QA pairs
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
        Process multiple documents in batch.
        
        Args:
            documents: List of document dictionaries
            prompt: Custom prompt for QA extraction
            
        Returns:
            Dictionary mapping document names to their QA pairs
        """
        results = {}
        
        for document in documents:
            file_name = document.get('file_name', 'unnamed_document')
            try:
                qa_pairs = self.extract_qa_pairs(document, prompt)
                results[file_name] = qa_pairs
                logger.info(f"Generated {len(qa_pairs)} QA pairs for {file_name}")
            except Exception as e:
                logger.error(f"Error processing document {file_name}: {e}")
                results[file_name] = []
        
        return results
    
    def save_qa_pairs_to_json(self, qa_pairs: Dict[str, List[Dict[str, Any]]], output_dir: str) -> List[str]:
        """
        Save QA pairs to JSON files.
        
        Args:
            qa_pairs: Dictionary mapping document names to their QA pairs
            output_dir: Directory to save the JSON files
            
        Returns:
            List of paths to created JSON files
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        for doc_name, pairs in qa_pairs.items():
            if not pairs:
                continue
                
            # Create a safe filename
            safe_name = "".join([c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in doc_name])
            output_file = os.path.join(output_dir, f"qa_{safe_name}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            
            created_files.append(output_file)
            logger.info(f"Saved {len(pairs)} QA pairs to {output_file}")
        
        return created_files 