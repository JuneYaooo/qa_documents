#!/usr/bin/env python3
"""
提取文档QA对的简化脚本。
接受一个文件或文件夹路径，解析文档，拆分段落，提取QA对，并将结果保存为JSON文件。
保存路径为 output/当前日期/原始相对路径。
"""

import os
import sys
import json
import argparse
from pathlib import Path
import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入我们的模块
from src.core import DocumentProcessor, QAExtractor
from src.utils.logger import BeijingLogger

# 加载环境变量
load_dotenv()

# 配置日志
logger_instance = BeijingLogger()
logger = logger_instance.get_logger()

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从文档中提取QA对")
    parser.add_argument(
        "input",
        type=str,
        help="要处理的输入文件或目录路径"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="保存QA对的输出目录 (默认: output)"
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=5000,
        help="文档处理的最大块大小 (默认: 5000)"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="从这段文本中提取有意义的问答对。包括事实信息和关键概念。格式化输出为包含'question','answer'字段的JSON数组。如果没有合适的内容，请返回空数组。",
        help="QA提取提示"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="递归处理目录"
    )
    return parser.parse_args()

def collect_files(input_path: str, recursive: bool = False) -> List[Dict[str, str]]:
    """
    从路径收集文件，可以是文件或目录。
    
    参数:
        input_path: 文件或目录路径
        recursive: 是否递归处理目录
        
    返回:
        文件路径信息列表，每个项目包含绝对路径和相对路径
    """
    all_files = []
    base_path = os.path.abspath(os.path.dirname(input_path) if os.path.isfile(input_path) else input_path)
    
    # 确保base_path是目录
    if os.path.isfile(base_path):
        base_path = os.path.dirname(base_path)
    
    input_path = os.path.abspath(input_path)
    
    if os.path.isfile(input_path):
        # 单个文件情况
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext in ['.pdf', '.docx', '.txt', '.md']:
            rel_path = os.path.relpath(input_path, base_path)
            all_files.append({
                'abs_path': input_path,
                'rel_path': rel_path
            })
    elif os.path.isdir(input_path):
        # 目录情况
        if recursive:
            for root, _, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in ['.pdf', '.docx', '.txt', '.md']:
                        rel_path = os.path.relpath(file_path, base_path)
                        all_files.append({
                            'abs_path': file_path,
                            'rel_path': rel_path
                        })
        else:
            # 只处理顶层目录中的文件
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in ['.pdf', '.docx', '.txt', '.md']:
                        rel_path = os.path.relpath(file_path, base_path)
                        all_files.append({
                            'abs_path': file_path,
                            'rel_path': rel_path
                        })
    
    return all_files

def main():
    """运行命令行工具的主函数。"""
    args = parse_args()
    
    # 检查OpenAI API密钥是否设置
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("环境变量中未找到OPENAI_API_KEY")
        print("错误：未找到OpenAI API密钥。请在.env文件中设置它。")
        sys.exit(1)
    
    # 获取当前日期（北京时间）
    beijing_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
    date_str = beijing_now.strftime('%Y-%m-%d')
    
    # 创建基本输出目录
    base_output_dir = os.path.join(args.output, date_str)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 收集要处理的文件
    files = collect_files(args.input, args.recursive)
    
    if not files:
        logger.error(f"在 {args.input} 中未找到要处理的文件")
        print(f"错误：在 {args.input} 中未找到要处理的文件。")
        sys.exit(1)
    
    logger.info(f"找到 {len(files)} 个文件要处理")
    print(f"处理 {len(files)} 个文件...")
    
    # 初始化文档处理器和QA提取器
    processor = DocumentProcessor(max_chunk_size=args.chunk_size)
    extractor = QAExtractor()
    
    # 处理文件并提取QA对
    total_qa_pairs = 0
    processed_docs_info = []
    
    for file_info in files:
        file_path = file_info['abs_path']
        rel_path = file_info['rel_path']
        
        try:
            # 处理文档
            logger.info(f"处理文件: {file_path}")
            print(f"处理文件: {rel_path}")
            
            doc = processor.process_single_file(file_path)
            if not doc:
                logger.warning(f"处理失败: {file_path}")
                print(f"警告: 处理失败 {rel_path}")
                continue
            
            # 提取QA对
            logger.info(f"从 {doc.get('file_name', 'unknown')} 中提取QA对")
            qa_pairs = extractor.extract_qa_pairs(doc, args.prompt)
            
            if not qa_pairs:
                logger.warning(f"从 {file_path} 中没有生成QA对")
                print(f"警告: 从 {rel_path} 中没有生成QA对")
                continue
            
            # 确定输出路径，保留原始目录结构
            rel_dir = os.path.dirname(rel_path)
            output_dir = os.path.join(base_output_dir, rel_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # 准备输出文件名
            file_name = os.path.basename(file_path)
            base_name, _ = os.path.splitext(file_name)
            output_file = os.path.join(output_dir, f"{base_name}.json")
            
            # 保存QA对到JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            
            total_qa_pairs += len(qa_pairs)
            
            logger.info(f"从 {file_path} 提取了 {len(qa_pairs)} 个QA对")
            print(f"成功: 从 {rel_path} 提取了 {len(qa_pairs)} 个QA对")
            
            # 记录处理信息用于汇总
            processed_docs_info.append({
                "file_path": rel_path,
                "chunks": len(doc.get('chunks', [])),
                "qa_pairs": len(qa_pairs)
            })
            
        except Exception as e:
            logger.error(f"处理 {file_path} 时出错: {e}", exc_info=True)
            print(f"错误: 处理 {rel_path} 时出错: {e}")
    
    # 创建汇总文件
    if processed_docs_info:
        summary = {
            "date": date_str,
            "total_documents": len(processed_docs_info),
            "total_qa_pairs": total_qa_pairs,
            "documents": processed_docs_info
        }
        
        summary_file = os.path.join(base_output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n成功！从 {len(processed_docs_info)} 个文档中提取了 {total_qa_pairs} 个QA对。")
        print(f"输出文件保存在: {os.path.abspath(base_output_dir)}")
        
        # 打印汇总表格
        print("\n汇总:")
        print("-" * 80)
        print(f"{'文档':<50} | {'段落数':<10} | {'QA对数':<10}")
        print("-" * 80)
        for doc in processed_docs_info:
            file_name = doc['file_path']
            print(f"{file_name[:47] + '...' if len(file_name) > 50 else file_name:<50} | {doc['chunks']:<10} | {doc['qa_pairs']:<10}")
        print("-" * 80)
        print(f"总计: {len(processed_docs_info)} 个文档, {total_qa_pairs} 个QA对")
    else:
        logger.error("没有成功处理任何文档")
        print("错误: 没有成功处理任何文档。")
        sys.exit(1)

if __name__ == "__main__":
    main() 