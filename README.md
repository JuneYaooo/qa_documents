# QA Documents Extractor

从文档中提取QA (Extract Question-Answer Pairs from Documents)

## 1. 工具功能

- 调用大模型API接口，从输入的文献中提取问答对（QA对）
- 输入一篇文献的路径，输出一个JSON文件，文件中存储提取的QA对
- 支持批量处理多个文件和递归处理目录

## 2. 输入要求

### 支持的文献格式
- `.docx` - Word文档
- `.pdf` - PDF文档（**仅支持矢量文件，不支持扫描件、图片、表格等复杂格式**）
- `.txt` - 纯文本文件
- `.md` - Markdown文件

## 3. 输出要求

- 输出文件为JSON格式，存储提取的QA对
- JSON文件命名与输入文件同名，后缀改为`.json`
- 输出保存在`output/当前日期/`目录下，保持原始文件的目录结构

### JSON字段定义
```json
[
  {
    "question": "问题",
    "answer": "答案",
    "chunk": "生成QA对的相关段落"
  }
]
```

## 4. 配置方式

工具支持以下方式配置API端点、文件路径等参数：

### 配置文件 `.env`

```
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4-1106-preview

# Optional: Set to True to enable detailed logging
DEBUG=False
```

### 环境变量
- `OPENAI_API_KEY` - OpenAI API密钥
- `OPENAI_BASE_URL` - API基础URL
- `OPENAI_MODEL_NAME` - 使用的模型名称

### 命令行参数

工具使用`argparse`库，支持以下命令行参数：

```bash
# 查看帮助
python extract_qa.py --help

# 处理单个文件
python extract_qa.py path/to/document.pdf

# 处理目录中的所有文件
python extract_qa.py path/to/documents/

# 递归处理目录及其子目录
python extract_qa.py path/to/documents/ -r

# 自定义输出目录和块大小
python extract_qa.py path/to/document.pdf -o custom_output -c 1500

# 自定义提取提示
python extract_qa.py path/to/document.pdf -p "生成关于技术概念的详细QA对"
```

## 命令行选项

`extract_qa.py` 脚本支持以下选项：

- `input`: 要处理的输入文件或目录路径（必需，位置参数）
- `--output`, `-o`: 保存QA对的输出目录（默认："output"）
- `--chunk-size`, `-c`: 文档处理的最大块大小（默认：2000）
- `--prompt`, `-p`: QA提取提示（默认：生成JSON格式的问答对）
- `--recursive`, `-r`: 递归处理目录

## 项目结构

```
qa_documents/
├── .env.example          # 环境变量示例文件
├── .env                  # 环境变量文件（不被git跟踪）
├── .gitignore            # Git忽略文件
├── extract_qa.py         # 主要脚本，直接处理文档并提取QA对
├── README.md             # 本文件
├── requirements.txt      # Python依赖项
├── output/               # QA对的默认输出目录
├── logs/                 # 日志文件目录
└── src/                  # 源代码
    ├── core/             # 核心功能
    │   ├── __init__.py   # 包初始化
    │   ├── document_processor.py # 文档处理模块
    │   └── qa_extractor.py      # QA提取模块
    └── utils/            # 工具模块
        ├── __init__.py
        └── logger.py     # 北京时区日志记录器模块
```

## 使用示例

处理单个文件：
```bash
python extract_qa.py document.pdf
```

递归处理目录中的所有文件：
```bash
python extract_qa.py documents/ -r
```

自定义提示和块大小：
```bash
python extract_qa.py documents/ -c 6000 -p "从这段文本中提取有意义的问答对。包括事实信息和关键概念。格式化输出为包含'question','answer'字段的JSON数组。如果没有合适的内容，请返回空数组。"
```

## 输出结构

```
output/
└── 2023-04-15/            # 当前日期文件夹
    ├── summary.json       # 处理汇总信息
    ├── document1.json     # 根目录文件的QA结果
    └── subfolder/         # 保持原始目录结构
        └── document2.json # 子文件夹中文件的QA结果
```