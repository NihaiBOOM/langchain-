# 扫地机器人智能客服助手

基于 LangChain + LangGraph + RAG 的企业级智能问答 Agent，集成向量检索、BM25 混合召回、Reranker 重排和 ReAct 推理框架。

## 技术栈

- **Agent 框架**: LangChain 1.4 + LangGraph 1.2（ReAct 模式）
- **RAG**: ChromaDB 向量库 + BM25 关键词混合召回
- **Embedding & LLM**: 阿里通义千问 DashScope
- **前端**: Streamlit
- **文本处理**: jieba 分词 + RecursiveCharacterTextSplitter

## 项目结构

```
rag_agent实战项目/
├── agent/                      # ReAct Agent 核心
│   ├── react_agent.py          # Agent 主类（create_agent）
│   └── tools/
│       ├── agent_tools.py      # Agent 可用工具（RAG 检索、天气等）
│       └── middleware.py        # 模型调用中间件（日志、监控）
├── rag/                        # RAG 服务
│   ├── rag_service.py          # RAG 问答服务
│   └── vector_store.py         # 向量库 + BM25 混合检索
├── model/
│   └── factory.py              # LLM / Embedding 模型工厂
├── utils/                      # 工具模块
│   ├── config_handler.py       # YAML 配置加载
│   ├── file_handler.py         # PDF / TXT 文件解析
│   ├── logger_handler.py       # 日志
│   ├── path_tool.py            # 路径工具
│   └── prompt_loader.py        # Prompt 模板加载
├── config/                     # YAML 配置文件
│   ├── agent.yml
│   ├── chroma.yml
│   ├── prompts.yml
│   └── rag.yml
├── prompts/                    # Prompt 模板（.txt）
├── data/external/              # 知识库原始文档
├── app.py                      # Streamlit 前端入口
├── main.py                     # 主程序（当前为空）
└── requirements.txt
```

## 核心特性

- **ReAct Agent**: 思考（Thought）→ 行动（Action）→ 观察（Observation）循环
- **混合召回**: ChromaDB 向量相似度 + BM25 关键词检索并行
- **Reranker 重排**: 对混合召回结果进行精排，提升 Top-K 质量
- **流式输出**: Streamlit chat 界面 + Agent 流式响应
- **中间件监控**: 工具调用日志、模型调用前后 Hook、Prompt 切换报告

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/NihaiBOOM/langchain-.git
cd langchain-
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. 配置 API Key

本项目通过环境变量 `DASHSCOPE_API_KEY` 读取通义千问 API Key，**不会硬编码在代码或 yml 中**。

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# Windows CMD
set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Linux / macOS
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

模型名称在 `config/rag.yml` 中配置（默认 `qwen3-max` + `text-embedding-v4`）。

### 4. 启动 Streamlit 应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

## 环境要求

- Python 3.12+
- Windows / Linux / macOS