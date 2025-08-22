# AI学习系统 - RAG与Agent实战

## 项目简介

这是一个8周AI学习计划的实战项目，专注于RAG（检索增强生成）和Agent技术的学习与产品化落地。

## 学习目标

- ✅ 掌握Embedding和向量数据库技术
- ✅ 实现RAG系统的完整流程
- ✅ 学习LangChain和AutoGen框架
- ✅ 构建智能Agent应用
- ✅ 完成产品化部署

## 技术栈

- **语言**: Python 3.8+
- **AI框架**: LangChain, AutoGen
- **向量数据库**: FAISS, ChromaDB
- **API服务**: 通义千问(优先), OpenAI, DeepSeek
- **Web框架**: FastAPI, Streamlit
- **部署**: Docker

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd llm_study

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

```bash
# 复制环境配置模板
cp .env.template .env

# 编辑.env文件，填入你的API密钥
vim .env
```

### 3. 运行第一个示例

```bash
# 进入第一天的学习目录
cd src/day01_embedding_demo

# 运行embedding示例
python 01_basic_embedding.py
```

## 项目结构

```
llm_study/
├── src/                    # 源代码目录
│   ├── day01_embedding_demo/   # 第1天：Embedding基础
│   ├── day02_faiss_demo/       # 第2天：FAISS向量库
│   └── day03_langchain_demo/   # 第3天：LangChain入门
├── docs/                   # 学习文档
│   ├── INDEX.md           # 学习目录索引
│   ├── day01/             # 第1天文档
│   ├── day02/             # 第2天文档
│   └── day03/             # 第3天文档
├── tests/                  # 测试代码
│   ├── day01/             # 第1天测试
│   ├── day02/             # 第2天测试
│   └── day03/             # 第3天测试
├── config/                 # 配置文件
├── utils/                  # 工具函数
├── logs/                   # 日志文件
├── requirements.txt        # 项目依赖
├── .env.template          # 环境配置模板
└── README.md              # 项目说明
```

## 学习进度

详细的学习进度和知识点请查看 [学习目录索引](docs/INDEX.md)

## 注意事项

- 🔐 请妥善保管API密钥，不要提交到版本控制
- 💰 注意API调用费用，建议设置使用限额
- 🧪 每个示例都包含完整的测试用例
- 📚 每天学习后请查看对应的知识点文档

## 联系方式

如有问题，请提交Issue或联系项目维护者。

---

**开始你的AI学习之旅吧！** 🚀