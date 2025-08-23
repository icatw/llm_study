#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: LangChain基础与RAG系统构建
实现完整的检索增强生成(RAG)系统
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src' / 'day01_embedding_demo'))
sys.path.insert(0, str(project_root / 'src' / 'day02_faiss_demo'))

# 导入前面的模块
from basic_embedding import EmbeddingDemo
from vector_database import VectorDatabaseDemo

# LangChain导入
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain导入失败: {e}")
    LANGCHAIN_AVAILABLE = False

# LLM基类导入
try:
    from langchain_community.llms.base import LLM
    LLM_BASE_AVAILABLE = True
except ImportError:
    try:
        from langchain.llms.base import LLM
        LLM_BASE_AVAILABLE = True
    except ImportError:
        print("⚠️ LLM基类不可用，使用简化实现")
        LLM_BASE_AVAILABLE = False
        
        # 创建简化的LLM基类
        class LLM:
            def __init__(self):
                pass
            
            @property
            def _llm_type(self) -> str:
                return "base"
            
            def _call(self, prompt: str, stop=None) -> str:
                return "Mock response"

# 通义千问API导入
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    print("⚠️ DashScope未安装，将使用Mock LLM")
    DASHSCOPE_AVAILABLE = False


class QwenLLM(LLM):
    """通义千问LLM包装器，兼容LangChain接口"""
    
    def __init__(self, api_key: str = None, model: str = "qwen-turbo"):
        super().__init__()
        # 使用object.__setattr__避免Pydantic验证问题
        object.__setattr__(self, 'api_key', api_key or os.getenv('DASHSCOPE_API_KEY'))
        object.__setattr__(self, 'model', model)
        
        if self.api_key and DASHSCOPE_AVAILABLE:
            dashscope.api_key = self.api_key
            object.__setattr__(self, 'available', True)
        else:
            object.__setattr__(self, 'available', False)
            print("⚠️ 通义千问API不可用，将使用Mock响应")
    
    @property
    def _llm_type(self) -> str:
        return "qwen"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用通义千问API生成回答"""
        if not self.available:
            return self._mock_response(prompt)
        
        try:
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text.strip()
            else:
                print(f"❌ API调用失败: {response.message}")
                return self._mock_response(prompt)
                
        except Exception as e:
            print(f"❌ 生成回答时出错: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Mock响应，用于API不可用时"""
        if "什么是" in prompt or "介绍" in prompt:
            return "这是一个关于您询问主题的详细介绍。由于API限制，这里显示的是模拟回答。在实际应用中，这里会显示通义千问的智能回答。"
        elif "如何" in prompt or "怎么" in prompt:
            return "以下是解决您问题的步骤和方法。这是一个模拟回答，实际使用时会调用通义千问API生成更准确的回答。"
        else:
            return "基于检索到的相关文档，这里是对您问题的回答。这是模拟回答，实际会使用通义千问API生成智能回答。"


class RAGSystem:
    """完整的RAG(检索增强生成)系统"""
    
    def __init__(self, persist_directory: str = "./rag_db"):
        """初始化RAG系统"""
        self.persist_directory = persist_directory
        
        # 初始化组件
        print("🚀 初始化RAG系统...")
        
        # 1. 初始化Embedding模块
        self.embedding_demo = EmbeddingDemo()
        print("✅ Embedding模块初始化完成")
        
        # 2. 初始化向量数据库
        self.vector_db = VectorDatabaseDemo(persist_directory=persist_directory)
        print("✅ 向量数据库初始化完成")
        
        # 3. 初始化LLM
        self.llm = QwenLLM()
        print("✅ LLM初始化完成")
        
        # 4. 初始化文本分割器
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "！", "？", ";", "，", " ", ""]
            )
            print("✅ 文本分割器初始化完成")
        else:
            self.text_splitter = None
            print("⚠️ LangChain不可用，使用简单分割")
        
        # 5. 初始化提示模板
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
基于以下上下文信息，请回答用户的问题。如果上下文中没有相关信息，请说明无法从提供的信息中找到答案。

上下文信息：
{context}

用户问题：{question}

请提供详细和准确的回答：
"""
        )
        
        print("🎉 RAG系统初始化完成！")
    
    def split_text(self, text: str) -> List[str]:
        """分割文本为小块"""
        if self.text_splitter:
            # 使用LangChain的文本分割器
            docs = self.text_splitter.split_text(text)
            return docs
        else:
            # 简单分割
            sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk + sentence) < 500:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """添加文档到知识库"""
        print(f"📚 开始处理 {len(texts)} 个文档...")
        
        all_chunks = []
        all_metadatas = []
        
        for i, text in enumerate(texts):
            # 分割文本
            chunks = self.split_text(text)
            all_chunks.extend(chunks)
            
            # 为每个chunk添加元数据
            base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            for j, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'doc_id': i,
                    'chunk_id': j,
                    'chunk_count': len(chunks),
                    'original_length': len(text),
                    'chunk_length': len(chunk)
                })
                all_metadatas.append(chunk_metadata)
        
        print(f"📄 文档分割完成，共生成 {len(all_chunks)} 个文本块")
        
        # 添加到向量数据库
        doc_ids = self.vector_db.add_documents(all_chunks, all_metadatas)
        
        print(f"✅ 成功添加 {len(doc_ids)} 个文档块到知识库")
        return doc_ids
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关文档"""
        print(f"🔍 检索查询: '{query}'")
        
        results = self.vector_db.search_similar(query, n_results=top_k)
        
        print(f"📋 找到 {len(results)} 个相关文档")
        for i, result in enumerate(results, 1):
            print(f"  {i}. 相似度: {result['similarity']:.3f} | 长度: {len(result['document'])} 字符")
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """基于检索到的文档生成回答"""
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"文档{i}: {doc['document']}")
        
        context = "\n\n".join(context_parts)
        
        # 生成提示
        prompt = self.prompt_template.format(context=context, question=query)
        
        print("🤖 正在生成回答...")
        
        # 调用LLM生成回答
        answer = self.llm._call(prompt)
        
        return answer
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """完整的RAG查询流程"""
        print(f"\n🎯 RAG查询: '{question}'")
        print("=" * 50)
        
        start_time = time.time()
        
        # 1. 检索相关文档
        retrieved_docs = self.retrieve_documents(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': '抱歉，没有找到相关的文档来回答您的问题。',
                'retrieved_docs': [],
                'retrieval_time': time.time() - start_time,
                'generation_time': 0,
                'total_time': time.time() - start_time
            }
        
        retrieval_time = time.time() - start_time
        
        # 2. 生成回答
        generation_start = time.time()
        answer = self.generate_answer(question, retrieved_docs)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        result = {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': total_time
        }
        
        print(f"\n📊 查询统计:")
        print(f"  检索时间: {retrieval_time:.3f}秒")
        print(f"  生成时间: {generation_time:.3f}秒")
        print(f"  总时间: {total_time:.3f}秒")
        
        return result
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return self.vector_db.get_collection_stats()
    
    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_db.delete_collection()
        print("🗑️ 知识库已清空")


def demo_rag_system():
    """RAG系统演示"""
    print("🎉 Day 3: LangChain基础与RAG系统演示")
    print("=" * 60)
    
    # 1. 初始化RAG系统
    rag = RAGSystem(persist_directory="./day03_rag_db")
    
    # 2. 准备测试文档
    test_documents = [
        """
        人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
        AI包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。
        机器学习是AI的核心技术之一，通过算法让计算机从数据中学习模式和规律。
        深度学习使用神经网络来模拟人脑的学习过程，在图像识别、语音识别等领域取得了突破性进展。
        """,
        """
        RAG(检索增强生成)是一种结合了信息检索和文本生成的AI技术。
        RAG系统首先从大型知识库中检索相关信息，然后基于检索到的信息生成回答。
        这种方法可以提供更准确、更具体的回答，因为它基于实际的知识而不是仅仅依赖模型的参数知识。
        RAG在问答系统、聊天机器人、知识管理等领域有广泛应用。
        """,
        """
        向量数据库是专门用于存储和检索高维向量的数据库系统。
        在AI应用中，文本、图像等数据通常被转换为向量表示，然后存储在向量数据库中。
        向量数据库支持高效的相似性搜索，可以快速找到与查询向量最相似的向量。
        常见的向量数据库包括Pinecone、Weaviate、ChromaDB、FAISS等。
        """,
        """
        LangChain是一个用于构建基于大语言模型应用的框架。
        它提供了丰富的组件和工具，包括文档加载器、文本分割器、向量存储、检索器、链等。
        LangChain简化了RAG系统的构建过程，让开发者可以快速搭建智能问答系统。
        通过LangChain，可以轻松集成不同的LLM、向量数据库和其他AI服务。
        """,
        """
        Embedding是将文本、图像等数据转换为数值向量的技术。
        好的embedding能够捕捉数据的语义信息，使得语义相似的数据在向量空间中距离较近。
        常用的文本embedding模型包括Word2Vec、GloVe、BERT、OpenAI的text-embedding-ada-002等。
        在RAG系统中，embedding用于将文档和查询转换为向量，以便进行相似性搜索。
        """
    ]
    
    # 文档元数据
    metadatas = [
        {"topic": "人工智能基础", "category": "AI概念", "difficulty": "初级"},
        {"topic": "RAG技术", "category": "AI应用", "difficulty": "中级"},
        {"topic": "向量数据库", "category": "数据存储", "difficulty": "中级"},
        {"topic": "LangChain框架", "category": "开发工具", "difficulty": "中级"},
        {"topic": "Embedding技术", "category": "AI技术", "difficulty": "中级"}
    ]
    
    print("\n📚 3. 添加文档到知识库")
    doc_ids = rag.add_documents(test_documents, metadatas)
    
    print("\n📊 4. 知识库统计")
    stats = rag.get_knowledge_base_stats()
    print(f"  总文档数: {stats['total_documents']}")
    print(f"  数据库类型: {stats['database_type']}")
    
    print("\n🔍 5. RAG查询演示")
    
    # 测试查询
    test_queries = [
        "什么是RAG技术？",
        "向量数据库有什么作用？",
        "LangChain框架的主要功能是什么？",
        "深度学习和机器学习的关系？",
        "如何选择合适的embedding模型？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 查询 {i} ---")
        result = rag.query(query, top_k=2)
        
        print(f"\n💬 问题: {result['question']}")
        print(f"🤖 回答: {result['answer']}")
        print(f"\n📋 检索到的相关文档:")
        for j, doc in enumerate(result['retrieved_docs'], 1):
            print(f"  {j}. [{doc['metadata'].get('topic', 'Unknown')}] 相似度: {doc['similarity']:.3f}")
            print(f"     {doc['document'][:100]}...")
    
    print("\n🎯 6. 性能分析")
    print("RAG系统展示了以下能力:")
    print("✅ 文档智能分割和向量化存储")
    print("✅ 基于语义的文档检索")
    print("✅ 上下文感知的回答生成")
    print("✅ 端到端的问答流程")
    print("✅ 可扩展的知识库管理")
    
    print("\n🎉 Day 3 学习完成！")
    print("\n📚 今日学习要点:")
    print("1. ✅ LangChain框架的核心组件和使用方法")
    print("2. ✅ 文档加载、分割和向量化的完整流程")
    print("3. ✅ 检索增强生成(RAG)的实现原理")
    print("4. ✅ 集成前两天的成果构建端到端系统")
    print("5. ✅ 提示工程和上下文管理技巧")
    
    print("\n🔗 下一步学习:")
    print("- Day 4: 高级RAG技术(重排序、多轮对话)")
    print("- Day 5: Agent系统构建")
    
    return rag


if __name__ == "__main__":
    # 运行演示
    rag_system = demo_rag_system()