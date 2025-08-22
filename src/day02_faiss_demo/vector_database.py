#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2: 向量数据库实战学习
使用ChromaDB作为向量数据库，学习向量存储、检索和管理

学习目标:
1. 理解向量数据库的作用和原理
2. 掌握ChromaDB的基本操作
3. 实现向量的批量存储和检索
4. 构建可持久化的向量索引
5. 学习向量数据库的性能优化
"""

import os
import sys
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from dotenv import load_dotenv

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入Day 1的embedding功能
from src.day01_embedding_demo.basic_embedding import EmbeddingDemo

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB未安装，将使用Mock数据进行演示")

class VectorDatabaseDemo:
    """向量数据库演示类"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 数据库持久化目录
        """
        self.persist_directory = persist_directory
        self.embedding_demo = EmbeddingDemo()
        self.client = None
        self.collection = None
        
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            self._init_mock_db()
    
    def _init_chromadb(self):
        """初始化ChromaDB客户端"""
        try:
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 创建或获取集合
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            
            print("✅ ChromaDB初始化成功")
            
        except Exception as e:
            print(f"❌ ChromaDB初始化失败: {e}")
            self._init_mock_db()
    
    def _init_mock_db(self):
        """初始化Mock数据库（用于演示）"""
        self.mock_vectors = {}
        self.mock_documents = {}
        self.mock_metadata = {}
        print("✅ 使用Mock向量数据库进行演示")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表
            
        Returns:
            文档ID列表
        """
        if not documents:
            return []
        
        # 生成文档ID
        doc_ids = [f"doc_{int(time.time() * 1000)}_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        if CHROMADB_AVAILABLE and self.collection:
            return self._add_documents_chromadb(documents, doc_ids, metadatas)
        else:
            return self._add_documents_mock(documents, doc_ids, metadatas)
    
    def _add_documents_chromadb(self, documents: List[str], doc_ids: List[str], metadatas: List[Dict]) -> List[str]:
        """使用ChromaDB添加文档"""
        try:
            # 获取文档的embedding
            embeddings = []
            for doc in documents:
                embedding = self.embedding_demo.get_embedding(doc)
                embeddings.append(embedding)
            
            # 添加到ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )
            
            print(f"✅ 成功添加 {len(documents)} 个文档到ChromaDB")
            return doc_ids
            
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return self._add_documents_mock(documents, doc_ids, metadatas)
    
    def _add_documents_mock(self, documents: List[str], doc_ids: List[str], metadatas: List[Dict]) -> List[str]:
        """使用Mock数据库添加文档"""
        for i, (doc_id, doc, metadata) in enumerate(zip(doc_ids, documents, metadatas)):
            # 生成mock embedding
            embedding = np.random.rand(1536).tolist()
            
            self.mock_vectors[doc_id] = embedding
            self.mock_documents[doc_id] = doc
            self.mock_metadata[doc_id] = metadata
        
        print(f"✅ 成功添加 {len(documents)} 个文档到Mock数据库")
        return doc_ids
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            相似文档列表，包含文档内容、相似度和元数据
        """
        if CHROMADB_AVAILABLE and self.collection:
            return self._search_chromadb(query, n_results)
        else:
            return self._search_mock(query, n_results)
    
    def _search_chromadb(self, query: str, n_results: int) -> List[Dict]:
        """使用ChromaDB搜索"""
        try:
            # 获取查询的embedding
            query_embedding = self.embedding_demo.get_embedding(query)
            
            # 在ChromaDB中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # ChromaDB返回距离，转换为相似度
                    'metadata': results['metadatas'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ ChromaDB搜索失败: {e}")
            return self._search_mock(query, n_results)
    
    def _search_mock(self, query: str, n_results: int) -> List[Dict]:
        """使用Mock数据库搜索"""
        if not self.mock_documents:
            return []
        
        # 获取查询embedding
        query_embedding = self.embedding_demo.get_embedding(query)
        
        # 计算与所有文档的相似度
        similarities = []
        for doc_id, doc_embedding in self.mock_vectors.items():
            similarity = self.embedding_demo.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前n_results个结果
        results = []
        for doc_id, similarity in similarities[:n_results]:
            results.append({
                'document': self.mock_documents[doc_id],
                'similarity': similarity,
                'metadata': self.mock_metadata[doc_id]
            })
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        if CHROMADB_AVAILABLE and self.collection:
            try:
                count = self.collection.count()
                return {
                    'total_documents': count,
                    'database_type': 'ChromaDB',
                    'persist_directory': self.persist_directory
                }
            except Exception as e:
                print(f"❌ 获取ChromaDB统计失败: {e}")
                # 如果ChromaDB出错，返回0
                return {
                    'total_documents': 0,
                    'database_type': 'ChromaDB (Error)',
                    'persist_directory': self.persist_directory,
                    'error': str(e)
                }
        
        # 确保mock_documents存在
        if not hasattr(self, 'mock_documents'):
            self.mock_documents = []
            
        return {
            'total_documents': len(self.mock_documents),
            'database_type': 'Mock Database',
            'persist_directory': 'Memory Only'
        }
    
    def delete_collection(self):
        """删除集合（用于清理）"""
        if CHROMADB_AVAILABLE and self.client:
            try:
                self.client.delete_collection("documents")
                print("✅ 成功删除ChromaDB集合")
            except Exception as e:
                print(f"❌ 删除集合失败: {e}")
        else:
            self.mock_vectors.clear()
            self.mock_documents.clear()
            self.mock_metadata.clear()
            print("✅ 成功清空Mock数据库")

def demo_vector_database():
    """向量数据库功能演示"""
    print("🚀 Day 2: 向量数据库实战学习")
    print("=" * 50)
    
    # 初始化向量数据库
    db = VectorDatabaseDemo()
    
    # 准备测试文档
    documents = [
        "Python是一种高级编程语言，广泛用于数据科学和机器学习",
        "机器学习是人工智能的一个重要分支，通过算法让计算机学习数据模式",
        "深度学习使用神经网络来模拟人脑的学习过程",
        "自然语言处理帮助计算机理解和生成人类语言",
        "计算机视觉让机器能够识别和理解图像内容",
        "数据科学结合统计学、编程和领域知识来从数据中提取洞察",
        "人工智能正在改变各个行业的工作方式",
        "云计算提供了可扩展的计算资源和服务",
        "区块链技术确保数据的安全性和透明性",
        "物联网连接了日常设备，创造了智能环境"
    ]
    
    # 添加元数据
    metadatas = [
        {"category": "编程语言", "difficulty": "初级"},
        {"category": "机器学习", "difficulty": "中级"},
        {"category": "深度学习", "difficulty": "高级"},
        {"category": "自然语言处理", "difficulty": "中级"},
        {"category": "计算机视觉", "difficulty": "中级"},
        {"category": "数据科学", "difficulty": "中级"},
        {"category": "人工智能", "difficulty": "初级"},
        {"category": "云计算", "difficulty": "中级"},
        {"category": "区块链", "difficulty": "高级"},
        {"category": "物联网", "difficulty": "中级"}
    ]
    
    print("\n📚 1. 批量添加文档到向量数据库")
    doc_ids = db.add_documents(documents, metadatas)
    print(f"📄 添加了 {len(doc_ids)} 个文档")
    
    # 显示数据库统计
    print("\n📊 2. 数据库统计信息")
    stats = db.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试相似性搜索
    queries = [
        "什么是人工智能？",
        "如何学习编程？",
        "数据分析的方法",
        "神经网络的原理"
    ]
    
    print("\n🔍 3. 相似性搜索测试")
    for query in queries:
        print(f"\n🔍 查询: '{query}'")
        results = db.search_similar(query, n_results=3)
        
        print("📊 搜索结果:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. 📄 '{result['document'][:50]}...'")
            print(f"     💯 相似度: {result['similarity']:.4f}")
            print(f"     🏷️  类别: {result['metadata'].get('category', 'unknown')}")
            print(f"     📈 难度: {result['metadata'].get('difficulty', 'unknown')}")
    
    # 性能测试
    print("\n⚡ 4. 性能测试")
    start_time = time.time()
    
    # 批量搜索测试
    for _ in range(10):
        db.search_similar("机器学习算法", n_results=5)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    print(f"📈 平均搜索时间: {avg_time:.4f} 秒")
    
    # 向量数据库优势演示
    print("\n🎯 5. 向量数据库优势演示")
    
    # 语义搜索 vs 关键词搜索
    semantic_query = "AI技术的应用"
    print(f"\n🔍 语义搜索: '{semantic_query}'")
    semantic_results = db.search_similar(semantic_query, n_results=3)
    
    for i, result in enumerate(semantic_results, 1):
        print(f"  {i}. 📄 '{result['document'][:60]}...'")
        print(f"     💯 相似度: {result['similarity']:.4f}")
    
    print("\n💡 注意: 即使查询中没有出现'人工智能'等确切词汇，")
    print("    向量数据库仍能理解语义并找到相关文档！")
    
    print("\n🎉 Day 2 学习完成！")
    
    print("\n📚 今日学习要点:")
    print("1. ✅ 向量数据库能高效存储和检索高维向量")
    print("2. ✅ ChromaDB提供了简单易用的向量数据库接口")
    print("3. ✅ 支持元数据存储，便于结果过滤和分析")
    print("4. ✅ 语义搜索比关键词搜索更智能和准确")
    print("5. ✅ 持久化存储确保数据不丢失")
    
    print("\n🔗 下一步学习:")
    print("- Day 3: LangChain框架入门")
    print("- 学习如何构建完整的RAG应用链路")
    
    return db

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()
    
    # 运行演示
    demo_vector_database()