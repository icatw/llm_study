#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 1: Embedding基础实战
学习目标：
1. 理解文本向量化的基本概念
2. 掌握通义千问API的embedding调用
3. 实现文本相似度计算
4. 完成简单的语义搜索

预计用时：30分钟
"""

import os
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv

# 尝试导入dashscope，如果失败则使用模拟数据
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("⚠️  DashScope未安装，将使用模拟数据进行学习")

class EmbeddingDemo:
    """Embedding基础演示类"""
    
    def __init__(self):
        """初始化"""
        load_dotenv()
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        
        if DASHSCOPE_AVAILABLE and self.api_key:
            dashscope.api_key = self.api_key
            self.use_real_api = True
            print("✅ 使用真实API进行embedding")
        else:
            self.use_real_api = False
            print("🔧 使用模拟数据进行学习（适合离线练习）")
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            向量列表
        """
        if self.use_real_api:
            try:
                response = dashscope.TextEmbedding.call(
                    model=dashscope.TextEmbedding.Models.text_embedding_v1,
                    input=text
                )
                
                if response.status_code == 200:
                    return response.output['embeddings'][0]['embedding']
                else:
                    print(f"❌ API调用失败: {response.message}")
                    return self._get_mock_embedding(text)
            except Exception as e:
                print(f"❌ API调用异常: {str(e)}")
                return self._get_mock_embedding(text)
        else:
            return self._get_mock_embedding(text)
    
    def _get_mock_embedding(self, text: str) -> List[float]:
        """生成模拟的embedding向量（用于学习和测试）
        
        Args:
            text: 输入文本
            
        Returns:
            模拟的1536维向量
        """
        # 使用文本hash作为随机种子，确保相同文本得到相同向量
        np.random.seed(hash(text) % (2**32))
        
        # 生成1536维的随机向量（与通义千问embedding维度一致）
        vector = np.random.normal(0, 1, 1536)
        
        # 归一化向量
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数 (-1到1之间)
        """
        # 转换为numpy数组
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def find_most_similar(self, query: str, documents: List[str]) -> Tuple[str, float]:
        """在文档列表中找到与查询最相似的文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            (最相似的文档, 相似度分数)
        """
        query_embedding = self.get_embedding(query)
        
        best_doc = ""
        best_score = -1.0
        
        print(f"\n🔍 查询: '{query}'")
        print("📊 相似度分析:")
        
        for doc in documents:
            doc_embedding = self.get_embedding(doc)
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            
            print(f"  📄 '{doc}' -> 相似度: {similarity:.4f}")
            
            if similarity > best_score:
                best_score = similarity
                best_doc = doc
        
        return best_doc, best_score

def main():
    """主函数 - 演示embedding的基本用法"""
    print("🚀 Day 1: Embedding基础实战")
    print("=" * 50)
    
    # 创建演示实例
    demo = EmbeddingDemo()
    
    # 1. 基础embedding演示
    print("\n📝 1. 基础文本向量化演示")
    text = "人工智能是未来的发展方向"
    embedding = demo.get_embedding(text)
    print(f"文本: '{text}'")
    print(f"向量维度: {len(embedding)}")
    print(f"向量前5个值: {embedding[:5]}")
    
    # 2. 相似度计算演示
    print("\n🔢 2. 文本相似度计算演示")
    texts = [
        "人工智能技术发展迅速",
        "AI技术正在快速进步",
        "今天天气很好",
        "机器学习是AI的重要分支"
    ]
    
    embeddings = [demo.get_embedding(text) for text in texts]
    
    print("\n相似度矩阵:")
    for i, text1 in enumerate(texts):
        for j, text2 in enumerate(texts):
            if i <= j:  # 只显示上三角矩阵
                similarity = demo.cosine_similarity(embeddings[i], embeddings[j])
                print(f"'{text1}' <-> '{text2}': {similarity:.4f}")
    
    # 3. 语义搜索演示
    print("\n🔍 3. 简单语义搜索演示")
    documents = [
        "Python是一种流行的编程语言",
        "机器学习算法可以从数据中学习模式",
        "深度学习是机器学习的一个子领域",
        "自然语言处理帮助计算机理解人类语言",
        "计算机视觉让机器能够理解图像",
        "今天的午餐很美味"
    ]
    
    queries = [
        "什么是机器学习？",
        "编程语言有哪些？",
        "如何处理文本数据？"
    ]
    
    for query in queries:
        best_doc, score = demo.find_most_similar(query, documents)
        print(f"\n✨ 最佳匹配: '{best_doc}' (相似度: {score:.4f})")
    
    print("\n🎉 Day 1 学习完成！")
    print("\n📚 今日学习要点:")
    print("1. ✅ 文本可以转换为高维向量表示")
    print("2. ✅ 相似的文本在向量空间中距离更近")
    print("3. ✅ 余弦相似度是衡量文本相似性的常用方法")
    print("4. ✅ Embedding是RAG系统的基础组件")
    
    print("\n🔗 下一步学习:")
    print("- Day 2: 向量数据库(FAISS)的使用")
    print("- 学习如何高效存储和检索大量向量")

if __name__ == "__main__":
    main()