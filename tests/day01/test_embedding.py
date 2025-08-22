#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 1 测试用例
测试embedding基础功能
"""

import sys
import os
import pytest
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'day01_embedding_demo'))

# 导入模块
sys.path.append(os.path.join(project_root, 'src', 'day01_embedding_demo'))
from src.day01_embedding_demo.basic_embedding import EmbeddingDemo

class TestEmbeddingDemo:
    """Embedding演示测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.demo = EmbeddingDemo()
    
    def test_get_embedding_returns_list(self):
        """测试embedding返回列表"""
        text = "测试文本"
        embedding = self.demo.get_embedding(text)
        
        assert isinstance(embedding, list), "embedding应该返回列表"
        assert len(embedding) > 0, "embedding不应该为空"
    
    def test_embedding_dimension(self):
        """测试embedding维度"""
        text = "测试文本维度"
        embedding = self.demo.get_embedding(text)
        
        # 通义千问的embedding维度是1536
        assert len(embedding) == 1536, f"embedding维度应该是1536，实际是{len(embedding)}"
    
    def test_same_text_same_embedding(self):
        """测试相同文本产生相同embedding"""
        text = "相同的文本"
        embedding1 = self.demo.get_embedding(text)
        embedding2 = self.demo.get_embedding(text)
        
        # 转换为numpy数组进行比较
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # 应该完全相同
        assert np.allclose(vec1, vec2), "相同文本应该产生相同的embedding"
    
    def test_different_text_different_embedding(self):
        """测试不同文本产生不同embedding"""
        text1 = "第一个文本"
        text2 = "第二个文本"
        
        embedding1 = self.demo.get_embedding(text1)
        embedding2 = self.demo.get_embedding(text2)
        
        # 转换为numpy数组
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # 不应该完全相同
        assert not np.allclose(vec1, vec2), "不同文本应该产生不同的embedding"
    
    def test_cosine_similarity_range(self):
        """测试余弦相似度范围"""
        text1 = "人工智能"
        text2 = "机器学习"
        
        embedding1 = self.demo.get_embedding(text1)
        embedding2 = self.demo.get_embedding(text2)
        
        similarity = self.demo.cosine_similarity(embedding1, embedding2)
        
        # 余弦相似度应该在-1到1之间
        assert -1 <= similarity <= 1, f"余弦相似度应该在-1到1之间，实际值：{similarity}"
    
    def test_cosine_similarity_identical_vectors(self):
        """测试相同向量的余弦相似度"""
        text = "测试向量"
        embedding = self.demo.get_embedding(text)
        
        similarity = self.demo.cosine_similarity(embedding, embedding)
        
        # 相同向量的余弦相似度应该接近1
        assert abs(similarity - 1.0) < 1e-6, f"相同向量的余弦相似度应该是1，实际值：{similarity}"
    
    def test_find_most_similar(self):
        """测试查找最相似文档"""
        query = "人工智能技术"
        documents = [
            "AI技术发展",
            "今天天气很好",
            "机器学习算法"
        ]
        
        best_doc, score = self.demo.find_most_similar(query, documents)
        
        assert best_doc in documents, "返回的文档应该在候选列表中"
        assert isinstance(score, float), "相似度分数应该是浮点数"
        assert -1 <= score <= 1, "相似度分数应该在-1到1之间"
    
    def test_embedding_normalization(self):
        """测试embedding向量的范数计算"""
        text = "测试归一化"
        embedding = self.demo.get_embedding(text)
        
        # 计算向量的L2范数
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)
        
        # 通义千问的embedding不是预归一化的，但范数应该大于0
        assert norm > 0, f"embedding向量范数应该大于0，实际范数：{norm}"
        
        # 测试手动归一化
        normalized_vec = vec / norm
        normalized_norm = np.linalg.norm(normalized_vec)
        assert abs(normalized_norm - 1.0) < 1e-6, f"手动归一化后范数应该接近1，实际：{normalized_norm}"
    
    def test_empty_text_handling(self):
        """测试空文本处理"""
        embedding = self.demo.get_embedding("")
        
        assert isinstance(embedding, list), "空文本也应该返回embedding列表"
        assert len(embedding) == 1536, "空文本的embedding维度也应该是1536"
    
    def test_chinese_text_embedding(self):
        """测试中文文本embedding"""
        chinese_text = "这是一段中文文本，用于测试中文embedding功能"
        embedding = self.demo.get_embedding(chinese_text)
        
        assert len(embedding) == 1536, "中文文本的embedding维度应该是1536"
        assert all(isinstance(x, (int, float)) for x in embedding), "embedding应该包含数值"

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])