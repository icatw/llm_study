#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2: 向量数据库功能测试
测试ChromaDB向量数据库的各项功能
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'day02_faiss_demo'))

# 导入模块
from src.day02_faiss_demo.vector_database import VectorDatabaseDemo

class TestVectorDatabaseDemo:
    """向量数据库功能测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.db = VectorDatabaseDemo(persist_directory=self.temp_dir)
        
        # 测试文档
        self.test_documents = [
            "Python是一种编程语言",
            "机器学习是AI的分支",
            "深度学习使用神经网络",
            "自然语言处理理解文本",
            "计算机视觉处理图像"
        ]
        
        self.test_metadatas = [
            {"category": "编程", "level": "初级"},
            {"category": "AI", "level": "中级"},
            {"category": "AI", "level": "高级"},
            {"category": "NLP", "level": "中级"},
            {"category": "CV", "level": "中级"}
        ]
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """测试数据库初始化"""
        assert self.db is not None
        assert hasattr(self.db, 'embedding_demo')
        assert self.db.persist_directory == self.temp_dir
    
    def test_add_documents_returns_ids(self):
        """测试添加文档返回ID列表"""
        doc_ids = self.db.add_documents(self.test_documents[:2])
        
        assert isinstance(doc_ids, list)
        assert len(doc_ids) == 2
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        assert all(doc_id.startswith('doc_') for doc_id in doc_ids)
    
    def test_add_documents_with_metadata(self):
        """测试添加带元数据的文档"""
        doc_ids = self.db.add_documents(
            self.test_documents[:3], 
            self.test_metadatas[:3]
        )
        
        assert len(doc_ids) == 3
        
        # 验证文档可以被搜索到
        results = self.db.search_similar("编程语言", n_results=1)
        assert len(results) > 0
        assert 'metadata' in results[0]
    
    def test_add_empty_documents(self):
        """测试添加空文档列表"""
        doc_ids = self.db.add_documents([])
        assert doc_ids == []
    
    def test_search_similar_returns_results(self):
        """测试相似性搜索返回结果"""
        # 先添加文档
        self.db.add_documents(self.test_documents, self.test_metadatas)
        
        # 搜索相似文档
        results = self.db.search_similar("人工智能技术", n_results=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # 检查结果格式
        for result in results:
            assert 'document' in result
            assert 'similarity' in result
            assert 'metadata' in result
            assert isinstance(result['similarity'], (int, float))
    
    def test_search_similarity_scores(self):
        """测试相似度分数的合理性"""
        # 添加文档
        self.db.add_documents(self.test_documents, self.test_metadatas)
        
        # 搜索与第一个文档完全相同的内容
        results = self.db.search_similar(self.test_documents[0], n_results=1)
        
        assert len(results) > 0
        # 相似度应该很高（接近1）
        assert results[0]['similarity'] > 0.8
    
    def test_search_results_ordering(self):
        """测试搜索结果按相似度排序"""
        # 添加文档
        self.db.add_documents(self.test_documents, self.test_metadatas)
        
        # 搜索多个结果
        results = self.db.search_similar("机器学习和AI", n_results=5)
        
        # 验证结果按相似度降序排列
        similarities = [result['similarity'] for result in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_search_with_different_n_results(self):
        """测试不同的结果数量参数"""
        # 添加文档
        self.db.add_documents(self.test_documents, self.test_metadatas)
        
        # 测试不同的n_results值
        for n in [1, 3, 5, 10]:
            results = self.db.search_similar("技术", n_results=n)
            expected_count = min(n, len(self.test_documents))
            assert len(results) <= expected_count
    
    def test_search_empty_database(self):
        """测试在空数据库中搜索"""
        results = self.db.search_similar("任何查询", n_results=5)
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_get_collection_stats(self):
        """测试获取集合统计信息"""
        # 空数据库统计
        stats = self.db.get_collection_stats()
        assert isinstance(stats, dict)
        assert 'total_documents' in stats
        assert 'database_type' in stats
        assert stats['total_documents'] == 0
        
        # 添加文档后的统计
        self.db.add_documents(self.test_documents[:3])
        stats = self.db.get_collection_stats()
        assert stats['total_documents'] == 3
    
    def test_metadata_preservation(self):
        """测试元数据保存和检索"""
        # 添加带特定元数据的文档
        test_metadata = {"author": "测试作者", "date": "2024-12-21", "type": "技术文档"}
        self.db.add_documents(["测试文档内容"], [test_metadata])
        
        # 搜索并验证元数据
        results = self.db.search_similar("测试", n_results=1)
        assert len(results) > 0
        
        retrieved_metadata = results[0]['metadata']
        assert retrieved_metadata['author'] == "测试作者"
        assert retrieved_metadata['date'] == "2024-12-21"
        assert retrieved_metadata['type'] == "技术文档"
    
    def test_chinese_text_support(self):
        """测试中文文本支持"""
        chinese_docs = [
            "人工智能是计算机科学的一个分支",
            "机器学习让计算机能够自动学习",
            "深度学习模拟人脑神经网络结构"
        ]
        
        doc_ids = self.db.add_documents(chinese_docs)
        assert len(doc_ids) == 3
        
        # 中文查询
        results = self.db.search_similar("人工智能技术", n_results=2)
        assert len(results) > 0
        assert all('人工智能' in result['document'] or 
                  '机器学习' in result['document'] or 
                  '深度学习' in result['document'] 
                  for result in results)
    
    def test_batch_operations_performance(self):
        """测试批量操作性能"""
        import time
        
        # 准备大量文档
        large_doc_set = [f"这是第{i}个测试文档，内容关于技术和编程" for i in range(50)]
        
        # 测试批量添加
        start_time = time.time()
        doc_ids = self.db.add_documents(large_doc_set)
        add_time = time.time() - start_time
        
        assert len(doc_ids) == 50
        assert add_time < 30  # 应该在30秒内完成
        
        # 测试批量搜索
        start_time = time.time()
        for i in range(10):
            results = self.db.search_similar(f"测试文档{i}", n_results=5)
            assert len(results) > 0
        search_time = time.time() - start_time
        
        assert search_time < 10  # 10次搜索应该在10秒内完成
    
    def test_delete_collection(self):
        """测试删除集合功能"""
        # 添加一些文档
        self.db.add_documents(self.test_documents[:2])
        
        # 验证文档存在
        stats_before = self.db.get_collection_stats()
        assert stats_before['total_documents'] == 2
        
        # 删除集合
        self.db.delete_collection()
        
        # 验证集合已清空
        stats_after = self.db.get_collection_stats()
        assert stats_after['total_documents'] == 0
    
    def test_persistence_across_instances(self):
        """测试跨实例的持久化"""
        # 在第一个实例中添加文档
        self.db.add_documents(self.test_documents[:2], self.test_metadatas[:2])
        
        # 创建新的数据库实例（相同目录）
        db2 = VectorDatabaseDemo(persist_directory=self.temp_dir)
        
        # 验证数据在新实例中可用
        stats = db2.get_collection_stats()
        
        # 注意：Mock数据库不支持持久化，ChromaDB支持
        if stats['database_type'] == 'ChromaDB':
            assert stats['total_documents'] == 2
            
            # 验证可以搜索到之前添加的文档
            results = db2.search_similar("编程", n_results=1)
            assert len(results) > 0

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])