#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API连接测试脚本
测试通义千问API是否配置正确
"""

import os
import dashscope
from dotenv import load_dotenv

def test_dashscope_connection():
    """测试DashScope API连接"""
    print("🔧 正在测试通义千问API连接...")
    
    # 加载环境变量
    load_dotenv()
    api_key = os.getenv('DASHSCOPE_API_KEY')
    
    if not api_key:
        print("❌ 错误：未找到DASHSCOPE_API_KEY环境变量")
        print("请检查.env文件是否正确配置")
        return False
    
    # 设置API密钥
    dashscope.api_key = api_key
    print(f"🔑 API密钥已加载: {api_key[:10]}...")
    
    try:
        # 测试文本生成
        print("\n📝 测试文本生成功能...")
        response = dashscope.Generation.call(
            model='qwen-turbo',
            prompt='你好，请回复：API连接成功！',
            max_tokens=50
        )
        
        if response.status_code == 200:
            print(f"✅ 文本生成成功: {response.output.text}")
        else:
            print(f"❌ 文本生成失败: {response.message}")
            return False
            
        # 测试文本向量化
        print("\n🔢 测试文本向量化功能...")
        embedding_response = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v1,
            input="这是一个测试文本"
        )
        
        if embedding_response.status_code == 200:
            embeddings = embedding_response.output['embeddings'][0]['embedding']
            print(f"✅ 向量化成功: 维度={len(embeddings)}, 前5个值={embeddings[:5]}")
        else:
            print(f"❌ 向量化失败: {embedding_response.message}")
            return False
            
        print("\n🎉 所有API测试通过！可以开始学习任务了。")
        return True
        
    except Exception as e:
        print(f"❌ API测试失败: {str(e)}")
        print("请检查网络连接和API密钥是否正确")
        return False

if __name__ == "__main__":
    success = test_dashscope_connection()
    if not success:
        print("\n💡 解决建议:")
        print("1. 检查.env文件中的DASHSCOPE_API_KEY是否正确")
        print("2. 确认API密钥是否已激活")
        print("3. 检查网络连接")
        print("4. 查看阿里云控制台余额是否充足")
        exit(1)
    else:
        print("\n🚀 准备开始Day 1学习任务！")
        exit(0)