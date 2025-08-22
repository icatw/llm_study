# 通义千问API申请指南

## 📋 申请步骤

### 1. 注册阿里云账号

1. 访问 [阿里云官网](https://www.aliyun.com/)
2. 点击右上角「免费注册」
3. 使用手机号或邮箱完成注册
4. 完成实名认证（个人认证即可）

### 2. 开通DashScope服务

1. 访问 [DashScope控制台](https://dashscope.console.aliyun.com/)
2. 首次访问会提示开通服务，点击「立即开通」
3. 阅读并同意服务协议
4. 开通成功后进入控制台

### 3. 获取API Key

1. 在DashScope控制台左侧菜单找到「API-KEY管理」
2. 点击「创建新的API-KEY」
3. 输入API Key名称（如：llm_study_key）
4. 复制生成的API Key（格式类似：sk-xxxxxxxxxx）

⚠️ **重要提醒**：
- API Key只显示一次，请立即复制保存
- 不要将API Key分享给他人或提交到代码仓库

### 4. 配置到项目中

```bash
# 1. 复制环境配置模板
cp .env.template .env

# 2. 编辑.env文件
vim .env
# 或使用其他编辑器
code .env

# 3. 将API Key填入DASHSCOPE_API_KEY字段
DASHSCOPE_API_KEY=sk-your-actual-api-key-here
```

## 💰 费用说明

### 免费额度
- 新用户注册后有一定的免费调用额度
- 通常包含数万次API调用
- 足够完成整个8周学习计划

### 计费方式
- 按Token数量计费
- Embedding模型：约0.0007元/1K tokens
- 对话模型：约0.008元/1K tokens
- 学习阶段每日费用预计<1元

### 费用控制建议
1. 在控制台设置消费限额
2. 开启余额不足提醒
3. 定期检查使用量统计

## 🔧 测试API连接

申请成功后，运行以下命令测试连接：

```bash
# 进入项目目录
cd /Users/cyberserval/PycharmProjects/llm_study

# 创建虚拟环境（如果还没有）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install dashscope python-dotenv

# 测试API连接
python -c "
import dashscope
import os
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

response = dashscope.Generation.call(
    model='qwen-turbo',
    prompt='你好，请回复：API连接成功！'
)
print(response.output.text)
"
```

如果看到"API连接成功！"的回复，说明配置正确！

## 🆘 常见问题

### Q1: 提示"API Key无效"
- 检查API Key是否正确复制（注意前后空格）
- 确认API Key是否已激活
- 检查.env文件格式是否正确

### Q2: 提示"余额不足"
- 登录阿里云控制台充值
- 或等待免费额度刷新

### Q3: 网络连接超时
- 检查网络连接
- 尝试使用代理（如果在海外）

## 📞 技术支持

- [DashScope官方文档](https://help.aliyun.com/zh/dashscope/)
- [API参考手册](https://help.aliyun.com/zh/dashscope/developer-reference/)
- 阿里云工单系统

---

**完成API申请后，请回复"API已配置"开始第一个学习任务！** 🚀