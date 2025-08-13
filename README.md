# 文思引擎 - 智能作文素材检索系统

[![Python 3.10+](https://img.shields.io/badge/python-3.12%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

## ✨ 项目亮点
- **深度语义理解**：突破关键词匹配局限，理解"挫折→逆境成长"的抽象关联
- **动态学习系统**：10轮迭代训练机制，持续提升素材推荐精准度
- **多维度过滤**：类别/主题/相似度三级检索体系
- **轻量化部署**：预计算嵌入向量技术，CPU环境0.5秒响应

## 📚 素材库示例
```json
{
  "content": "真正的太空探索不是为霸权，而是为人类共同梦想",
  "source": "中国航天白皮书",
  "type": "examples",
  "theme": "科技创新",
  "keywords": ["航天精神", "人类命运共同体", "探索精神"]
}

## TODO List
- 1.素材自动更新

## 🚀 快速开始
### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动训练
```bash
python -m src.model_trainer
```

### 运行GUI
```bash
python gui_main.py
```

## 🤝 加入我们
欢迎贡献素材库或改进算法：
1. 提交PR更新`data/`目录下的JSON文件
2. 优化模型见`src/model_trainer.py`
3. 扩展主题词库在`src/semantic_search.py`的THEME_KEYWORDS
