import json
import os
import re
import jieba
import logging
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataProcessor")

class DataProcessor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, 'data')
        self.stopwords = self._load_stopwords()
        jieba.setLogLevel(logging.INFO)
        
    def _load_stopwords(self):
        """加载中文停用词表"""
        stopwords = set([
            "的", "了", "和", "是", "就", "都", "而", "及", "与", "等", "在", "这", 
            "有", "以", "于", "之", "为", "对", "中", "下", "后", "由", "来", "到", 
            "去", "上", "出", "要", "但", "从", "并", "也", "又", "或", "一个", "没有"
        ])
        return stopwords
    
    def clean_text(self, text):
        """清理文本 - 更健壮的实现"""
        # 移除特殊字符和标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)  # 保留中文字符
        
        # 分词
        try:
            words = jieba.cut(text)
        except:
            words = text.split()
        
        # 过滤停用词和单字
        words = [word.strip() for word in words if word.strip()]
        words = [word for word in words if word not in self.stopwords and len(word) > 1]
        
        return ' '.join(words)

    def load_and_preprocess(self):
        """加载并预处理所有素材"""
        datasets = defaultdict(list)
        file_types = ['quotes', 'examples', 'poems']
        
        for file_type in file_types:
            file_path = os.path.join(self.data_dir, f"{file_type}.json")
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        # 创建模型输入文本
                        text = f"{item['content']} [SEP] {' '.join(item['keywords'])}"
                        if 'theme' in item:
                            text += f" [SEP] {item['theme']}"
                        
                        # 添加清理后的文本
                        item['cleaned_text'] = self.clean_text(text)
                        datasets[file_type].append(item)
                logger.info(f"成功加载 {len(data)} 条 {file_type} 数据")
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        
        return datasets