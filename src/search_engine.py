import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MaterialSearchEngine:
    def __init__(self):
        self.vectorizer_path = os.path.join(os.path.dirname(__file__), 'model/tfidf_vectorizer.pkl')
        self.vectorizer = self._load_vectorizer()
        self.datasets = {}
        self.indexed_data = {}
        
    def _load_vectorizer(self):
        """加载预训练的TF-IDF向量化器"""
        if os.path.exists(self.vectorizer_path):
            return joblib.load(self.vectorizer_path)
        else:
            return TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.01)
    
    def load_data(self):
        """加载并预处理数据"""
        from .date_loader import load_dataset
        self.datasets = load_dataset()
        
        # 合并所有文本用于训练TF-IDF
        all_texts = []
        for category, items in self.datasets.items():
            for item in items:
                text = f"{item['content']} {' '.join(item['keywords'])} {item.get('theme', '')}"
                all_texts.append(text)
                self.indexed_data[len(all_texts)-1] = (category, item)
        
        # 训练或加载向量器
        if not os.path.exists(self.vectorizer_path):
            self.vectorizer.fit(all_texts)
            joblib.dump(self.vectorizer, self.vectorizer_path)
        
        self.tfidf_matrix = self.vectorizer.transform(all_texts)
    
    def search(self, query, top_k=5, category="all"):
        """执行素材检索"""
        query_vec = self.vectorizer.transform([query])
        cos_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 获取最相关结果
        related_indices = cos_similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in related_indices:
            cat, item = self.indexed_data[idx]
            if category != "all" and cat != category:
                continue
            results.append({
                'type': cat.capitalize(),
                'content': item['content'],
                'source': item.get('source', ''),
                'tags': item['keywords'],
                'score': round(cos_similarities[idx], 3)
            })
        
        return results