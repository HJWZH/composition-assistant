import os
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from .semantic_search import SemanticSearchEngine
from .model_loader import ModelLoader

class ModelLoaderThread(QThread):
    """后台加载模型的线程"""
    loaded = pyqtSignal(object, bool, bool)
    error = pyqtSignal(str)
    
    def __init__(self, model_dir, use_fine_tuned):
        super().__init__()
        self.model_dir = model_dir
        self.use_fine_tuned = use_fine_tuned
    
    def run(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            engine = SemanticSearchEngine(
                model_dir=self.model_dir,
                use_fine_tuned=self.use_fine_tuned,
                device=device
            )
            
            has_fine_tuned = os.path.exists(os.path.join(self.model_dir, "fine_tuned"))
            has_embeddings = os.path.exists(os.path.join(self.model_dir, "embeddings.pt"))
            
            self.loaded.emit(engine, has_fine_tuned, has_embeddings)
        except Exception as e:
            self.error.emit(str(e))

class GUIInterface:
    def __init__(self, model_dir="model", use_fine_tuned=True):
        self.model_dir = model_dir
        self.use_fine_tuned = use_fine_tuned
        self.engine = None
        
    def load_model_async(self):
        """异步加载模型"""
        self.loader_thread = ModelLoaderThread(self.model_dir, self.use_fine_tuned)
        return self.loader_thread
    
    def search(self, query, category="all", top_k=5, similarity_threshold=0.0):
        """执行搜索"""
        if not self.engine:
            return []
            
        return self.engine.search(
            query, 
            top_k=top_k, 
            category=category,
            similarity_threshold=similarity_threshold
        )