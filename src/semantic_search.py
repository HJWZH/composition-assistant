import os
import json
import torch
import numpy as np
from sentence_transformers import util
from .model_loader import ModelLoader
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SemanticSearch")

class SemanticSearchEngine:
    def __init__(self, model_dir="model", use_fine_tuned=True, device=None):
        self.model_dir = model_dir
        self.use_fine_tuned = use_fine_tuned
        self.device = device
        
        # 加载模型
        self.model_loader = ModelLoader(model_dir)
        self.model, self.device = self.model_loader.load_model(
            use_fine_tuned=use_fine_tuned, 
            device=device
        )
        logger.info(f"模型加载完成! 设备: {self.device}")
        
        # 加载预计算嵌入
        self.embeddings = self._load_embeddings()
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        # 预计算嵌入范数以加速相似度计算
        if self.embeddings is not None:
            self.emb_norms = torch.norm(self.embeddings, dim=1, keepdim=True)
            logger.info(f"嵌入向量加载完成: {self.embeddings.shape[0]} 条, 维度: {self.embeddings.shape[1]}")
        else:
            logger.warning("未找到预计算嵌入，将使用实时编码")
    
    def _load_embeddings(self):
        """加载嵌入向量，自动处理设备"""
        embeddings_path = os.path.join(self.model_dir, "embeddings.pt")
        if not os.path.exists(embeddings_path):
            logger.warning(f"嵌入文件不存在: {embeddings_path}")
            return None
        
        # 测量加载时间
        start_time = time.time()
        
        # 加载并转换设备
        embeddings = torch.load(embeddings_path, map_location=self.device)
        
        # 确保嵌入向量在正确设备上
        if self.device.startswith("cuda") and embeddings.device.type == "cpu":
            embeddings = embeddings.to(self.device)
        
        logger.info(f"嵌入加载完成: 耗时 {time.time()-start_time:.2f}s")
        return embeddings
    
    def _load_metadata(self):
        """加载元数据"""
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(f"元数据文件不存在: {metadata_path}")
            return []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            logger.info(f"加载元数据: {len(metadata)} 条")
            return metadata
    
    def search(self, query, top_k=5, category="all", similarity_threshold=0.3):
        """语义搜索素材 - 使用预计算嵌入"""
        start_time = time.time()
        
        if self.embeddings is None:
            # 如果没有预计算嵌入，回退到实时编码
            return self._realtime_search(query, top_k, category, similarity_threshold)
        
        # 编码查询
        query_embedding = self.model.encode(
            query, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False
        )
        
        # 确保查询嵌入在正确设备上
        if self.embeddings.device != query_embedding.device:
            query_embedding = query_embedding.to(self.embeddings.device)
        
        # 使用优化的余弦相似度计算
        query_norm = torch.norm(query_embedding, keepdim=True)
        dot_products = torch.mm(query_embedding.unsqueeze(0), self.embeddings.t()).squeeze(0)
        cos_scores = dot_products / (query_norm * self.emb_norms.squeeze(1))
        
        # 获取最相关结果
        top_scores, top_indices = torch.topk(cos_scores, k=min(top_k * 3, len(cos_scores)))
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            if score < similarity_threshold:
                continue
                
            meta = self.metadata[idx]
            if category != "all" and meta['type'] != category:
                continue
                
            results.append({
                'type': meta['type'].capitalize(),
                'content': meta['content'],
                'source': meta.get('source', ''),
                'tags': meta['keywords'],
                'score': float(score)
            })
            
            if len(results) >= top_k:
                break
        
        logger.info(f"搜索完成: 查询 '{query[:20]}...', 耗时: {time.time()-start_time:.4f}s, 结果: {len(results)}条")
        return results
    
    def _realtime_search(self, query, top_k=5, category="all", similarity_threshold=0.3):
        """实时编码搜索 - 当没有预计算嵌入时使用"""
        logger.warning("使用实时编码搜索，性能可能较低")
        start_time = time.time()
        
        # 编码查询
        query_embedding = self.model.encode(
            query, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False
        )
        
        # 实时编码所有素材
        all_texts = [item['cleaned_text'] for item in self.metadata]
        embeddings = self.model.encode(
            all_texts, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False
        )
        
        # 计算相似度
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        
        # 获取最相关结果
        top_results = torch.topk(cos_scores, k=min(top_k * 3, len(cos_scores)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score < similarity_threshold:
                continue
                
            meta = self.metadata[idx]
            if category != "all" and meta['type'] != category:
                continue
                
            results.append({
                'type': meta['type'].capitalize(),
                'content': meta['content'],
                'source': meta.get('source', ''),
                'tags': meta['keywords'],
                'score': float(score)
            })
            
            if len(results) >= top_k:
                break
        
        logger.info(f"实时搜索完成: 耗时 {time.time()-start_time:.4f}s, 结果: {len(results)}条")
        return results