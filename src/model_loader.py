import os
import torch
from sentence_transformers import SentenceTransformer
import logging
import warnings

# 忽略transformers的某些警告
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelLoader")

class ModelLoader:
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.pretrained_path = os.path.join(model_dir, "pretrained")
        self.fine_tuned_path = os.path.join(model_dir, "fine_tuned")
        
    def load_model(self, use_fine_tuned=True, device=None):
        """加载模型并自动选择设备"""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 优先加载微调模型
        if use_fine_tuned and os.path.exists(self.fine_tuned_path):
            logger.info(f"加载微调模型: {self.fine_tuned_path}")
            model = SentenceTransformer(self.fine_tuned_path, device=device)
        elif os.path.exists(self.pretrained_path):
            logger.info(f"加载预训练模型: {self.pretrained_path}")
            model = SentenceTransformer(self.pretrained_path, device=device)
        else:
            raise FileNotFoundError(f"未找到模型文件，请检查目录: {self.pretrained_path}")
        
        # 优化模型
        model = self.optimize_model(model, device)
        
        return model, device
    
    def optimize_model(self, model, device):
        """优化模型性能 - 检索优化版"""

        # 启用TF32支持（如果可用）
        if device == "cuda":
            if torch.cuda.is_available():
                # 自动检测GPU能力
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 7:  # Volta及更新架构
                    torch.backends.cuda.matmul.allow_tf32 = True
                    logger.info("已启用TF32加速")
                
                # 启用cudnn基准测试
                torch.backends.cudnn.benchmark = True
                logger.info("已启用cudnn基准测试")
        
        # 编译模型 (PyTorch 2.0+)
        #if hasattr(torch, 'compile') and device == "cuda":
        #    try:
        #        model = torch.compile(
        #            model, 
        #            mode="reduce-overhead",
        #            fullgraph=False,
        #            dynamic=False
        #        )
        #        logger.info("已启用模型编译优化")
        #    except Exception as e:
        #        logger.warning(f"模型编译失败: {str(e)}")

        # 启用推理模式
        model.eval()
        
        return model
    
    def save_model(self, model):
        """保存微调后的模型"""
        os.makedirs(self.fine_tuned_path, exist_ok=True)
        model.save(self.fine_tuned_path)
        logger.info(f"模型已保存到 {self.fine_tuned_path}")